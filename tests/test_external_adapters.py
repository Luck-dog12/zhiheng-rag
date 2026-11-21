import base64
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from botocore.exceptions import ReadTimeoutError
from pymilvus import FunctionType, MilvusClient
from rag.config import Settings
from rag.errors import AppError
from rag.storage import S3Storage
from rag.vector import MilvusIndex


@pytest.mark.parametrize("signature,style", [("s3v4", "path"), ("s3", "virtual")])
def test_storage_passes_provider_signing_and_addressing_configuration(monkeypatch, signature, style):
    calls = []
    client = SimpleNamespace(meta=SimpleNamespace(events=SimpleNamespace(register=lambda *args: None)))
    monkeypatch.setattr("rag.storage.boto3.client", lambda *args, **kwargs: calls.append(kwargs) or client)
    storage = S3Storage(Settings(_env_file=None, s3_endpoint_url="https://cos.example.com",
        s3_bucket="test", s3_access_key="test-key", s3_secret_key="test-secret",
        s3_signature_version=signature, s3_addressing_style=style))
    assert storage.client is not None
    assert len(calls) == 1
    assert calls[0]["endpoint_url"] == "https://cos.example.com"
    assert calls[0]["config"].signature_version == signature
    assert calls[0]["config"].s3["addressing_style"] == style


def test_bulk_delete_has_content_md5_for_exact_serialized_xml_without_network():
    storage = S3Storage(Settings(_env_file=None, s3_endpoint_url="https://storage.example",
        s3_bucket="test", s3_access_key="test-key", s3_secret_key="test-secret"))
    requests = []

    class StopBeforeNetwork(Exception):
        pass

    def capture(request, **kwargs):
        requests.append(request)
        raise StopBeforeNetwork

    storage.client.meta.events.register("before-send.s3.DeleteObjects", capture)
    try:
        with pytest.raises(StopBeforeNetwork):
            storage.client.delete_objects(Bucket="test", Delete={"Objects": [{"Key": "中文/a&b.txt"}]})
        assert len(requests) == 1
        request = requests[0]
        assert request.headers["Content-MD5"] == base64.b64encode(
            hashlib.md5(request.body, usedforsecurity=False).digest())
        assert b"a&amp;b.txt" in request.body
    finally:
        storage.client.close()


class MilvusContractStub:
    """Exercise the real pymilvus schema builder without a network connection."""
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.schema, self.indexes, self.searches = None, None, []
        self.hit_id = str(uuid4())

    def has_collection(self, name):
        return self.schema is not None

    def create_collection(self, name, *, schema, index_params, consistency_level):
        assert consistency_level == "Strong"
        self.schema, self.indexes = schema.to_dict(), [item.to_dict() for item in index_params]

    def describe_collection(self, name):
        return self.schema

    def search(self, name, **kwargs):
        self.searches.append(kwargs)
        return [[{"id": self.hit_id, "distance": .75}]]


async def test_milvus_native_schema_and_both_paths_use_authorized_filter():
    settings = Settings(_env_file=None, milvus_uri="https://milvus.example")
    index, client = MilvusIndex(settings), MilvusContractStub()
    index._client = client
    await index.initialize()
    fields = {f["name"]: f for f in client.schema["fields"]}
    assert fields["dense"]["params"]["dim"] == 1024
    assert json.loads(fields["body"]["params"]["analyzer_params"]) == {"type": "chinese"}
    assert client.schema["functions"][0]["type"] == FunctionType.BM25
    assert {i["metric_type"] for i in client.indexes} == {"BM25", "COSINE"}
    client.searches.clear()
    allowed = str(uuid4())
    result = await index.search("编号 ATLAS-A100", [1.] * 1024, [allowed], 50)
    assert result["dense"][0]["chunk_id"] == client.hit_id
    assert len(client.searches) == 2
    assert {s["anns_field"] for s in client.searches} == {"dense", "sparse"}
    assert all(json.loads(s["filter"].split(" in ")[1]) == [allowed] for s in client.searches)
    assert all(s["limit"] == 50 and s["consistency_level"] == "Strong" for s in client.searches)


@pytest.mark.parametrize("strategy,field", [("dense", "dense"), ("sparse", "sparse")])
async def test_selected_milvus_strategy_calls_only_the_selected_index(strategy, field):
    index, client = MilvusIndex(Settings(_env_file=None)), MilvusContractStub()
    index._client = client
    allowed = str(uuid4())
    result = await index.search("ATLAS-A100", [1.], [allowed], 50, strategy=strategy)
    assert set(result) == {strategy} and len(client.searches) == 1
    assert client.searches[0]["anns_field"] == field
    assert json.loads(client.searches[0]["filter"].split(" in ")[1]) == [allowed]


@pytest.mark.parametrize("primary_key", ["chunk_id", "id"])
async def test_milvus_search_supports_schema_primary_name_and_legacy_id(primary_key):
    chunk_id = str(uuid4())
    client = SimpleNamespace(search=lambda *args, **kwargs: [[{primary_key: chunk_id, "distance": .5}]])
    index = MilvusIndex(Settings(_env_file=None))
    index._client = client
    result = await index.search("问题", [1.], [str(uuid4())], 50)
    assert result == {"dense": [{"chunk_id": chunk_id, "score": .5}],
                      "sparse": [{"chunk_id": chunk_id, "score": .5}]}


@pytest.mark.parametrize("hit", [{"distance": .5}, {"chunk_id": "invalid", "distance": .5},
                                   {"chunk_id": str(uuid4()), "distance": float("nan")}])
async def test_milvus_invalid_search_hit_is_a_sanitized_service_error(hit):
    index = MilvusIndex(Settings(_env_file=None))
    index._client = SimpleNamespace(search=lambda *args, **kwargs: [[hit]])
    with pytest.raises(AppError) as error:
        await index.search("问题", [1.], [str(uuid4())], 50)
    assert error.value.code == "milvus_protocol"


@pytest.mark.parametrize("mismatch,expected", [("model", "index_model_mismatch"), ("dimension", "index_dimension_mismatch")])
async def test_existing_milvus_collection_is_not_silently_reused(mismatch, expected):
    index, client = MilvusIndex(Settings(_env_file=None)), MilvusContractStub()
    index._client = client
    await index.initialize()
    if mismatch == "model":
        client.schema["description"] = "different-model"
    else:
        next(f for f in client.schema["fields"] if f["name"] == "dense")["params"]["dim"] = 512
    with pytest.raises(AppError) as error:
        await index.check()
    assert error.value.code == expected


async def test_milvus_rejects_filter_injection_before_network():
    index, client = MilvusIndex(Settings(_env_file=None)), MilvusContractStub()
    index._client = client
    with pytest.raises(ValueError):
        await index.search("问题", [1.], ['" or kb_id != "'], 50)
    assert client.searches == []


async def test_streamed_s3_read_failure_is_retryable_and_body_is_closed():
    class Body:
        closed = False

        def read(self):
            raise ReadTimeoutError(endpoint_url="https://storage.example/private-token")

        def close(self):
            self.closed = True

    body = Body()
    storage = S3Storage(Settings(_env_file=None, s3_bucket="test"))
    storage.__dict__["client"] = SimpleNamespace(get_object=lambda **kwargs: {"Body": body})
    with pytest.raises(AppError) as error:
        await storage.get("document")
    assert error.value.code == "storage_unavailable" and "private-token" not in str(error.value)
    assert body.closed
