import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from rag.config import Settings
from rag.errors import AppError
from rag.models import User
from rag.providers import QUERY_INSTRUCTION, RemoteModels
from rag.vector import MilvusIndex

from scripts.compare_retrieval import compare_questions, measure_ranking, prepare_corpus, summarize
from tests.fakes import FakeIndex, FakeModels
from tests.test_external_adapters import MilvusContractStub


def dashscope(handler, **settings):
    config = Settings(_env_file=None, embedding_protocol="dashscope", embedding_endpoint="https://example.test/native-embedding",
                      embedding_api_key="test", embedding_store_sparse=True, embedding_dimension=3,
                      rerank_endpoint="https://example.test/native-rerank", rerank_api_key="test", **settings)
    return RemoteModels(config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_dashscope_combined_vectors_batches_and_native_query_instruction():
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        texts = body["input"]["texts"]
        rows = [{"index": i, "embedding": [int(text), 1, 0], "sparse_embedding": [{"index": 100 + int(text), "value": .5}]}
                for i, text in enumerate(texts)]
        return httpx.Response(200, json={"output": {"embeddings": list(reversed(rows))}})
    model = dashscope(handler)
    result = await model.embed_vectors([str(i) for i in range(45)], include_sparse=True)
    assert [len(r["input"]["texts"]) for r in requests] == [20, 20, 5]
    assert [v[0] for v in result.dense] == list(range(45))
    assert result.sparse[44] == {144: .5}
    assert all(r["parameters"] == {"text_type": "document", "dimension": 3, "output_type": "dense&sparse"} for r in requests)
    await model.embed_vectors(["3"], query=True, include_sparse=True)
    assert requests[-1]["input"]["texts"] == ["3"]
    assert requests[-1]["parameters"]["instruct"] == QUERY_INSTRUCTION
    assert requests[-1]["parameters"]["text_type"] == "query"
    await model.embed(["3"], query=True)
    assert requests[-1]["parameters"]["output_type"] == "dense"
    await model.close()


@pytest.mark.parametrize("sparse", [None, [], [{"index": 1, "value": 0}], [{"index": -1, "value": 1}],
    [{"index": 1, "value": -1}], [{"index": 1, "value": True}], [{"index": 2 ** 32 - 1, "value": 1}],
    [{"index": 1, "value": 1}, {"index": 1, "value": 2}]])
async def test_invalid_sparse_output_is_not_silently_treated_as_bm25(sparse):
    model = dashscope(lambda request: httpx.Response(200, json={"output": {"embeddings": [
        {"index": 0, "embedding": [1, 0, 0], "sparse_embedding": sparse}]}}))
    with pytest.raises(AppError) as error:
        await model.embed_vectors(["文本"], include_sparse=True)
    assert error.value.code == "embedding_mismatch"
    await model.close()


async def test_native_rerank_uses_nested_protocol_and_returned_order():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"output": {"results": [
            {"index": 0, "relevance_score": .2}, {"index": 1, "relevance_score": .9}]}})
    model = dashscope(handler, rerank_protocol="dashscope")
    assert await model.rerank("问题", ["a", "b"]) == [(1, .9), (0, .2)]
    assert requests[0]["input"] == {"query": "问题", "documents": ["a", "b"]}
    assert requests[0]["parameters"] == {"top_n": 2}
    await model.close()


async def test_model_sparse_index_uses_ip_and_same_authorization_filter():
    config = Settings(_env_file=None, embedding_protocol="dashscope", embedding_store_sparse=True)
    index, stub = MilvusIndex(config), MilvusContractStub()
    index._client = stub
    await index.initialize()
    assert {i["metric_type"] for i in stub.indexes} == {"COSINE", "BM25", "IP"}
    stub.searches.clear()
    kb = str(uuid4())
    await index.search("问题", [1.] * 1024, [kb], 50, sparse_mode="model", model_sparse={127: .8})
    assert len(stub.searches) == 2
    sparse = next(s for s in stub.searches if s["anns_field"] == "model_sparse")
    assert sparse["data"] == [{127: .8}] and sparse["search_params"] == {"metric_type": "IP"}
    assert all(json.loads(s["filter"].split(" in ")[1]) == [kb] for s in stub.searches)
    with pytest.raises(AppError, match="稀疏"):
        await index.search("问题", [1.], [kb], 50, sparse_mode="model")


def test_sparse_switch_reuses_same_index_but_protocol_and_storage_changes_do_not():
    a = Settings(_env_file=None, embedding_protocol="dashscope", embedding_store_sparse=True)
    b = Settings(_env_file=None, embedding_protocol="dashscope", embedding_store_sparse=True, retrieval_sparse_mode="model")
    assert a.index_fingerprint == b.index_fingerprint
    assert a.processing_fingerprint == b.processing_fingerprint
    assert a.index_fingerprint != Settings(_env_file=None, embedding_protocol="dashscope").index_fingerprint
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_sparse_mode="model")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_store_sparse=True)
    assert Settings(_env_file=None, embedding_dimension=2560).embedding_dimension == 2560


async def test_pipeline_stores_model_sparse_and_retrieval_keeps_permissions(harness):
    h = harness
    h.settings.embedding_protocol = "dashscope"
    h.settings.embedding_store_sparse = True
    h.settings.retrieval_sparse_mode = "model"
    await h.upload("可乐是一种饮料，每瓶330毫升。")
    await h.upload("可乐内部成本为1元。", kb=h.private_kb)
    await h.drain()
    assert all(row.get("model_sparse") for row in h.services.index.rows.values())
    async with h.database.sessions() as db:
        user = await db.get(User, h.member_id)
        response = await h.services.retriever.search(db, user, "可乐", [h.kb])
    assert response["sparse_mode"] == "model"
    assert response["hits"] and all(hit["kb_id"] == h.kb for hit in response["hits"])
    assert h.services.index.searches[-1]["sparse_mode"] == "model"


async def test_comparison_uses_same_corpus_both_paths_and_scores_real_evidence():
    root = Path(__file__).resolve().parents[1]
    corpus = json.loads((root / "evals/corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((root / "evals/questions.json").read_text(encoding="utf-8"))[:2]
    settings = Settings(_env_file=None)
    rows, lookup, digest = prepare_corpus(corpus, questions, settings)
    again, _, other_digest = prepare_corpus(corpus, questions, settings)
    assert rows == again and digest == other_digest
    model, index = FakeModels(), FakeIndex()
    vectors = await model.embed_vectors([r["body"] for r in rows], include_sparse=True)
    await index.upsert([{**r, "dense": vectors.dense[i], "model_sparse": vectors.sparse[i]} for i, r in enumerate(rows)])
    results = await compare_questions(settings, model, index, rows, lookup, questions)
    assert [s["sparse_mode"] for s in index.searches] == ["bm25", "model", "model", "bm25"]
    scores = summarize(results)
    assert scores["bm25"]["recall_candidates"] == 1
    assert scores["model"]["reranked_ndcg_at_5"] == 1
    assert scores["reranked_paired_questions"] == {"model_wins": 0, "bm25_wins": 0, "ties": 2}
    assert measure_ranking(["a"], {"a": {"document_id": "travel", "body": "无关正文"}}, questions[0])["ndcg_at_5"] == 0
