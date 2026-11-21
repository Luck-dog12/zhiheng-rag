import asyncio
import json
import math
import threading
from uuid import UUID

from rag.config import Settings
from rag.errors import AppError


def uuid_list(values: list[str]) -> str:
    return json.dumps([str(UUID(value)) for value in values])


class MilvusIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self._lock = threading.Lock()

    def configured(self):
        if not self.settings.milvus_uri:
            raise AppError("missing_configuration", "缺少 MILVUS_URI 配置", 503)
        if not self.settings.milvus_uri.startswith(("http://", "https://")):
            raise AppError("invalid_configuration", "Milvus 必须使用远程 HTTP(S) 地址", 503)

    def client(self):
        with self._lock:
            if self._client is None:
                from pymilvus import MilvusClient

                self.configured()
                s = self.settings
                self._client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token.get_secret_value(),
                                            db_name=s.milvus_database, timeout=30)
        return self._client

    async def _run(self, operation, *args, **kwargs):
        try:
            return await asyncio.to_thread(operation, *args, **kwargs)
        except AppError:
            raise
        except Exception:
            raise AppError("milvus_unavailable", "Milvus 操作失败，请检查连接、集合及服务能力", 502) from None

    async def initialize(self):
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        def create():
            c, s = self.client(), self.settings
            if c.has_collection(s.milvus_collection):
                return
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False,
                                               description=s.index_fingerprint)
            for name in ("chunk_id", "kb_id", "document_id", "version_id"):
                schema.add_field(name, DataType.VARCHAR, max_length=36, is_primary=name == "chunk_id")
            schema.add_field("body", DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                             analyzer_params={"type": "chinese"})
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=s.embedding_dimension)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            if s.embedding_store_sparse:
                schema.add_field("model_sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(Function(name="body_bm25", function_type=FunctionType.BM25,
                                         input_field_names=["body"], output_field_names=["sparse"]))
            indexes = c.prepare_index_params()
            indexes.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
            indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
            if s.embedding_store_sparse:
                indexes.add_index("model_sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            c.create_collection(s.milvus_collection, schema=schema, index_params=indexes,
                                consistency_level="Strong")
        await self._run(create)
        await self.check()

    async def check(self):
        from pymilvus import DataType

        def validate():
            c, s = self.client(), self.settings
            info = c.describe_collection(s.milvus_collection)
            fields = {f["name"]: f for f in info["fields"]}
            if info.get("description") != s.index_fingerprint:
                raise AppError("index_model_mismatch", "嵌入配置与集合指纹不一致，请新建集合并重新入库", 503)
            if not {"chunk_id", "kb_id", "document_id", "version_id", "body", "dense", "sparse"} <= fields.keys():
                raise AppError("index_schema_mismatch", "Milvus 集合字段不符合 RAG schema", 503)
            if int(fields["dense"].get("params", {}).get("dim", 0)) != s.embedding_dimension:
                raise AppError("index_dimension_mismatch", "Milvus 向量维度不匹配", 503)
            if s.embedding_store_sparse and ("model_sparse" not in fields
                                            or fields["model_sparse"]["type"] != DataType.SPARSE_FLOAT_VECTOR):
                raise AppError("index_schema_mismatch", "集合缺少模型稀疏向量字段，请新建集合并重新入库", 503)
            # A lexical probe verifies the server actually supports its BM25 function.
            c.search(s.milvus_collection, data=["能力检查"], anns_field="sparse", limit=1,
                     search_params={"metric_type": "BM25"})
            if s.embedding_store_sparse:
                c.search(s.milvus_collection, data=[{0: 1.0}], anns_field="model_sparse", limit=1,
                         search_params={"metric_type": "IP"})
        await self._run(validate)

    async def upsert(self, rows: list[dict]):
        await self._run(lambda: self.client().upsert(self.settings.milvus_collection, data=rows))

    async def verify(self, ids: list[str]):
        for offset in range(0, len(ids), 200):
            batch = ids[offset:offset + 200]
            rows = await self._run(lambda: self.client().query(
                self.settings.milvus_collection, filter=f"chunk_id in {uuid_list(batch)}",
                output_fields=["chunk_id"], consistency_level="Strong", limit=len(batch)))
            if {x["chunk_id"] for x in rows} != set(batch):
                raise AppError("index_incomplete", "Milvus 索引尚未完整可读", 502)

    async def search(self, question: str, vector: list[float], kb_ids: list[str], limit: int,
                     *, model_sparse: dict[int, float] | None = None, sparse_mode: str | None = None,
                     strategy: str = "hybrid") -> dict:
        if strategy not in ("hybrid", "dense", "sparse"):
            raise AppError("invalid_strategy", "检索策略无效")
        mode = sparse_mode or self.settings.retrieval_sparse_mode
        if mode not in ("bm25", "model"):
            raise AppError("invalid_sparse_mode", "稀疏检索模式无效")
        if strategy != "dense" and mode == "model" and (not self.settings.embedding_store_sparse or not model_sparse):
            raise AppError("missing_sparse_vector", "模型稀疏检索需要完整索引及查询稀疏向量", 503)
        if not kb_ids:
            return {"dense": [], "sparse": []}
        expr = f"kb_id in {uuid_list(kb_ids)}"

        async def path(field, value, metric):
            result = await self._run(lambda: self.client().search(
                self.settings.milvus_collection, data=[value], anns_field=field, filter=expr,
                limit=limit, output_fields=["chunk_id"], consistency_level="Strong",
                search_params={"metric_type": metric}))
            try:
                hits = []
                for hit in result[0]:
                    # Recent pymilvus uses the schema's primary field name; older versions return id.
                    chunk_id = hit["chunk_id"] if "chunk_id" in hit else hit["id"]
                    score = float(hit["distance"])
                    if not math.isfinite(score):
                        raise ValueError("Invalid score")
                    hits.append({"chunk_id": str(UUID(str(chunk_id))), "score": score})
                return hits
            except (KeyError, IndexError, TypeError, ValueError):
                raise AppError("milvus_protocol", "Milvus 检索结果缺少有效分块 ID 或分数", 502) from None

        calls = {}
        if strategy != "sparse":
            calls["dense"] = path("dense", vector, "COSINE")
        if strategy != "dense":
            calls["sparse"] = path("model_sparse", model_sparse, "IP") if mode == "model" else path("sparse", question, "BM25")
        return dict(zip(calls, await asyncio.gather(*calls.values())))

    async def delete_document(self, document_id: str):
        expr = f"document_id == {json.dumps(str(UUID(document_id)))}"
        await self._run(lambda: self.client().delete(self.settings.milvus_collection, filter=expr))

    async def delete_versions(self, version_ids: list[str]):
        if version_ids:
            await self._run(lambda: self.client().delete(self.settings.milvus_collection,
                                                        filter=f"version_id in {uuid_list(version_ids)}"))

    async def close(self):
        if self._client:
            await asyncio.to_thread(self._client.close)
