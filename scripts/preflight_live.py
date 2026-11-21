"""Real dependency probes, with sanitized results; this is NOT end-to-end RAG acceptance.

Run from the project root: uv run python scripts/preflight_live.py
Performs small billable model requests and a temporary object-storage round trip.
"""
import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from rag.config import Settings
from rag.db import Database
from rag.errors import AppError
from rag.processing import Piece, parse_document, resolve_document, summarize
from rag.schemas import RewriteOutput
from rag.services import Services
from sqlalchemy import text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


async def main():
    settings = Settings()
    services, database = Services(settings), Database(settings)
    report = {
        "mode": "live_external_dependencies",
        "started_at": datetime.now(UTC).isoformat(),
        "rag_end_to_end_completed": False,
        "models": {name: getattr(settings, name + "_model") for name in
                   ("deepseek", "coreference", "embedding", "rerank")},
        "checks": {},
    }
    report["models"]["coreference"] = (settings.deepseek_model if settings.coreference_use_deepseek
                                        else settings.coreference_model)
    report["coreference_uses_deepseek"] = settings.coreference_use_deepseek
    report["coreference_thinking"] = (settings.coreference_deepseek_thinking if settings.coreference_use_deepseek
                                       else "provider-default")

    async def check(name, action, timeout=90):
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(action(), timeout)
            value = {"status": "ready", **(result or {})}
        except Exception as exc:
            value = {"status": "failed", "code": exc.code if isinstance(exc, AppError) else type(exc).__name__,
                     "message": exc.message if isinstance(exc, AppError) else
                     "Check failed; inspect configuration and service availability."}
        value["seconds"] = round(time.perf_counter() - start, 3)
        report["checks"][name] = value
        print(json.dumps({name: value}, ensure_ascii=False), flush=True)

    async def configuration():
        services.models.configured()

    async def tcp(host, port):
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 10)
        writer.close()
        await writer.wait_closed()
        return {"port": port}

    async def postgres():
        async with database.sessions() as db:
            await db.execute(text("SELECT 1"))
            migrated = await db.scalar(text("SELECT to_regclass('alembic_version') IS NOT NULL"))
            return {"migration_table_present": migrated}

    async def storage():
        await services.storage.check()
        key = f"rag-live-probes/{uuid4()}.txt"
        payload = "真实 RAG 对象存储联调：写入、读取、删除。".encode()
        try:
            await services.storage.put(key, payload, "text/plain; charset=utf-8")
            if await services.storage.get(key) != payload:
                raise AppError("storage_roundtrip_mismatch", "对象存储读取内容与写入不一致", 502)
        finally:
            # Only remove this probe's unique object; never touch application objects.
            await services.storage.delete([key])
        return {"operations": ["head_bucket", "put", "get", "delete"], "bytes": len(payload)}

    async def embedding():
        batch = await services.models.embed_vectors(["临时 API Key 默认有效期为 60 秒。"],
                                                    include_sparse=settings.embedding_store_sparse)
        return {"dimension": len(batch.dense[0]),
                "sparse_entries": len(batch.sparse[0]) if batch.sparse else 0}

    async def rerank():
        result = await services.models.rerank("临时 API Key 默认有效期", [
            "临时 API Key 默认有效期为 60 秒。", "向量检索支持稠密和稀疏模式。"])
        if not result or result[0][0] != 0:
            raise AppError("probe_rerank_order", "相关片段未排在首位", 502)
        return {"results": len(result), "relevant_document_first": True}

    async def metadata():
        # Verbatim evidence from the downloaded official documentation, no retrieval is claimed here.
        body = "临时API Key 默认有效期为 60 秒，支持通过 `expire_in_seconds` 参数设置有效期（TTL），范围为 [1, 1800] 秒。"
        source = ROOT / "test-documents/aliyun/documents/generate-temporary-api-key.md"
        if body not in source.read_text(encoding="utf-8"):
            raise ValueError("Evidence snapshot changed")
        card = await summarize(Piece(body, ["生成临时API Key", "请求示例"], [], []), services.models)
        return {"metadata": card.model_dump()}

    async def rewrite():
        output = await services.models.structured("rewrite", "明确当前追问，保持原意，不需要澄清。",
            {"history": [{"question": "临时 API Key 默认多久过期？", "answer": "默认 60 秒。"}],
             "question": "它最长能设置为多少秒？"}, RewriteOutput)
        return {"output": output.model_dump()}

    async def answer():
        tokens = [token async for token in services.models.stream_answer([
            {"role": "system", "content": "仅依据给定资料简洁回答，附上引用编号 [1]。"},
            {"role": "user", "content": "资料[1]：临时API Key默认有效期为60秒，可设置为1至1800秒。问题：默认多久过期？"},
        ])]
        content = "".join(tokens)
        if "60" not in content or "[1]" not in content:
            raise AppError("probe_answer_evidence", "探针回答未包含预期事实和引用编号", 502)
        return {"stream_deltas": len(tokens), "answer": content,
                "scope": "provider_stream_only_not_rag"}

    async def coreference():
        services.models.chat_config("coreference")
        cases = [
            ("cross_paragraph", "可乐是一种饮料。\n\n它含有二氧化碳。", "可乐含有二氧化碳。"),
            ("same_paragraph", "可乐是一种饮料。它含有二氧化碳。", "可乐是一种饮料。可乐含有二氧化碳。"),
            ("ambiguous", "桌上有可乐和雪碧。\n\n它是冰的。", "它是冰的。"),
            ("negation_numbers_units", "可乐在本次试验中被编号为A100。\n\n它不适用于儿童，单价为5元，容量为330 ml。",
             "可乐不适用于儿童，单价为5元，容量为330 ml。"),
        ]
        results = []
        for name, body, expected in cases:
            parsed = await resolve_document(parse_document("probe.txt", body.encode()), services.models)
            actual = parsed.blocks[-1].text
            results.append({"case": name, "passed": actual == expected, "resolved": actual,
                            "edits": parsed.blocks[-1].edits})
        # Preserve failing case evidence without labelling a failed probe ready.
        report["coreference_cases"] = results
        if not all(case["passed"] for case in results):
            raise AppError("probe_coreference", "部分真实指代消解用例未通过，见 coreference_cases", 502)
        return {"cases_passed": len(results)}

    try:
        pg_url = make_url(settings.database_url)
        milvus_url = urlsplit(settings.milvus_uri)
        await asyncio.gather(
            check("model_configuration", configuration), check("postgresql", postgres, 20),
            check("postgresql_tcp", lambda: tcp(pg_url.host, pg_url.port or 5432), 12),
            check("milvus_tcp", lambda: tcp(milvus_url.hostname, milvus_url.port or
                  (443 if milvus_url.scheme == "https" else 19530)), 12),
            check("s3", storage), check("milvus", services.index.check, 30),
            check("embedding", embedding), check("rerank", rerank),
            check("deepseek_metadata", metadata, 140), check("deepseek_rewrite", rewrite, 140),
            check("deepseek_stream", answer, 140), check("coreference", coreference, 140),
        )
        report["ready_for_rag"] = all(v["status"] == "ready" for v in report["checks"].values())
        report["finished_at"] = datetime.now(UTC).isoformat()
        output = ROOT / "output/rag-live-preflight.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if report["ready_for_rag"] else 1
    finally:
        await services.close()
        await database.close()


if __name__ == "__main__":
    # pymilvus diagnostic tracebacks can contain connection details. Report sanitized errors above.
    logging.disable(logging.CRITICAL)
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        print(json.dumps({"status": "failed", "code": type(exc).__name__}))
        raise SystemExit(1) from None
