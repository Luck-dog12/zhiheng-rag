import time

from sqlalchemy import select

from rag.errors import AppError
from rag.models import Chunk, ChunkMetadata, Document, DocumentVersion
from rag.security import require_kbs
from rag.telemetry import STAGES


def reciprocal_rank_fusion(paths: dict, k: int = 60) -> list[dict]:
    combined: dict[str, dict] = {}
    for name, results in paths.items():
        seen = set()
        for rank, row in enumerate(results, 1):
            chunk_id = row["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            item = combined.setdefault(chunk_id, {"chunk_id": chunk_id, "rrf_score": 0.0, "paths": {}})
            item["rrf_score"] += 1 / (k + rank)
            item["paths"][name] = {"rank": rank, "score": row["score"]}
    return sorted(combined.values(), key=lambda x: (-x["rrf_score"], x["chunk_id"]))


async def load_current_chunks(db, ids: list[str], kb_ids: list[str], embedding_fingerprint: str | None = None):
    if not ids:
        return {}
    stmt = (select(Chunk, ChunkMetadata, Document, DocumentVersion)
            .join(ChunkMetadata, ChunkMetadata.chunk_id == Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
            .where(Chunk.id.in_(ids), Chunk.kb_id.in_(kb_ids), Document.deleted_at.is_(None),
                   Document.current_version_id == Chunk.version_id, DocumentVersion.status == "ready"))
    if embedding_fingerprint:
        stmt = stmt.where(DocumentVersion.embedding_fingerprint == embedding_fingerprint)
    rows = (await db.execute(stmt)).all()
    return {chunk.id: (chunk, metadata, doc, version) for chunk, metadata, doc, version in rows}


class Retriever:
    def __init__(self, settings, models, index):
        self.settings, self.models, self.index = settings, models, index

    async def search(self, db, user, question: str, kb_ids: list[str], *, strategy: str = "hybrid") -> dict:
        started = time.monotonic()
        if strategy not in ("hybrid", "dense", "sparse"):
            raise AppError("invalid_strategy", "检索策略无效")
        allowed = await require_kbs(db, user, kb_ids)
        sparse_vector = None
        if self.settings.retrieval_sparse_mode == "model" and strategy != "dense":
            embeddings = await self.models.embed_vectors([question], query=True, include_sparse=True)
            vectors, sparse_vector = embeddings.dense, embeddings.sparse[0]
        elif strategy != "sparse":
            vectors = await self.models.embed([question], query=True)
        else:
            # Native BM25 needs text only; do not pay for an unused dense embedding.
            vectors = [[]]
        size = self.settings.retrieval_per_path
        candidates, discarded = [], 0
        while True:
            paths = await self.index.search(question, vectors[0], allowed, size,
                                            strategy=strategy,
                                            **({"model_sparse": sparse_vector} if sparse_vector is not None else {}))
            fused = reciprocal_rank_fusion(paths, self.settings.rrf_k)
            rows = await load_current_chunks(db, [x["chunk_id"] for x in fused], allowed,
                                             self.settings.embedding_fingerprint)
            candidates = []
            for hit in fused:
                if hit["chunk_id"] not in rows:
                    continue
                chunk, metadata, doc, version = rows[hit["chunk_id"]]
                candidates.append({**hit, "body": chunk.body, "summary": metadata.summary,
                    "keywords": metadata.keywords, "time_mentions": metadata.time_mentions,
                    "locations": metadata.locations, "heading_path": chunk.heading_path,
                    "kb_id": chunk.kb_id, "document_id": doc.id, "version_id": version.id,
                    "title": doc.title, "sources": chunk.sources})
            discarded = len(fused) - len(candidates)
            if (len(candidates) >= self.settings.answer_top_k or size >= self.settings.retrieval_max_per_path
                    or all(len(v) < size for v in paths.values())):
                break
            size = min(size * 2, self.settings.retrieval_max_per_path)
        candidates = candidates[:self.settings.retrieval_candidates]
        STAGES.labels("retrieval").observe(time.monotonic() - started)
        if not candidates:
            return {"query": question, "strategy": strategy, "sparse_mode": self.settings.retrieval_sparse_mode,
                    "top_k": self.settings.answer_top_k,
                    "hits": [], "candidates": [], "discarded": discarded}
        rerank_started = time.monotonic()
        ranking = await self.models.rerank(question, [item["body"] for item in candidates])
        for index, score in ranking:
            candidates[index]["rerank_score"] = score
        ordered = [candidates[index] for index, _ in ranking]
        # Re-check after the potentially slow model call, before exposing any snippets.
        await db.commit()
        await require_kbs(db, user, allowed)
        still_current = await load_current_chunks(db, [x["chunk_id"] for x in ordered], allowed)
        ordered = [x for x in ordered if x["chunk_id"] in still_current]
        STAGES.labels("rerank").observe(time.monotonic() - rerank_started)
        return {"query": question, "strategy": strategy, "sparse_mode": self.settings.retrieval_sparse_mode,
                "top_k": self.settings.answer_top_k,
                "hits": ordered[:self.settings.answer_top_k],
                "candidates": ordered, "discarded": discarded}


def citations_from_hits(hits: list[dict]) -> list[dict]:
    return [{"number": i, "chunk_id": hit["chunk_id"], "document_id": hit["document_id"],
             "version_id": hit["version_id"], "kb_id": hit["kb_id"], "title": hit["title"],
             "heading_path": hit["heading_path"], "sources": [
                 {k: source[k] for k in ("block_id", "page", "paragraph", "original")} for source in hit["sources"]
             ], "score": hit["rerank_score"]} for i, hit in enumerate(hits, 1)]


async def assert_citations_current(db, user, citations: list[dict]):
    if not citations:
        return
    kb_ids = list({c["kb_id"] for c in citations})
    await require_kbs(db, user, kb_ids)
    rows = await load_current_chunks(db, [c["chunk_id"] for c in citations], kb_ids)
    if any(c["chunk_id"] not in rows for c in citations):
        raise AppError("source_changed", "引用资料已更新或删除，请重新提问", 409)
