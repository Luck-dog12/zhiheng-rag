"""Compare Qwen dense + Milvus BM25 against Qwen dense + Qwen sparse using remote services."""
import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from rag.config import Settings
from rag.errors import AppError
from rag.processing import clean_block, make_chunks, parse_document
from rag.providers import RemoteModels
from rag.retrieval import reciprocal_rank_fusion
from rag.vector import MilvusIndex

ROOT = Path(__file__).resolve().parents[1]
MODES = ("bm25", "model")


def prepare_corpus(corpus, questions, settings):
    """Use identical deterministic chunks. No metadata/coreference/answer generation in this retrieval experiment."""
    docs = {d["id"]: d for d in corpus}
    if len(docs) != len(corpus) or not corpus or not questions:
        raise ValueError("Corpus and questions must be non-empty, with unique document IDs")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    for q in questions:
        if not q["evidence"] or q["evidence"] not in docs[q["document_id"]]["text"]:
            raise ValueError("Question evidence must be an exact excerpt from its document")
    digest = hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    kb_id = str(uuid5(NAMESPACE_URL, "rag-retrieval-comparison:" + digest))
    rows, lookup = [], {}
    for document in corpus:
        doc_id = str(uuid5(NAMESPACE_URL, kb_id + document["id"]))
        version_id = str(uuid5(NAMESPACE_URL, doc_id + settings.processing_fingerprint))
        parsed = parse_document(document["id"] + ".md", document["text"].encode())
        for block in parsed.blocks:
            clean_block(block, normalize_whitespace=settings.clean_normalize_whitespace,
                        remove_fillers=settings.clean_remove_fillers)
        pieces = make_chunks(parsed.blocks, settings.chunk_target, settings.chunk_max, settings.chunk_overlap)
        if not pieces:
            raise ValueError("Empty document after parsing")
        for ordinal, piece in enumerate(pieces):
            chunk_id = str(uuid5(NAMESPACE_URL, version_id + ":" + str(ordinal)))
            rows.append({"chunk_id": chunk_id, "kb_id": kb_id, "document_id": doc_id,
                         "version_id": version_id, "body": piece.body})
            lookup[chunk_id] = {"document_id": document["id"], "body": piece.body}
    for q in questions:
        if not any(r["document_id"] == q["document_id"] and q["evidence"] in r["body"] for r in lookup.values()):
            raise ValueError("Evidence must remain in at least one chunk; revise the evaluation annotation")
    return rows, lookup, digest


def measure_ranking(ids, lookup, question, k=5):
    # Evidence-level binary relevance; duplicate chunks from one document earn no extra gain.
    top = ids[:k]
    relevant = [i + 1 for i, cid in enumerate(top) if lookup[cid]["document_id"] == question["document_id"]
                and question["evidence"] in lookup[cid]["body"]]
    rank = relevant[0] if relevant else None
    return {"recall_at_5": float(rank is not None), "ndcg_at_5": 1 / math.log2(rank + 1) if rank else 0.0,
            "mrr_at_5": 1 / rank if rank else 0.0}


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


async def compare_questions(settings, models, index, rows, lookup, questions, progress=None):
    results = []
    kb_id = rows[0]["kb_id"]
    for number, question in enumerate(questions):
        started = time.perf_counter()
        query = await models.embed_vectors([question["question"]], query=True, include_sparse=True)
        embedding_seconds = time.perf_counter() - started
        case = {"id": question["id"], "question": question["question"],
                "expected_document": question["document_id"], "expected_evidence": question["evidence"],
                "shared_query_embedding_seconds": embedding_seconds, "variants": {}}
        # Alternate execution order to reduce systematic latency bias.
        for mode in MODES if number % 2 == 0 else reversed(MODES):
            started = time.perf_counter()
            paths = await index.search(question["question"], query.dense[0], [kb_id],
                                       settings.retrieval_per_path, model_sparse=query.sparse[0], sparse_mode=mode)
            fused = reciprocal_rank_fusion(paths, settings.rrf_k)[:settings.retrieval_candidates]
            ids = [h["chunk_id"] for h in fused]
            if any(cid not in lookup for cid in ids):
                raise ValueError("Index returned a chunk outside the isolated corpus")
            retrieval_seconds = time.perf_counter() - started
            started = time.perf_counter()
            ranking = await models.rerank(question["question"], [lookup[cid]["body"] for cid in ids])
            rerank_seconds = time.perf_counter() - started
            ordered = [ids[i] for i, _ in ranking]
            candidate_hit = any(lookup[cid]["document_id"] == question["document_id"]
                                and question["evidence"] in lookup[cid]["body"] for cid in ids)
            case["variants"][mode] = {
                "candidate_count": len(ids), "recall_candidates": float(candidate_hit),
                "fusion": measure_ranking(ids, lookup, question), "reranked": measure_ranking(ordered, lookup, question),
                "retrieval_seconds": retrieval_seconds, "rerank_seconds": rerank_seconds,
                "total_seconds_with_shared_embedding": embedding_seconds + retrieval_seconds + rerank_seconds,
                "paths": paths, "fusion_order": ids,
                "hits": [{"chunk_id": ids[i], "document_id": lookup[ids[i]]["document_id"], "score": score}
                         for i, score in ranking[:settings.answer_top_k]],
            }
        results.append(case)
        if progress:
            progress(results)
    return results


def summarize(results):
    summary = {}
    for mode in MODES:
        variants = [r["variants"][mode] for r in results]
        summary[mode] = {
            "recall_candidates": sum(v["recall_candidates"] for v in variants) / len(variants),
            **{f"{stage}_{metric}": sum(v[stage][metric] for v in variants) / len(variants)
               for stage in ("fusion", "reranked") for metric in ("recall_at_5", "ndcg_at_5", "mrr_at_5")},
            **{f"{stage}_p{p}_seconds": percentile([v[stage] for v in variants], p / 100)
               for stage in ("retrieval_seconds", "rerank_seconds", "total_seconds_with_shared_embedding") for p in (50, 95)},
        }
    summary["model_minus_bm25"] = {key: summary["model"][key] - summary["bm25"][key] for key in summary["bm25"]}
    for stage in ("fusion", "reranked"):
        differences = [r["variants"]["model"][stage]["ndcg_at_5"] - r["variants"]["bm25"][stage]["ndcg_at_5"] for r in results]
        summary[f"{stage}_paired_questions"] = {"model_wins": sum(v > 1e-9 for v in differences),
            "bm25_wins": sum(v < -1e-9 for v in differences), "ties": sum(abs(v) <= 1e-9 for v in differences)}
    return summary


async def run(args):
    settings = Settings()
    if settings.embedding_protocol != "dashscope":
        raise ValueError("This comparison requires EMBEDDING_PROTOCOL=dashscope and EMBEDDING_ENDPOINT")
    # Never write to a production collection or change its schema.
    collection = "rag_ab_" + uuid4().hex
    settings = Settings(**{**settings.model_dump(), "milvus_collection": collection,
                           "embedding_store_sparse": True, "retrieval_sparse_mode": "bm25"})
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    rows, lookup, digest = prepare_corpus(corpus, questions, settings)
    models, index = RemoteModels(settings), MilvusIndex(settings)
    report = {"status": "running", "mode": "live_remote_retrieval_comparison", "created_at": datetime.now(timezone.utc).isoformat(),
              "embedding_model": settings.embedding_model, "rerank_model": settings.rerank_model,
              "dimension": settings.embedding_dimension, "index_fingerprint": settings.index_fingerprint,
              "collection": collection, "corpus_sha256": digest,
              "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
              "documents": len(corpus), "chunks": len(rows), "question_count": len(questions),
              "per_path": settings.retrieval_per_path, "rrf_k": settings.rrf_k,
              "candidate_limit": settings.retrieval_candidates, "top_k": settings.answer_top_k,
              "limitations": ["Retrieval experiment only: no metadata, coreference, database ACL, or answer generation.",
                  "Both variants share a single dense&sparse query call; timing is not independent dense-only latency.",
                  "Controlled evaluation corpus; if chunks <= per_path, candidate recall is saturated and cannot discriminate methods.",
                  "Relevance requires the annotated document and exact evidence; equivalent passages elsewhere are not counted.",
                  "One pass of sequential questions with alternating A/B order; not a concurrent capacity or business-quality baseline."],
              "questions": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def progress(results):
        report["questions"] = results
        save()
        print(f"Compared {len(results)}/{len(questions)} questions", flush=True)

    try:
        report["stage"] = "milvus_initialize"
        save()
        await index.initialize()
        report["stage"] = "embed_and_index"
        save()
        started = time.perf_counter()
        for start in range(0, len(rows), settings.embedding_batch_size):
            batch = rows[start:start + settings.embedding_batch_size]
            vectors = await models.embed_vectors([row["body"] for row in batch], include_sparse=True)
            await index.upsert([{**row, "dense": vectors.dense[i], "model_sparse": vectors.sparse[i]} for i, row in enumerate(batch)])
            report["indexed_chunks"] = min(start + len(batch), len(rows))
            save()
            print(f"Indexed {report['indexed_chunks']}/{len(rows)} chunks", flush=True)
        await index.verify([row["chunk_id"] for row in rows])
        report["indexing_seconds"] = time.perf_counter() - started
        report["stage"] = "compare_queries"
        results = await compare_questions(settings, models, index, rows, lookup, questions, progress)
        report.update(status="completed", summary=summarize(results))
        save()
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report.update(status="failed", error={"code": getattr(exc, "code", type(exc).__name__),
                                              "message": getattr(exc, "message", "Comparison failed; inspect configuration and service availability.")})
        save()
        print(json.dumps({"status": report["status"], "stage": report["stage"], "error": report["error"]}, ensure_ascii=False))
        return 1
    finally:
        await models.close()
        await index.close()
        print(f"Report: {args.output}", flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "evals/corpus.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "evals/questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/retrieval-comparison.json")
    return parser.parse_args()


if __name__ == "__main__":
    # SDK exception logging can include endpoint details; the report only exposes sanitized application errors.
    logging.getLogger("pymilvus").setLevel(logging.CRITICAL)
    try:
        raise SystemExit(asyncio.run(run(arguments())))
    except (ValueError, KeyError, OSError, AppError) as exc:
        print(f"Comparison could not start: {type(exc).__name__}. Check configuration and evaluation inputs.")
        raise SystemExit(1) from None
