"""Opt-in, real-service evaluation. Secrets are read only from environment variables."""
import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


async def evaluate(args):
    corpus = json.loads((ROOT / "evals/corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "evals/questions.json").read_text(encoding="utf-8"))
    negatives = json.loads((ROOT / "evals/unanswerable.json").read_text(encoding="utf-8"))
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    mapping_path = output / "live-eval-mapping.json"
    async with httpx.AsyncClient(base_url=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000"), timeout=300) as client:
        async def call(method, path, **kwargs):
            response = await client.request(method, "/api/v1" + path, **kwargs)
            response.raise_for_status()
            return response.json()

        login = await call("POST", "/auth/login", json={"username": os.environ["EVAL_USERNAME"], "password": os.environ["EVAL_PASSWORD"]})
        client.headers["X-CSRF-Token"] = login["csrf_token"]
        if args.ingest:
            kbs = await call("GET", "/knowledge-bases")
            kb = next((k for k in kbs if k["name"] == "RAG 合成评测"), None)
            if kb is None:
                kb = await call("POST", "/knowledge-bases", json={"name": "RAG 合成评测", "description": "合成测试数据，不作为业务知识"})
            mapping = {"kb_id": kb["id"], "documents": {}}
            for doc in corpus:
                result = await call("POST", "/documents", data={"kb_id": kb["id"]},
                                    files={"file": (doc["id"] + ".md", doc["text"].encode(), "text/markdown")})
                mapping["documents"][doc["id"]] = result["document_id"]
                deadline = time.monotonic() + 1800
                while True:
                    job = await call("GET", "/jobs/" + result["job_id"])
                    if job["status"] == "done":
                        break
                    if job["status"] in ("failed", "superseded") or time.monotonic() > deadline:
                        raise RuntimeError(f"Ingestion did not complete: {job['status']} {job.get('error_code')}")
                    await asyncio.sleep(3)
            mapping_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        else:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        results = []
        for question in questions:
            start = time.perf_counter()
            response = await call("POST", "/retrieval/search", json={"question": question["question"], "kb_ids": [mapping["kb_id"]]})
            wanted = mapping["documents"][question["document_id"]]
            hits = [r["document_id"] for r in response["hits"]]
            rank = hits.index(wanted) + 1 if wanted in hits else None
            results.append({"id": question["id"], "question": question["question"], "expected_document": wanted,
                "recall_candidates": wanted in [r["document_id"] for r in response["candidates"]],
                "recall_at_5": wanted in hits, "ndcg_at_5": 1 / math.log2(rank + 1) if rank else 0,
                "expected_evidence": question["evidence"], "expected_answer": question["answer"],
                "evidence_in_top5": any(question["evidence"] in r["body"] for r in response["hits"]),
                "retrieval_seconds": time.perf_counter() - start, "hits": response["hits"]})
        if args.answers:
            # 20 conversations started concurrently, staying inside the default per-user rate limit.
            semaphore = asyncio.Semaphore(args.concurrency)
            async def answer(result):
                async with semaphore:
                    start = time.perf_counter()
                    try:
                        conversation = await call("POST", "/conversations")
                        response = await client.post("/api/v1/chat", json={"question": result["question"],
                            "conversation_id": conversation["id"], "kb_ids": [mapping["kb_id"]]})
                        events = response.text.replace("\r\n", "\n").split("\n\n")
                        result["chat_success"] = response.status_code == 200 and any(e.startswith("event: done\n") for e in events)
                        if result["chat_success"]:
                            turns = (await call("GET", "/conversations/" + conversation["id"]))["turns"]
                            result["answer"] = turns[-1]["answer"]
                            citations = turns[-1]["citations"]
                            result["citations"] = citations
                            if result.get("expected_document"):
                                correct = sum(c["document_id"] == result["expected_document"] for c in citations)
                                result["citation_document_hit"] = correct > 0
                                result["citation_document_precision"] = correct / len(citations) if citations else 0
                            else:
                                # Lexical signal for triage; humans must verify whether the answer still invents facts.
                                result["abstention_phrase_present"] = any(p in result["answer"] for p in
                                    ("资料不足", "无法确定", "无法回答", "没有足够", "没有找到足够", "未提供", "未提及"))
                    except httpx.HTTPError as exc:
                        result["chat_success"], result["error_type"] = False, type(exc).__name__
                    finally:
                        result["chat_seconds"] = time.perf_counter() - start
            answer_cases = [*results, *negatives]
            for offset in range(0, len(answer_cases), 20):
                if offset:
                    await asyncio.sleep(61)
                await asyncio.gather(*(answer(result) for result in answer_cases[offset:offset + 20]))
        times = sorted(r.get("chat_seconds", r["retrieval_seconds"]) for r in results)
        report = {"mode": "live_external_services", "question_count": len(results), "concurrency": args.concurrency if args.answers else 1,
            "recall_candidates": sum(r["recall_candidates"] for r in results) / len(results),
            "recall_at_5": sum(r["recall_at_5"] for r in results) / len(results),
            "ndcg_at_5": sum(r["ndcg_at_5"] for r in results) / len(results),
            "p50_seconds": times[len(times) // 2], "p95_seconds": times[math.ceil(len(times) * .95) - 1],
            "questions": results}
        if args.answers:
            report.update({"chat_requests": len(answer_cases),
                "chat_success_rate": sum(r["chat_success"] for r in answer_cases) / len(answer_cases),
                "citation_document_hit_rate": sum(r.get("citation_document_hit", False) for r in results) / len(results),
                "citation_document_precision": sum(r.get("citation_document_precision", 0) for r in results) / len(results),
                "unanswerable_phrase_rate": sum(r.get("abstention_phrase_present", False) for r in negatives) / len(negatives),
                "unanswerable_questions": negatives,
                "manual_review_required": "Document IDs and abstention phrases are proxies; review answer entailment and invented facts against expected evidence."})
        (output / "evaluation-live.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "questions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=range(1, 21), default=20)
    asyncio.run(evaluate(parser.parse_args()))
