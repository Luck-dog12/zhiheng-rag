import asyncio
import json
import math
import time
from pathlib import Path

from rag.models import User

ROOT = Path(__file__).resolve().parents[1]


async def test_fifty_question_evaluation_and_twenty_concurrent_chats(harness):
    h = harness
    corpus = json.loads((ROOT / "evals/corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "evals/questions.json").read_text(encoding="utf-8"))
    assert len(questions) >= 50
    documents = {d["id"]: d for d in corpus}
    mapping = {}
    for doc in corpus:
        result = await h.upload(doc["text"], filename=doc["id"] + ".md")
        mapping[doc["id"]] = result["document_id"]
    await h.drain(max_jobs=40)
    scores = []
    for item in questions:
        assert item["evidence"] in documents[item["document_id"]]["text"]
        async with h.database.sessions() as db:
            user = await db.get(User, h.admin_id)
            result = await h.services.retriever.search(db, user, item["question"], [h.kb])
        relevant = mapping[item["document_id"]]
        candidates = [v["document_id"] for v in result["candidates"]]
        hits = [v["document_id"] for v in result["hits"]]
        rank = hits.index(relevant) + 1 if relevant in hits else None
        scores.append({"id": item["id"], "recall_candidates": int(relevant in candidates),
                       "recall_at_5": int(relevant in hits), "ndcg_at_5": 1 / math.log2(rank + 1) if rank else 0,
                       "evidence_in_top5": any(item["evidence"] in v["body"] for v in result["hits"])})

    # One conversation per independent request; test transport, not a network SLA benchmark.
    ids = [(await h.client.post("/api/v1/conversations")).json()["id"] for _ in range(20)]
    async def ask(index):
        start = time.perf_counter()
        response = await h.client.post("/api/v1/chat", json={"question": questions[index]["question"],
            "kb_ids": [h.kb], "conversation_id": ids[index]})
        return {"seconds": time.perf_counter() - start,
                "success": response.status_code == 200 and "event: done" in response.text}
    h.services.models.delay = 0.001
    results = await asyncio.gather(*(ask(i) for i in range(20)))
    times = sorted(r["seconds"] for r in results)
    report = {"mode": "simulated_external_services_sqlite_test_only", "question_count": len(scores),
              "recall_candidates": sum(s["recall_candidates"] for s in scores) / len(scores),
              "recall_at_5": sum(s["recall_at_5"] for s in scores) / len(scores),
              "ndcg_at_5": sum(s["ndcg_at_5"] for s in scores) / len(scores),
              "concurrent_chats": 20, "successful_chats": sum(r["success"] for r in results),
              "p50_seconds": times[9], "p95_seconds": times[18], "questions": scores,
              "limitation": "Test doubles do not measure real embedding quality, PostgreSQL locking or provider latency."}
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "evaluation-simulated.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(r["success"] for r in results)
    assert report["recall_candidates"] == 1
