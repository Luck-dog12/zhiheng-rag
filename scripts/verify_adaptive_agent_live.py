"""Real thinking/tool-loop checks; optionally inject an unrelated first retrieval as a controlled fault."""
import argparse
import asyncio
import json
import time
from pathlib import Path

from aliyun_e2e_live import Acceptance
from rag.chat import ChatEngine
from rag.config import Settings
from rag.db import Database
from rag.models import Turn
from rag.services import Services

OUT = Path(__file__).resolve().parents[1] / "output/adaptive-agent-live"
QUESTIONS = {
    "concise": "百炼临时 API Key 默认有效期是多少，最短和最长是多少？",
    "guide": "我应该怎么通过百炼 OpenAI 兼容接口调用模型？请简要指导我。",
    "multi": "比较临时 API Key 的默认及最长有效期，并说明 Rerank 的 index 表示什么、relevance_score 能否跨请求比较。",
    "missing": "公司食堂下周三的午餐菜单是什么？",
    "mismatch": "百炼临时 API Key 默认有效期是多少，最短和最长是多少？",
}


async def main(case):
    OUT.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    database, services, runner = Database(settings), Services(settings), Acceptance()
    report = {"case": case, "thinking": {"agent": settings.agent_deepseek_thinking,
              "answer": settings.answer_deepseek_thinking}, "requests": [], "events": []}
    async def record_request(request):
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            report["requests"].append({"model": body["model"], "thinking": body.get("thinking"),
                "reasoning_effort": body.get("reasoning_effort"), "stream": body.get("stream", False),
                "tool_choice": body.get("tool_choice"), "max_tokens": body.get("max_tokens"),
                "reasoning_messages_returned": sum("reasoning_content" in m for m in body["messages"])})
    services.models.client.event_hooks["request"].append(record_request)
    if case == "mismatch":
        original, first = services.retriever.search, True
        async def faulty_search(db, user, query, kb_ids, **kwargs):
            nonlocal first
            if first:
                first = False
                report["injected_first_query"] = "qwen3.7-text-embedding 向量维度 dense sparse 输出格式"
                return await original(db, user, report["injected_first_query"], kb_ids, **kwargs)
            return await original(db, user, query, kb_ids, **kwargs)
        services.retriever.search = faulty_search
    started = time.monotonic()
    try:
        identity = await runner.login()
        cid = (await runner.call("POST", "/conversations"))["id"]
        report["conversation_id"] = cid
        engine = ChatEngine(database, services, settings)
        async for frame in engine.events(identity["user"]["id"], cid, QUESTIONS[case], [runner.state["kb_id"]], "adaptive-live"):
            event, data = frame.strip().split("\n", 1)
            event, data = event[7:], json.loads(data[6:])
            if event != "delta":
                report["events"].append({"event": event, "data": data})
                if event in {"status", "error"}:
                    print(json.dumps({"case": case, "event": event, **data}, ensure_ascii=False), flush=True)
        async with database.sessions() as db:
            from sqlalchemy import select
            turn = await db.scalar(select(Turn).where(Turn.conversation_id == cid))
            report.update(status=turn.status, answer=turn.answer, citations=turn.citations,
                          agent_state=turn.agent_state, error_code=turn.error_code)
        assert report["status"] == "completed", report["error_code"]
        assert all(r["thinking"]["type"] == "enabled" for r in report["requests"])
        assert any(r["reasoning_messages_returned"] > 0 for r in report["requests"])
        if case == "mismatch":
            traces = report["agent_state"]["trace"]
            assert len([t for t in traces if t["tool"] == "search_knowledge_base"]) >= 2
            assert any(t["tool"] == "assess_evidence" and t["verdict"] != "sufficient" for t in traces)
            assert report["citations"]
        report["passed"] = True
    finally:
        report["seconds"] = round(time.monotonic() - started, 2)
        (OUT / f"{case}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"case": case, "passed": report.get("passed", False), "seconds": report["seconds"],
            "characters": len(report.get("answer", "")), "citations": len(report.get("citations", []))}), flush=True)
        await services.close()
        await database.close()
        await runner.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=list(QUESTIONS))
    asyncio.run(main(parser.parse_args().case))
