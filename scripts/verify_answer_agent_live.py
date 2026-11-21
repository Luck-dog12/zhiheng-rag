"""Real DeepSeek answer-agent checks; reads ignored local test credentials, never prints them."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from aliyun_e2e_live import Acceptance
from rag.config import Settings
from rag.db import Database
from rag.models import Turn

OUT = Path(__file__).resolve().parents[1] / "output/answer-agent-live"


async def chat(runner, client, database, question, conversation_id=None):
    if conversation_id is None:
        conversation_id = (await runner.call("POST", "/conversations", client=client))["id"]
    events, event, displayed = [], None, ""
    started = time.perf_counter()
    async with client.stream("POST", "/api/v1/chat", json={"question": question,
            "conversation_id": conversation_id, "kb_ids": [runner.state["kb_id"]]}) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Chat HTTP {response.status_code}")
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "delta":
                    displayed += data["text"]
                elif event == "reset":
                    displayed = data["text"]
                if event != "delta":
                    events.append({"event": event, "data": data})
    detail = await runner.call("GET", "/conversations/" + conversation_id, client=client)
    turn = detail["turns"][-1]
    async with database.sessions() as db:
        saved = await db.get(Turn, turn["id"])
        state, resolved = saved.agent_state, saved.resolved_question
    if turn["status"] == "completed":
        assert displayed == turn["answer"], "SSE display must match saved final answer"
        for citation in turn["citations"]:
            chunk = await runner.call("GET", "/chunks/" + citation["chunk_id"], client=client)
            assert chunk["version_id"] == citation["version_id"] and citation["sources"]
    result = {"conversation_id": conversation_id, "turn": turn, "resolved_question": resolved,
              "state": state, "events": events, "seconds": round(time.perf_counter() - started, 2)}
    print(json.dumps({"question": question, "conversation_id": conversation_id, "status": turn["status"],
        "mode": state.get("mode"), "characters": len(turn["answer"] or turn["draft_answer"]),
        "citations": len(turn["citations"]), "tools": [x["tool"] for x in state["trace"]],
        "budgets": [x["max_tokens"] for x in state["trace"] if x["tool"] == "generate_answer"],
        "error": turn["error_code"], "seconds": result["seconds"]}, ensure_ascii=False), flush=True)
    return result


async def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    runner, database = Acceptance(), Database(Settings())
    report = {"mode": mode, "retrieval": Settings().retrieval_sparse_mode, "model": Settings().deepseek_model,
              "turns": []}
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8004", timeout=550) as normal:
            await runner.login(normal)
            if mode == "context":
                cid = None
                for question in ["opemai兼容接口是那个？", "我应该怎么调用模型？指导我", "继续"]:
                    result = await chat(runner, normal, database, question, cid)
                    report["turns"].append(result)
                    (OUT / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    assert result["turn"]["status"] == "completed", result["turn"]["error_code"]
                    cid = result["conversation_id"]
                assert report["turns"][-1]["state"]["mode"] == "continue"
                assert report["turns"][-1]["turn"]["citations"]
            else:
                port = 8002 if mode == "exhausted" else 8003
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=550) as limited:
                    await runner.login(limited)
                    first = await chat(runner, limited, database,
                        "请指导我通过阿里云百炼 OpenAI 兼容接口调用模型，说明配置、鉴权、Python 示例和常见错误。")
                    report["turns"].append(first)
                (OUT / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                resets = [e for e in first["events"] if e["event"] == "reset"]
                assert resets, "Test did not trigger real provider truncation"
                if mode == "exhausted":
                    assert first["turn"]["status"] == "failed" and first["turn"]["can_resume"]
                    assert first["turn"]["draft_answer"] and not first["turn"]["answer"]
                    assert [x["max_tokens"] for x in first["state"]["trace"] if x["tool"] == "generate_answer"] == [64, 128, 256]
                    resumed = await chat(runner, normal, database, "继续", first["conversation_id"])
                    report["turns"].append(resumed)
                    assert resumed["turn"]["status"] == "completed" and resumed["state"]["mode"] == "restart"
                else:
                    assert first["turn"]["status"] == "completed"
            report["passed"] = True
    finally:
        (OUT / f"{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await runner.client.aclose()
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["context", "exhausted", "automatic"])
    args = parser.parse_args()
    asyncio.run(main(args.mode))
