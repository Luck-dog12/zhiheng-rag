import json

import pytest
from rag.chat import ChatEngine
from rag.errors import GenerationInterrupted
from rag.models import Document, Grant, Turn
from sqlalchemy import delete, select


def events(text):
    frames = []
    for frame in text.split("\n\n"):
        if frame.startswith("event:"):
            name, data = frame.split("\n", 1)
            frames.append((name[7:], json.loads(data[6:])))
    return frames


async def seed(h):
    await h.upload("# 可乐说明\n\n可乐的容量为330 ml，单价5元，不含酒精。")
    await h.drain()
    return (await h.client.post("/api/v1/conversations")).json()["id"]


async def ask(h, cid, question="可乐容量是多少？", kb=None):
    return await h.client.post("/api/v1/chat", json={"question": question, "kb_ids": [kb or h.kb], "conversation_id": cid})


async def test_continue_reuses_current_evidence_and_passes_actual_conversation_to_deepseek(harness):
    h = harness
    cid = await seed(h)
    assert "event: done" in (await ask(h, cid)).text
    searches = len(h.services.index.searches)
    assert "event: done" in (await ask(h, cid, "继续")).text
    answer_calls = [data for purpose, data in h.services.models.calls if purpose == "answer"]
    assert answer_calls[-1]["mode"] == "continue"
    assert "330" in answer_calls[-1]["previous_answer"]
    assert answer_calls[-1]["history"][0]["question"] == "可乐容量是多少？"
    assert len(h.services.index.searches) == searches
    async with h.database.sessions() as db:
        turns = list((await db.scalars(select(Turn).where(Turn.conversation_id == cid).order_by(Turn.created_at))).all())
        assert [a["tool"] for a in turns[-1].agent_state["trace"]] == ["reuse_evidence", "assess_evidence", "answer", "generate_answer"]


async def test_old_failed_question_survives_intervening_clarification(harness):
    h = harness
    cid = await seed(h)
    await ask(h, cid)
    async with h.database.sessions() as db, db.begin():
        db.add(Turn(conversation_id=cid, question="如何选购可乐？指导我", scope_kb_ids=[h.kb],
                    status="failed", error_code="generation_incomplete"))
        await db.flush()
        db.add(Turn(conversation_id=cid, question="继续", scope_kb_ids=[h.kb], status="completed",
                    answer="您想继续哪部分内容？"))
    response = await ask(h, cid, "继续")
    assert "event: done" in response.text
    data = [d for p, d in h.services.models.calls if p == "answer"][-1]
    assert data["mode"] == "restart" and "选购" in data["question"]
    assert data["previous_answer"] is None
    assert any(t["status"] == "failed" and t["question"] == "如何选购可乐？指导我" and not t["answer"]
               for t in data["history"])


@pytest.mark.parametrize("partial", ["未完成草稿", ""])
async def test_answer_truncation_grows_budget_and_resets_stream_before_complete_answer(harness, partial):
    h = harness
    cid = await seed(h)
    budgets = []

    async def stream(messages, *, max_tokens=None):
        budgets.append(max_tokens)
        if len(budgets) < 3:
            if partial:
                yield partial
            raise GenerationInterrupted("length")
        yield "可乐容量为330 ml。[1]"

    h.services.models.stream_answer = stream
    response = await ask(h, cid)
    frames = events(response.text)
    assert budgets == [8192, 16384, 32768]
    assert sum(e == "reset" for e, _ in frames) == 2
    displayed = ""
    for event, data in frames:
        if event == "delta":
            displayed += data["text"]
        if event == "reset":
            displayed = data["text"]
    assert displayed == "可乐容量为330 ml。[1]"
    turn = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"][-1]
    assert turn["status"] == "completed" and turn["answer"] == displayed and not turn["draft_answer"]


async def test_exhausted_draft_is_durable_but_not_factual_history_and_can_resume_after_engine_restart(harness):
    h = harness
    cid = await seed(h)
    original = h.services.models.stream_answer

    async def truncated(messages, *, max_tokens=None):
        yield "不可信未完成草稿"
        raise GenerationInterrupted("length")

    h.services.models.stream_answer = truncated
    failed = await ask(h, cid)
    assert "event: done" not in failed.text and '"can_resume": true' in failed.text
    turn = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"][-1]
    assert turn["answer"] == "" and turn["draft_answer"] == "不可信未完成草稿" and turn["can_resume"]
    h.services.models.stream_answer = original
    engine = ChatEngine(h.database, h.services, h.settings)
    frames = [part async for part in engine.events(h.admin_id, cid, "继续", [h.kb], "resumed-trace")]
    assert "event: done" in "".join(frames)
    data = [d for p, d in h.services.models.calls if p == "answer"][-1]
    assert data["mode"] == "restart" and data["question"] != "继续"
    assert "不可信未完成草稿" not in json.dumps(data, ensure_ascii=False)


async def test_recovery_honors_permission_revocation_and_hides_drafts(harness):
    h = harness
    cid = await seed(h)
    await h.login("member")
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]

    async def revoked(messages, *, max_tokens=None):
        yield "可乐秘密草稿"
        async with h.database.sessions() as db, db.begin():
            await db.execute(delete(Grant).where(Grant.user_id == h.member_id))
        raise GenerationInterrupted("length")

    h.services.models.stream_answer = revoked
    response = await ask(h, cid)
    assert "event: done" not in response.text
    turn = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"][-1]
    assert turn["status"] == "redacted" and not turn["draft_answer"] and not turn["can_resume"]
    assert (await ask(h, cid, "继续")).status_code == 403


async def test_selected_scope_change_does_not_pass_other_scope_history_to_agent(harness):
    h = harness
    cid = await seed(h)
    await ask(h, cid)
    assert "event: done" in (await ask(h, cid, "继续", h.private_kb)).text
    data = [d for p, d in h.services.models.calls if p == "agent"][-1]
    assert data["history"] == [] and data["continuation_target_id"] is None


async def test_document_version_change_hides_unverified_draft_and_excludes_it_from_context(harness):
    h = harness
    cid = await seed(h)
    original = h.services.models.stream_answer
    async def truncated(messages, *, max_tokens=None):
        yield "旧版本的未完成内容"
        raise GenerationInterrupted("length")
    h.services.models.stream_answer = truncated
    await ask(h, cid)
    async with h.database.sessions() as db, db.begin():
        doc = await db.scalar(select(Document).where(Document.kb_id == h.kb))
        doc.current_version_id = None
    turn = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"][-1]
    assert turn["status"] == "redacted" and not turn["draft_answer"] and not turn["can_resume"]
    h.services.models.stream_answer = original
    await ask(h, cid, "继续")
    assert [d for p, d in h.services.models.calls if p == "agent"][-1]["history"] == []


async def test_agent_rejects_injected_acl_arguments_and_stops_at_step_budget(harness):
    h = harness
    cid = await seed(h)

    async def bad_agent(messages, tools):
        return {"role": "assistant", "content": None, "tool_calls": [{"id": "bad", "type": "function",
            "function": {"name": "search_knowledge_base", "arguments": json.dumps({"query": "机密", "kb_ids": [h.private_kb]})}}]}

    h.services.models.agent_step = bad_agent
    response = await ask(h, cid)
    assert "agent_limit" in response.text and "event: done" not in response.text
    assert not h.services.index.searches


async def test_agent_can_requery_but_search_tool_is_removed_after_two_searches(harness):
    h = harness
    h.settings.agent_max_searches = 2
    cid = await seed(h)
    decisions = []
    async def searching(messages, tools):
        available = {tool["function"]["name"] for tool in tools}
        decisions.append(available)
        if "assess_evidence" in available:
            data = json.loads(messages[-1]["content"])
            name, args = "assess_evidence", {"question": "可乐的容量和价格", "verdict": "sufficient",
                "relevant_chunk_ids": [h["chunk_id"] for h in data["evidence"]], "missing_aspects": []}
        elif "search_knowledge_base" in available:
            name, args = "search_knowledge_base", {"query": "可乐容量" if len(decisions) == 1 else "可乐330 ml价格"}
        else:
            name, args = "answer", {"question": "可乐的容量和价格", "mode": "answer"}
        return {"role": "assistant", "content": None, "tool_calls": [{"id": str(len(decisions)),
                "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    h.services.models.agent_step = searching
    assert "event: done" in (await ask(h, cid)).text
    assert len(decisions) == 5 and "search_knowledge_base" not in decisions[-1]
    async with h.database.sessions() as db:
        turn = await db.scalar(select(Turn).where(Turn.conversation_id == cid))
        assert [x["tool"] for x in turn.agent_state["trace"]].count("search_knowledge_base") == 2


async def test_content_filter_does_not_trigger_automatic_or_manual_resume(harness):
    h = harness
    cid = await seed(h)
    calls = []

    async def filtered(messages, *, max_tokens=None):
        calls.append(1)
        yield "未完成内容"
        raise GenerationInterrupted("content_filter")

    h.services.models.stream_answer = filtered
    response = await ask(h, cid)
    assert "event: done" not in response.text and len(calls) == 1
    turn = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"][-1]
    assert not turn["can_resume"] and not turn["draft_answer"] and not turn["answer"]
    continuation = await ask(h, cid, "继续")
    assert "event: done" in continuation.text and len(calls) == 1
    data = [d for p, d in h.services.models.calls if p == "agent"][-1]
    assert data["history"] == [] and data["continuation_target_id"] is None
