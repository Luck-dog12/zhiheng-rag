import json

import pytest
from rag.models import Turn, User
from sqlalchemy import select

from tests.test_agent import ask, events, seed


def call(name, args, step):
    return {"role": "assistant", "content": None, "reasoning_content": "private-reasoning",
        "tool_calls": [{"id": str(step), "type": "function", "function": {
            "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


async def test_mismatch_requires_reassessment_requery_and_combines_complementary_evidence(harness):
    h = harness
    cid = await seed(h)
    await h.upload("# 茶饮说明\n\n茶饮容量为500 ml，单价8元。", filename="tea.md")
    await h.drain()
    search = h.services.retriever.search
    async def narrow(db, user, question, kb_ids, **kwargs):
        result = await search(db, user, question, kb_ids, **kwargs)
        marker = "可乐" if "可乐" in question else "茶饮"
        result["hits"] = [hit for hit in result["hits"] if marker in hit["body"]]
        return result
    h.services.retriever.search = narrow
    step = 0
    seen = []
    async def decide(messages, tools):
        nonlocal step
        step += 1
        available = {t["function"]["name"] for t in tools}
        if step == 1:
            return call("search_knowledge_base", {"query": "可乐容量价格", "strategy": "hybrid"}, step)
        if step == 2:
            assert "answer" not in available
            # Deliberately attempt to bypass assessment; it must not execute.
            return call("answer", {"question": "对比可乐和茶饮", "mode": "answer"}, step)
        if step in (3, 5):
            pool = next(json.loads(m["content"])["evidence"] for m in reversed(messages)
                        if m["role"] == "tool" and "evidence" in json.loads(m["content"]))
            seen[:] = [x["chunk_id"] for x in pool]
            return call("assess_evidence", {"question": "对比可乐和茶饮容量价格",
                "verdict": "partial" if step == 3 else "sufficient", "relevant_chunk_ids": seen,
                "missing_aspects": ["茶饮容量价格"] if step == 3 else []}, step)
        if step == 4:
            assert "answer" not in available and "ask_clarification" not in available
            return call("search_knowledge_base", {"query": "茶饮容量价格", "strategy": "sparse"}, step)
        assert step == 6 and len(seen) == 2 and "answer" in available
        return call("answer", {"question": "对比可乐和茶饮容量价格", "mode": "answer", "chunk_ids": seen}, step)
    h.services.models.agent_step = decide
    response = await ask(h, cid, "对比可乐和茶饮容量价格")
    assert "event: done" in response.text and "private-reasoning" not in response.text
    assert [x["strategy"] for x in h.services.index.searches] == ["hybrid", "sparse"]
    data = [d for p, d in h.services.models.calls if p == "answer"][-1]
    assert len(data["evidence"]) == 2 and data["evidence_status"] == "sufficient"
    async with h.database.sessions() as db:
        turn = await db.scalar(select(Turn).where(Turn.conversation_id == cid))
        assert "private-reasoning" not in json.dumps(turn.agent_state)
        assert any(t.get("rejected") for t in turn.agent_state["trace"])


async def test_irrelevant_results_stop_at_search_budget_without_generating_from_them(harness):
    h = harness
    cid = await seed(h)
    searches = 0
    async def decide(messages, tools):
        nonlocal searches
        names = {t["function"]["name"] for t in tools}
        if "assess_evidence" in names:
            return call("assess_evidence", {"question": "工厂检修时间", "verdict": "irrelevant",
                "relevant_chunk_ids": [], "missing_aspects": ["检修时间"]}, len(messages))
        if "search_knowledge_base" in names:
            searches += 1
            return call("search_knowledge_base", {"query": f"检修计划{searches}",
                "strategy": ["hybrid", "sparse", "dense", "hybrid"][searches-1]}, len(messages))
        return call("answer", {"question": "工厂检修时间", "mode": "answer"}, len(messages))
    h.services.models.agent_step = decide
    response = await ask(h, cid, "工厂何时检修？")
    assert searches == 4 and any(e == "done" for e, _ in events(response.text))
    assert "可乐" not in response.text and not [c for c in h.services.models.calls if c[0] == "answer"]


async def test_final_evidence_limit_applies_to_explicit_selection_across_searches(harness):
    h = harness
    h.settings.answer_top_k = 1
    cid = await seed(h)
    await h.upload("# 茶饮\n\n茶饮容量500 ml。", filename="tea.md")
    await h.drain()
    step, selected = 0, []
    async def decide(messages, tools):
        nonlocal step, selected
        step += 1
        if step in (1, 3):
            return call("search_knowledge_base", {"query": "可乐330" if step == 1 else "茶饮500"}, step)
        if step in (2, 4):
            pool = json.loads(messages[-1]["content"])["evidence"]
            selected = [hit["chunk_id"] for hit in pool]
            return call("assess_evidence", {"question": "对比可乐茶饮容量", "verdict": "partial" if step == 2 else "sufficient",
                "relevant_chunk_ids": selected, "missing_aspects": ["茶饮"] if step == 2 else []}, step)
        if step == 5:
            assert len(selected) == 2
            return call("answer", {"question": "对比容量", "mode": "answer", "chunk_ids": selected}, step)
        assert step == 6 and "error" in json.loads(messages[-1]["content"])
        return call("answer", {"question": "对比容量", "mode": "answer", "chunk_ids": selected[:1]}, step)
    h.services.models.agent_step = decide
    response = await ask(h, cid, "对比可乐茶饮容量")
    assert "event: done" in response.text
    answer = [d for p, d in h.services.models.calls if p == "answer"][-1]
    assert len(answer["evidence"]) == 1


@pytest.mark.parametrize("strategy", ["sparse", "dense"])
async def test_search_strategy_selects_actual_path_and_bm25_skips_unused_embeddings(harness, strategy):
    h = harness
    await seed(h)
    if strategy == "sparse":
        h.services.models.embedding_failure = True
    else:
        h.settings.retrieval_sparse_mode = "model"
    async with h.database.sessions() as db:
        user = await db.get(User, h.admin_id)
        result = await h.services.retriever.search(db, user, "可乐330", [h.kb], strategy=strategy)
    assert result["hits"] and all(set(c["paths"]) == {strategy} for c in result["hits"])
