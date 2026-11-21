from datetime import timedelta
from uuid import uuid4

import pytest
from rag.db import utcnow
from rag.models import AuditEvent, Conversation, Document, Turn
from sqlalchemy import select


async def test_delete_conversation_removes_all_turns_but_preserves_documents_and_other_conversations(harness):
    h = harness
    uploaded = await h.upload("# 对话删除验证\n\n知识库原文必须保留。")
    await h.drain()
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    other = (await h.client.post("/api/v1/conversations")).json()["id"]
    response = await h.client.post("/api/v1/chat", json={
        "question": "原文是什么？", "kb_ids": [h.kb], "conversation_id": cid})
    assert "event: done" in response.text
    async with h.database.sessions() as db, db.begin():
        db.add(Turn(conversation_id=cid, question="失败轮次", status="failed",
                    agent_state={"draft": "未核验草稿", "recoverable": True}))
    response = await h.client.delete(f"/api/v1/conversations/{cid}")
    assert response.status_code == 204 and response.content == b""
    assert cid not in {c["id"] for c in (await h.client.get("/api/v1/conversations")).json()}
    assert (await h.client.get(f"/api/v1/conversations/{cid}")).status_code == 404
    assert (await h.client.delete(f"/api/v1/conversations/{cid}")).status_code == 404
    assert (await h.client.post("/api/v1/chat", json={
        "question": "继续", "kb_ids": [h.kb], "conversation_id": cid})).status_code == 404
    async with h.database.sessions() as db:
        assert await db.get(Conversation, cid) is None
        assert await db.scalar(select(Turn.id).where(Turn.conversation_id == cid)) is None
        assert await db.get(Conversation, other)
        assert (await db.get(Document, uploaded["document_id"])).current_version_id == uploaded["version_id"]
        event = await db.scalar(select(AuditEvent).where(AuditEvent.action == "conversation.delete"))
        assert event.target_id == cid and event.user_id == h.admin_id
        assert not event.detail  # No question, answer, or source text in the deletion audit event.
    assert h.services.index.rows and h.services.storage.objects


async def test_delete_requires_owner_and_csrf_even_for_administrator(harness):
    h = harness
    admin_cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    await h.login("member")
    assert (await h.client.delete(f"/api/v1/conversations/{admin_cid}")).status_code == 404
    member_cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    assert (await h.client.delete(f"/api/v1/conversations/{member_cid}", headers={"X-CSRF-Token": ""})).status_code == 403
    await h.login("admin")
    assert (await h.client.delete(f"/api/v1/conversations/{member_cid}")).status_code == 404
    await h.login("member")
    assert (await h.client.delete(f"/api/v1/conversations/{member_cid}")).status_code == 204
    assert (await h.client.delete(f"/api/v1/conversations/{uuid4()}")).status_code == 404
    h.client.cookies.clear()
    assert (await h.client.delete(f"/api/v1/conversations/{admin_cid}")).status_code == 401


async def test_delete_blocks_active_generation_but_allows_stale_interrupted_turn(harness):
    h = harness
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    async with h.database.sessions() as db, db.begin():
        turn = Turn(conversation_id=cid, question="正在生成", status="running")
        db.add(turn)
        await db.flush()
        turn_id = turn.id
    response = await h.client.delete(f"/api/v1/conversations/{cid}")
    assert response.status_code == 409 and response.json()["error"]["code"] == "conversation_busy"
    async with h.database.sessions() as db, db.begin():
        assert await db.get(Conversation, cid)
        (await db.get(Turn, turn_id)).created_at = utcnow() - timedelta(minutes=11)
    assert (await h.client.delete(f"/api/v1/conversations/{cid}")).status_code == 204


@pytest.mark.parametrize("top_k", [2, 5])
async def test_retrieval_api_distinguishes_selected_chunks_from_all_reranked_candidates(harness, top_k):
    h = harness
    h.settings.answer_top_k = top_k
    await h.upload("\n\n".join(f"# 配置章节 {i}\n\n可乐配置编号 {i}，容量 {300+i} ml。" for i in range(8)))
    await h.drain()
    response = await h.client.post("/api/v1/retrieval/search", json={"question": "可乐配置", "kb_ids": [h.kb]})
    assert response.status_code == 200
    result = response.json()
    assert result["top_k"] == top_k and result["sparse_mode"] == "bm25" and result["strategy"] == "hybrid"
    assert len(result["hits"]) == top_k < len(result["candidates"]) == 8
    assert result["hits"] == result["candidates"][:top_k]
    empty = (await h.client.post("/api/v1/retrieval/search", json={"question": "无资料", "kb_ids": [h.private_kb]})).json()
    assert empty["top_k"] == top_k and not empty["hits"] and not empty["candidates"]
