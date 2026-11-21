from datetime import timedelta

import pytest
from rag.chat import answer_urls_supported
from rag.db import utcnow
from rag.errors import LeaseLost
from rag.jobs import Pipeline, claim_job
from rag.models import Chunk, ChunkMetadata, Document, DocumentVersion, Grant, Job, Turn
from rag.retrieval import reciprocal_rank_fusion
from sqlalchemy import delete, select


def test_rrf_uses_rank_and_deduplicates():
    fused = reciprocal_rank_fusion({"dense": [{"chunk_id": "a", "score": 0.1}, {"chunk_id": "b", "score": 0.01}],
                                   "sparse": [{"chunk_id": "b", "score": 5000}, {"chunk_id": "b", "score": 4000}]})
    assert fused[0]["chunk_id"] == "b" and len(fused) == 2
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)


async def test_upload_ingest_search_chat_and_followup(harness):
    h = harness
    uploaded = await h.upload("# 可乐\n\n可乐是一种饮料。\n\n它的容量是 330 ml，价格为 5 元。")
    await h.drain()
    async with h.database.sessions() as db:
        doc = await db.get(Document, uploaded["document_id"])
        assert doc.current_version_id == uploaded["version_id"]
        chunks = list((await db.scalars(select(Chunk))).all())
        assert "可乐的容量" in chunks[0].body
        assert await db.get(ChunkMetadata, chunks[0].id)
    result = await h.client.post("/api/v1/retrieval/search", json={"question": "可乐的容量？", "kb_ids": [h.kb]})
    assert result.status_code == 200, result.text
    assert len(result.json()["hits"]) == 1
    conversation = (await h.client.post("/api/v1/conversations")).json()["id"]
    response = await h.client.post("/api/v1/chat", json={"question": "可乐容量是多少？", "kb_ids": [h.kb], "conversation_id": conversation})
    assert "event: done" in response.text and "event: citations" in response.text
    response = await h.client.post("/api/v1/chat", json={"question": "价格呢？", "kb_ids": [h.kb], "conversation_id": conversation})
    assert "event: done" in response.text
    assert any(p == "agent" and len(v["history"]) == 1 for p, v in h.services.models.calls)
    turns = (await h.client.get(f"/api/v1/conversations/{conversation}")).json()["turns"]
    assert len(turns) == 2 and turns[0]["citations"][0]["sources"][0]["original"]


async def test_csrf_and_member_write_restrictions(harness):
    h = harness
    response = await h.client.post("/api/v1/knowledge-bases", json={"name": "bad"}, headers={"X-CSRF-Token": "wrong"})
    assert response.status_code == 403
    await h.login("member")
    assert (await h.client.post("/api/v1/knowledge-bases", json={"name": "bad"})).status_code == 403
    assert (await h.client.get("/api/v1/users")).status_code == 403
    denied = await h.client.post("/api/v1/retrieval/search", json={"question": "秘密", "kb_ids": [h.private_kb]})
    assert denied.status_code == 403 and not h.services.index.searches


async def test_no_cross_knowledge_or_conversation_access(harness):
    h = harness
    private = await h.upload("董事会奖金秘密", kb=h.private_kb)
    await h.drain()
    conversation = (await h.client.post("/api/v1/conversations")).json()["id"]
    await h.login("member")
    assert (await h.client.get(f"/api/v1/documents/{private['document_id']}/download")).status_code == 403
    assert (await h.client.get(f"/api/v1/conversations/{conversation}")).status_code == 404
    response = await h.client.post("/api/v1/retrieval/search", json={"question": "董事会", "kb_ids": [h.kb]})
    assert response.json()["hits"] == []
    assert all(x["kb_ids"] == [h.kb] for x in h.services.index.searches)


async def test_permission_revocation_redacts_history_and_blocks_sources(harness):
    h = harness
    uploaded = await h.upload("可乐产品定价为5元。")
    await h.drain()
    await h.login("member")
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    await h.client.post("/api/v1/chat", json={"question": "可乐价格", "kb_ids": [h.kb], "conversation_id": cid})
    async with h.database.sessions() as db, db.begin():
        await db.execute(delete(Grant).where(Grant.user_id == h.member_id))
    history = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"]
    assert history[0]["status"] == "redacted" and not history[0]["answer"] and not history[0]["citations"]
    listing = (await h.client.get("/api/v1/conversations")).json()
    assert listing[0]["title"] == "资料已不可访问的对话"
    assert (await h.client.get(f"/api/v1/conversations/{cid}")).json()["title"] == "资料已不可访问的对话"
    assert (await h.client.get(f"/api/v1/documents/{uploaded['document_id']}/download")).status_code == 403


async def test_reupload_idempotent_and_failed_new_version_keeps_previous(harness):
    h = harness
    original = await h.upload("第一版可乐价格为5元。")
    await h.drain()
    duplicate = await h.upload("第一版可乐价格为5元。")
    assert duplicate["deduplicated"] and duplicate["version_id"] == original["version_id"]
    replacement = await h.upload("第二版可乐价格为6元。", document_id=original["document_id"])
    h.services.index.fail_verify = True
    claim = await claim_job(h.database, h.settings)
    await Pipeline(h.database, h.services, h.settings).execute(*claim)
    async with h.database.sessions() as db:
        doc = await db.get(Document, original["document_id"])
        assert doc.current_version_id == original["version_id"]
    response = await h.client.post("/api/v1/retrieval/search", json={"question": "可乐价格", "kb_ids": [h.kb]})
    assert all(hit["version_id"] == original["version_id"] for hit in response.json()["hits"])
    h.services.index.fail_verify = False
    async with h.database.sessions() as db, db.begin():
        job = await db.get(Job, replacement["job_id"])
        job.available_at = utcnow()
    await h.drain()
    async with h.database.sessions() as db:
        assert (await db.get(Document, original["document_id"])).current_version_id == replacement["version_id"]


async def test_delete_immediately_invalidates_retrieval_and_references(harness):
    h = harness
    uploaded = await h.upload("可乐秘密资料")
    await h.drain()
    chunk_id = next(iter(h.services.index.rows))
    assert (await h.client.delete(f"/api/v1/documents/{uploaded['document_id']}")).status_code == 202
    assert h.services.index.rows  # async cleanup has not run, SQL must still exclude it
    response = await h.client.post("/api/v1/retrieval/search", json={"question": "可乐", "kb_ids": [h.kb]})
    assert response.json()["hits"] == []
    assert (await h.client.get(f"/api/v1/chunks/{chunk_id}")).status_code == 404
    await h.drain()
    assert not h.services.index.rows and not h.services.storage.objects


async def test_expired_lease_reclaimed_stale_worker_cannot_publish(harness):
    h = harness
    uploaded = await h.upload("可乐租约恢复测试")
    first = await claim_job(h.database, h.settings)
    async with h.database.sessions() as db, db.begin():
        job = await db.get(Job, uploaded["job_id"])
        job.lease_until = utcnow() - timedelta(seconds=1)
    second = await claim_job(h.database, h.settings)
    assert first[0] == second[0] and first[1] != second[1]
    pipeline = Pipeline(h.database, h.services, h.settings)
    with pytest.raises(LeaseLost):
        await pipeline.stage(*first, "metadata")
    await pipeline.execute(*second)
    async with h.database.sessions() as db:
        assert (await db.get(Document, uploaded["document_id"])).current_version_id


async def test_fabricated_citation_marks_turn_failed(harness):
    h = harness
    await h.upload("可乐是饮料")
    await h.drain()
    h.services.models.answer_text = "资料表明可乐是饮料。[99]"
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    response = await h.client.post("/api/v1/chat", json={"question": "可乐", "kb_ids": [h.kb], "conversation_id": cid})
    assert "invalid_citation" in response.text and "event: done" not in response.text
    async with h.database.sessions() as db:
        turn = await db.scalar(select(Turn).where(Turn.conversation_id == cid))
        assert turn.status == "failed" and not turn.answer


@pytest.mark.parametrize("answer,valid", [
    ("接口为 `https://{WorkspaceId}.example.com/v1`。[1]", True),
    ("[接口](https://{WorkspaceId}.example.com/v1)。[1]", True),
    ("“[接口](https://{WorkspaceId}.example.com/v1)”页面。[1]", True),
    ("[接口](https://{WorkspaceId}.example.com/v1)页面。[1]", True),
    ("接口为 https://{WorkspaceId}.example.com/v1。 [1]", True),
    ("接口为 https://{WorkspaceId}.example.com/v1/extra。[1]", False),
    ("接口为 https://{WorkspaceId}.example.com。[1]", False),
    ("接口为 https://real-workspace.example.com/v1。[1]", False),
    ("接口为 https://{WorkspaceId}.example.com/v1?token=made-up。[1]", False),
    ("接口为 https://{WorkspaceId}.example.com.evil/v1。[1]", False),
    ("接口为 HTTPS://invented.example/v1。[1]", False),
])
async def test_answer_url_requires_exact_cited_evidence(harness, answer, valid):
    h = harness
    await h.upload("# 接口地址\n\n接口为 `https://{WorkspaceId}.example.com/v1`。")
    await h.drain()
    h.services.models.answer_text = answer
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    response = await h.client.post("/api/v1/chat", json={"question": "接口地址", "kb_ids": [h.kb],
                                                       "conversation_id": cid})
    assert ("event: done" in response.text) == valid
    history = (await h.client.get(f"/api/v1/conversations/{cid}")).json()["turns"]
    assert history[0]["status"] == ("completed" if valid else "failed")
    assert history[0]["answer"] == (answer if valid else "")


def test_url_in_uncited_evidence_is_not_allowed():
    hits = [{"body": "接口说明"}, {"body": "https://example.com/path_(literal)"}]
    answer = "[接口](https://example.com/path_(literal)) [1]"
    assert not answer_urls_supported(answer, hits, {1})
    assert answer_urls_supported(answer, hits, {2})
    assert not answer_urls_supported(answer, hits, set())


async def test_empty_knowledge_abstains_without_answer_model(harness):
    h = harness
    cid = (await h.client.post("/api/v1/conversations")).json()["id"]
    response = await h.client.post("/api/v1/chat", json={"question": "完全无资料的问题", "kb_ids": [h.kb], "conversation_id": cid})
    assert "没有找到足够的资料" in response.text and "event: done" in response.text


async def test_health_and_openapi(harness):
    response = await harness.client.get("/openapi.json")
    assert len(response.json()["paths"]) >= 20
    assert (await harness.client.get("/api/v1/health/ready")).status_code == 200


async def test_exhausted_lease_marks_version_failed(harness):
    h = harness
    uploaded = await h.upload("可乐重试上限")
    async with h.database.sessions() as db, db.begin():
        job = await db.get(Job, uploaded["job_id"])
        job.attempts, job.status = 3, "running"
        job.lease_until = utcnow() - timedelta(seconds=1)
        (await db.get(DocumentVersion, uploaded["version_id"])).status = "indexing"
    assert await claim_job(h.database, h.settings) is None
    async with h.database.sessions() as db:
        assert (await db.get(Job, uploaded["job_id"])).status == "failed"
        assert (await db.get(DocumentVersion, uploaded["version_id"])).status == "failed"


async def test_publication_records_cleanup_without_running_another_job(harness):
    h = harness
    uploaded = await h.upload("可乐发布与清理")
    await Pipeline(h.database, h.services, h.settings).execute(*await claim_job(h.database, h.settings))
    async with h.database.sessions() as db:
        assert (await db.get(Document, uploaded["document_id"])).current_version_id == uploaded["version_id"]
        gc = await db.scalar(select(Job).where(Job.document_id == uploaded["document_id"], Job.kind == "gc"))
        assert gc and gc.status == "queued"


async def test_delete_waits_for_running_ingestion(harness):
    h = harness
    uploaded = await h.upload("可乐删除时仍在入库")
    await claim_job(h.database, h.settings)
    await h.client.delete(f"/api/v1/documents/{uploaded['document_id']}")
    pipeline = Pipeline(h.database, h.services, h.settings)
    deletion = await claim_job(h.database, h.settings)
    await pipeline.execute(*deletion)
    async with h.database.sessions() as db:
        job = await db.get(Job, deletion[0])
        assert job.status == "queued" and job.attempts == 0 and job.error_code == "cleanup_waiting"
    assert h.services.storage.objects
