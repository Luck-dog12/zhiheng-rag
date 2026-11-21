"""Opt-in real PostgreSQL checks. No local database is installed or started."""
import asyncio
import os
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from rag.config import Settings
from rag.db import utcnow
from rag.errors import LeaseLost
from rag.jobs import Pipeline, claim_job
from rag.models import Base, Chunk, Document, DocumentVersion, Job, KnowledgeBase
from rag.services import Services
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tests.fakes import FakeIndex, FakeModels, FakeStorage

pytestmark = pytest.mark.postgres


@pytest.fixture
async def postgres_state():
    if os.environ.get("RAG_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("Set RAG_RUN_POSTGRES_TESTS=1 and TEST_POSTGRES_URL to use an external test database")
    url = os.environ.get("TEST_POSTGRES_URL", "")
    if not url.startswith("postgresql+asyncpg://"):
        pytest.fail("TEST_POSTGRES_URL must point to a dedicated external PostgreSQL test database", pytrace=False)
    # All identifiers are generated here, never obtained from an existing schema or user input.
    schema = "rag_acceptance_" + uuid4().hex
    administration = create_async_engine(url)
    engine = None
    created = False
    try:
        async with administration.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            created = True
        engine = create_async_engine(url, pool_size=10, max_overflow=10,
            connect_args={"server_settings": {"search_path": schema}})
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        database = SimpleNamespace(engine=engine, sessions=async_sessionmaker(engine, expire_on_commit=False))
        settings = Settings(_env_file=None, app_env="test", database_url=url)
        services = Services(settings, models=FakeModels(), storage=FakeStorage(), index=FakeIndex())
        try:
            yield database, settings, services
        finally:
            await services.close()
    finally:
        if engine:
            await engine.dispose()
        if created:
            # Only this fixture's freshly created, uniquely named schema is removed.
            async with administration.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await administration.dispose()


async def seed_jobs(state, count):
    database, settings, services = state
    jobs = []
    async with database.sessions() as db, db.begin():
        kb = KnowledgeBase(name="独立 PostgreSQL 测试")
        db.add(kb)
        await db.flush()
        for i in range(count):
            doc = Document(kb_id=kb.id, title=f"测试 {i}")
            db.add(doc)
            await db.flush()
            key = f"fixture/{uuid4().hex}.md"
            await services.storage.put(key, f"# 可乐测试 {i}\n\n可乐是一种饮料。\n\n它的容量是 330 ml。".encode())
            version = DocumentVersion(document_id=doc.id, filename=f"{i}.md", object_key=key,
                file_hash=str(i), fingerprint=settings.processing_fingerprint,
                embedding_fingerprint=settings.embedding_fingerprint)
            db.add(version)
            await db.flush()
            doc.latest_version_id = version.id
            job = Job(document_id=doc.id, version_id=version.id, created_at=utcnow() + timedelta(microseconds=i))
            db.add(job)
            await db.flush()
            jobs.append(job.id)
    return jobs


async def test_skip_locked_leaves_locked_job_for_another_worker(postgres_state):
    database, settings, _ = postgres_state
    ids = await seed_jobs(postgres_state, 2)
    async with database.sessions() as owner, owner.begin():
        await owner.scalar(select(Job).where(Job.id == ids[0]).with_for_update())
        claimed = await asyncio.wait_for(claim_job(database, settings), timeout=5)
        assert claimed[0] == ids[1]
    assert (await claim_job(database, settings))[0] == ids[0]


async def test_twenty_concurrent_claims_are_unique(postgres_state):
    database, settings, _ = postgres_state
    ids = await seed_jobs(postgres_state, 20)
    results = await asyncio.gather(*(claim_job(database, settings) for _ in ids))
    assert {job_id for job_id, _ in results} == set(ids)
    assert len({token for _, token in results}) == 20
    assert await claim_job(database, settings) is None


async def test_restart_fences_old_lease_and_publishes_once(postgres_state):
    database, settings, services = postgres_state
    ids = await seed_jobs(postgres_state, 1)
    first = await claim_job(database, settings)
    async with database.sessions() as db, db.begin():
        job = await db.get(Job, ids[0])
        job.lease_until = utcnow() - timedelta(seconds=1)
    second = await claim_job(database, settings)
    assert first[1] != second[1]
    pipeline = Pipeline(database, services, settings)
    with pytest.raises(LeaseLost):
        await pipeline.stage(*first, "metadata")
    await pipeline.execute(*second)
    async with database.sessions() as db:
        job = await db.get(Job, ids[0])
        document = await db.get(Document, job.document_id)
        assert job.status == "done" and document.current_version_id == job.version_id
        assert await db.scalar(select(func.count()).select_from(Chunk)) == 1
        assert await db.scalar(select(func.count()).select_from(Job).where(Job.kind == "gc")) == 1
