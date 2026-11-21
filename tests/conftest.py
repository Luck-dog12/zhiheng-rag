import os
from dataclasses import dataclass

import httpx
import pytest
from rag.api import create_app
from rag.config import Settings
from rag.db import Database
from rag.jobs import Pipeline, claim_job
from rag.models import Base, Grant, KnowledgeBase, User
from rag.security import hasher
from rag.services import Services
from sqlalchemy import event

from tests.fakes import FakeIndex, FakeModels, FakeStorage


@pytest.fixture(autouse=True)
def isolate_service_environment(monkeypatch):
    # pymilvus imports load_dotenv(). Never let a developer's real .env change mock tests.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    field_names = {name.upper() for name in Settings.model_fields}
    for key in list(os.environ):
        if key.upper() in field_names:
            monkeypatch.delenv(key)


@dataclass
class Harness:
    settings: Settings
    database: Database
    services: Services
    client: httpx.AsyncClient
    kb: str
    private_kb: str
    admin_id: str
    member_id: str

    async def login(self, username="admin"):
        response = await self.client.post("/api/v1/auth/login", json={"username": username, "password": "testing-password-2026"})
        assert response.status_code == 200, response.text
        self.client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        return response.json()

    async def upload(self, text, *, kb=None, document_id=None, filename="sample.md"):
        data = {"kb_id": kb or self.kb}
        if document_id:
            data["document_id"] = document_id
        response = await self.client.post("/api/v1/documents", data=data,
                                          files={"file": (filename, text.encode(), "text/plain")})
        assert response.status_code == 202, response.text
        return response.json()

    async def drain(self, max_jobs=20):
        pipeline = Pipeline(self.database, self.services, self.settings)
        for _ in range(max_jobs):
            claimed = await claim_job(self.database, self.settings)
            if not claimed:
                break
            await pipeline.execute(*claimed)


@pytest.fixture
async def harness(tmp_path):
    settings = Settings(_env_file=None, app_env="test", database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    database = Database(settings)

    @event.listens_for(database.engine.sync_engine, "connect")
    def pragma(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with database.sessions() as db, db.begin():
        password = hasher.hash("testing-password-2026")
        admin = User(username="admin", display_name="管理员", password_hash=password, role="admin")
        member = User(username="member", display_name="团队成员", password_hash=password, role="member")
        kb, private = KnowledgeBase(name="公开资料"), KnowledgeBase(name="管理资料")
        db.add_all([admin, member, kb, private])
        await db.flush()
        db.add(Grant(kb_id=kb.id, user_id=member.id))
    services = Services(settings, models=FakeModels(), storage=FakeStorage(), index=FakeIndex())
    app = create_app(settings, services, database)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        h = Harness(settings, database, services, client, kb.id, private.id, admin.id, member.id)
        await h.login()
        yield h
    await services.close()
    await database.close()
