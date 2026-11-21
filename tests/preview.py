"""Explicitly isolated browser fixture, never a production deployment command.

Run: uv run python -m tests.preview
Login: preview / preview-password-2026
"""
import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from rag.api import create_app
from rag.config import Settings
from rag.db import Database
from rag.jobs import Pipeline, claim_job, run_worker
from rag.models import Base, User
from rag.security import hasher
from rag.services import Services

from tests.fakes import FakeIndex, FakeModels, FakeStorage


def create_preview():
    root = Path(__file__).resolve().parents[1]
    output = root / "output"
    output.mkdir(exist_ok=True)
    settings = Settings(_env_file=None, app_env="test", app_origin="http://127.0.0.1:8000",
                        database_url=f"sqlite+aiosqlite:///{output / ('preview-' + uuid4().hex + '.db')}")
    database = Database(settings)
    models = FakeModels()
    models.delay = 0.025
    services = Services(settings, models=models, storage=FakeStorage(), index=FakeIndex())
    app = create_app(settings, services, database)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            async with database.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            async with database.sessions() as db, db.begin():
                db.add(User(username="preview", display_name="演示管理员", role="admin",
                            password_hash=hasher.hash("preview-password-2026")))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as client:
                login = await client.post("/api/v1/auth/login", json={"username": "preview", "password": "preview-password-2026"})
                client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
                kb = await client.post("/api/v1/knowledge-bases", json={"name": "员工服务与制度", "description": "汇集差旅、休假、入职及行政服务规则，让日常协作更顺畅。"})
                products = await client.post("/api/v1/knowledge-bases", json={"name": "产品与服务手册", "description": "产品规格、使用指南和客户支持标准，帮助团队准确回应每一个问题。"})
                corpus = json.loads((root / "evals/corpus.json").read_text(encoding="utf-8"))
                for doc in corpus:
                    target = products.json()["id"] if doc["id"] in ("atlas", "coffee", "support") else kb.json()["id"]
                    response = await client.post("/api/v1/documents", data={"kb_id": target},
                                                files={"file": (doc["title"] + ".md", doc["text"].encode(), "text/markdown")})
                    response.raise_for_status()
            pipeline = Pipeline(database, services, settings)
            while claim := await claim_job(database, settings):
                await pipeline.execute(*claim)
            worker = asyncio.create_task(run_worker(database, services, settings))
            print("TEST FIXTURE ONLY - external providers are simulated. Login: preview / preview-password-2026")
            try:
                yield
            finally:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
    app.router.lifespan_context = lifespan
    return app


if __name__ == "__main__":
    uvicorn.run("tests.preview:create_preview", factory=True, host="127.0.0.1", port=8000, log_level="warning")
