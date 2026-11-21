import asyncio
import json
import logging

import typer
from sqlalchemy import select, text

from rag.config import get_settings
from rag.db import Database
from rag.jobs import run_worker
from rag.models import User
from rag.security import hasher
from rag.services import Services

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def create_admin(username: str = typer.Option(..., prompt=True),
                 password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True),
                 display_name: str = "管理员"):
    """Create the first administrator after applying migrations; no default password."""
    if len(password) < 12:
        raise typer.BadParameter("密码至少 12 个字符")

    async def run():
        database = Database(get_settings())
        try:
            async with database.sessions() as db, db.begin():
                if await db.scalar(select(User).where(User.username == username.strip().lower())):
                    raise typer.BadParameter("用户已存在")
                db.add(User(username=username.strip().lower(), display_name=display_name,
                            password_hash=hasher.hash(password), role="admin"))
        finally:
            await database.close()
    asyncio.run(run())
    typer.echo("管理员已创建")


@app.command()
def init_index():
    """Explicitly create and validate the remote Milvus collection."""
    async def run():
        services = Services(get_settings())
        try:
            await services.index.initialize()
        finally:
            await services.close()
    asyncio.run(run())
    typer.echo("远程 Milvus 集合已就绪")


@app.command()
def check_services(probe_models: bool = False):
    """Check dependencies. --probe-models performs small billable model calls."""
    async def run():
        from rag.schemas import RewriteOutput

        services = Services(get_settings())
        database = Database(get_settings())
        results = {}
        failed = False
        try:
            services.models.configured()
            async def check_database():
                async with database.sessions() as db:
                    await db.execute(text("SELECT version_num FROM alembic_version"))
            for name, check in [("database", check_database), ("storage", services.storage.check), ("milvus", services.index.check)]:
                try:
                    await check()
                    results[name] = "ready"
                except Exception as exc:
                    failed = True
                    results[name] = getattr(exc, "message", type(exc).__name__)
            if probe_models:
                await services.models.embed_vectors(["企业知识库接口检查"],
                                                     include_sparse=get_settings().embedding_store_sparse)
                await services.models.rerank("可乐", ["可乐是一种饮料。"])
                for purpose in ("rewrite", "coreference"):
                    await services.models.structured(purpose, "返回问题本身，不需要澄清。",
                                                     {"question": "接口检查"}, RewriteOutput)
                results["models"] = "ready"
            else:
                results["models"] = "configuration_present_not_probed"
            typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
            if failed:
                raise typer.Exit(1)
        finally:
            await services.close()
            await database.close()
    asyncio.run(run())


@app.command()
def worker(metrics_port: int = 9001, metrics_host: str = "127.0.0.1"):
    """Run durable background ingestion, publication and deletion jobs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from prometheus_client import start_http_server

    metrics_server, _ = start_http_server(metrics_port, addr=metrics_host)

    async def run():
        settings = get_settings()
        database, services = Database(settings), Services(settings)
        try:
            await run_worker(database, services, settings)
        finally:
            await services.close()
            await database.close()
    try:
        asyncio.run(run())
    finally:
        metrics_server.shutdown()


if __name__ == "__main__":
    app()
