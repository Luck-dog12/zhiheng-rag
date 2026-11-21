from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from rag.config import Settings


def utcnow() -> datetime:
    return datetime.now(UTC)


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class Database:
    def __init__(self, settings: Settings):
        kwargs = {} if settings.app_env == "test" else {"pool_size": 10, "max_overflow": 10}
        self.engine = create_async_engine(settings.database_url, pool_pre_ping=True, **kwargs)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def close(self):
        await self.engine.dispose()
