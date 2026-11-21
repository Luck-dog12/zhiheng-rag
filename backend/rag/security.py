import asyncio
import hashlib
import secrets
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, Request
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag.db import aware, utcnow
from rag.errors import AppError
from rag.models import AuditEvent, Grant, KnowledgeBase, LoginSession, RateBucket, User

hasher = PasswordHasher()
DUMMY_HASH = hasher.hash("timing-equalization-only-not-a-login")


def token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def get_db(request: Request):
    async with request.app.state.db.sessions() as db:
        yield db


async def current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    token = request.cookies.get("rag_session")
    if not token:
        raise AppError("unauthenticated", "请先登录", 401)
    login = await db.get(LoginSession, token_hash(token))
    if not login or aware(login.expires_at) <= utcnow():
        raise AppError("unauthenticated", "登录已过期，请重新登录", 401)
    user = await db.get(User, login.user_id)
    if not user or not user.active:
        raise AppError("unauthenticated", "账号已停用", 401)
    request.state.login = login
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        csrf = request.headers.get("X-CSRF-Token", "")
        if not csrf or not secrets.compare_digest(csrf, login.csrf_token):
            raise AppError("csrf_invalid", "请求校验失败，请刷新页面重试", 403)
    return user


async def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise AppError("forbidden", "此操作仅限管理员", 403)
    return user


async def permitted_kbs(db: AsyncSession, user: User) -> list[str]:
    if user.role == "admin":
        return list((await db.scalars(select(KnowledgeBase.id))).all())
    return list((await db.scalars(select(Grant.kb_id).where(Grant.user_id == user.id))).all())


async def require_kbs(db: AsyncSession, user: User, ids: list[str]) -> list[str]:
    permitted = set(await permitted_kbs(db, user))
    requested = list(dict.fromkeys(ids))
    if not requested or not set(requested) <= permitted:
        raise AppError("forbidden", "无权访问所选知识库", 403)
    return requested


async def verify_password(password: str, password_hash: str) -> bool:
    try:
        return await asyncio.to_thread(hasher.verify, password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


async def check_rate(db: AsyncSession, identity: str, *, limit: int, seconds: int = 60):
    now = utcnow()
    bucket = int(now.timestamp()) // seconds
    key = f"{identity}:{bucket}"
    insert = sqlite_insert if db.bind.dialect.name == "sqlite" else pg_insert
    stmt = insert(RateBucket).values(key=key, count=1, expires_at=now + timedelta(seconds=seconds * 2))
    stmt = stmt.on_conflict_do_update(index_elements=[RateBucket.key], set_={"count": RateBucket.count + 1})
    count = (await db.execute(stmt.returning(RateBucket.count))).scalar_one()
    await db.execute(delete(RateBucket).where(RateBucket.expires_at < now))
    await db.commit()
    if count > limit:
        raise AppError("rate_limited", "请求过于频繁，请稍后重试", 429)


def audit(db: AsyncSession, user_id: str | None, action: str, target_id: str | None = None,
          detail: dict | None = None, trace_id: str = ""):
    db.add(AuditEvent(user_id=user_id, action=action, target_id=target_id,
                      detail=detail or {}, trace_id=trace_id))
