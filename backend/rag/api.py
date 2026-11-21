import asyncio
import hashlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from rag.chat import ChatEngine, visible_conversation_titles, visible_turns
from rag.config import Settings, get_settings
from rag.db import Database, utcnow
from rag.errors import AppError
from rag.models import (
    AuditEvent,
    Chunk,
    ChunkMetadata,
    Conversation,
    Document,
    DocumentVersion,
    Grant,
    Job,
    KnowledgeBase,
    LoginSession,
    Turn,
    User,
    new_id,
)
from rag.processing import validate_file
from rag.schemas import (
    AuthView,
    ChatInput,
    ChunkList,
    ChunkView,
    ConversationDetail,
    ConversationView,
    DocumentList,
    GrantInput,
    GrantView,
    JobView,
    KBCreate,
    KBView,
    LoginInput,
    SearchInput,
    SearchView,
    UploadView,
    UserCreate,
    UserUpdate,
    UserView,
    VersionView,
)
from rag.security import (
    DUMMY_HASH,
    admin_user,
    audit,
    check_rate,
    current_user,
    get_db,
    hasher,
    permitted_kbs,
    require_kbs,
    token_hash,
    verify_password,
)
from rag.services import Services
from rag.telemetry import LATENCY, REQUESTS

logger = logging.getLogger("rag.api")
router = APIRouter(prefix="/api/v1")


def user_view(user):
    return {"id": user.id, "username": user.username, "display_name": user.display_name,
            "role": user.role, "active": user.active}


def job_view(job):
    return {"id": job.id, "document_id": job.document_id, "version_id": job.version_id,
            "kind": job.kind, "status": job.status, "stage": job.stage, "attempts": job.attempts,
            "error_code": job.error_code, "error_message": job.error_message,
            "created_at": job.created_at, "updated_at": job.updated_at}


async def accessible_document(db, user, document_id, *, lock=False):
    stmt = select(Document).where(Document.id == str(document_id), Document.deleted_at.is_(None))
    if lock:
        stmt = stmt.with_for_update()
    doc = await db.scalar(stmt)
    if not doc:
        raise AppError("not_found", "文档不存在", 404)
    await require_kbs(db, user, [doc.kb_id])
    return doc


@router.post("/auth/login", response_model=AuthView)
async def login(body: LoginInput, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    origin = request.headers.get("origin")
    if origin and origin != request.app.state.settings.app_origin:
        raise AppError("origin_invalid", "登录请求来源无效", 403)
    identity = hashlib.sha256((request.client.host if request.client else "unknown").encode()).hexdigest()
    await check_rate(db, "login:" + identity, limit=20, seconds=300)
    user = await db.scalar(select(User).where(User.username == body.username.strip().lower()))
    valid = await verify_password(body.password, user.password_hash if user else DUMMY_HASH)
    if not valid or not user or not user.active:
        audit(db, user.id if user else None, "auth.failed", trace_id=request.state.trace_id)
        await db.commit()
        raise AppError("invalid_credentials", "用户名或密码错误", 401)
    token, csrf = secrets.token_urlsafe(48), secrets.token_hex(32)
    expires = utcnow() + timedelta(hours=request.app.state.settings.session_hours)
    db.add(LoginSession(token_hash=token_hash(token), user_id=user.id, csrf_token=csrf, expires_at=expires))
    await db.execute(delete(LoginSession).where(LoginSession.expires_at < utcnow()))
    audit(db, user.id, "auth.login", trace_id=request.state.trace_id)
    await db.commit()
    response.set_cookie("rag_session", token, httponly=True, samesite="lax",
                        secure=request.app.state.settings.app_env == "production",
                        max_age=request.app.state.settings.session_hours * 3600, path="/")
    return {"user": user_view(user), "csrf_token": csrf}


@router.get("/auth/me", response_model=AuthView)
async def me(request: Request, user: User = Depends(current_user)):
    return {"user": user_view(user), "csrf_token": request.state.login.csrf_token}


@router.post("/auth/logout", status_code=204)
async def logout(request: Request, response: Response, user: User = Depends(current_user),
                 db: AsyncSession = Depends(get_db)):
    await db.delete(request.state.login)
    audit(db, user.id, "auth.logout", trace_id=request.state.trace_id)
    await db.commit()
    response.delete_cookie("rag_session", path="/")


@router.get("/users", response_model=list[UserView])
async def users(user: User = Depends(admin_user), db: AsyncSession = Depends(get_db)):
    return [user_view(u) for u in (await db.scalars(select(User).order_by(User.created_at))).all()]


@router.post("/users", status_code=201, response_model=UserView)
async def create_user(body: UserCreate, request: Request, user: User = Depends(admin_user),
                      db: AsyncSession = Depends(get_db)):
    item = User(username=body.username.strip().lower(), display_name=body.display_name,
                password_hash=await asyncio.to_thread(hasher.hash, body.password), role=body.role)
    db.add(item)
    await db.flush()
    audit(db, user.id, "user.created", item.id, trace_id=request.state.trace_id)
    await db.commit()
    return user_view(item)


@router.patch("/users/{user_id}", response_model=UserView)
async def update_user(user_id: UUID, body: UserUpdate, request: Request,
                      user: User = Depends(admin_user), db: AsyncSession = Depends(get_db)):
    if str(user_id) == user.id and not body.active:
        raise AppError("self_deactivation", "不能停用当前登录管理员")
    item = await db.get(User, str(user_id))
    if not item:
        raise AppError("not_found", "用户不存在", 404)
    item.active = body.active
    if not body.active:
        await db.execute(delete(LoginSession).where(LoginSession.user_id == item.id))
    audit(db, user.id, "user.updated", item.id, {"active": body.active}, request.state.trace_id)
    await db.commit()
    return user_view(item)


@router.get("/knowledge-bases", response_model=list[KBView])
async def knowledge_bases(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    ids = await permitted_kbs(db, user)
    items = (await db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(ids))
                              .order_by(KnowledgeBase.created_at))).all()
    counts = dict((await db.execute(select(Document.kb_id, func.count()).where(
        Document.kb_id.in_(ids), Document.deleted_at.is_(None)).group_by(Document.kb_id))).all())
    return [{"id": k.id, "name": k.name, "description": k.description, "document_count": counts.get(k.id, 0)} for k in items]


@router.post("/knowledge-bases", status_code=201, response_model=KBView)
async def create_kb(body: KBCreate, request: Request, user: User = Depends(admin_user),
                    db: AsyncSession = Depends(get_db)):
    item = KnowledgeBase(**body.model_dump())
    db.add(item)
    await db.flush()
    audit(db, user.id, "knowledge_base.created", item.id, trace_id=request.state.trace_id)
    await db.commit()
    return {"id": item.id, "name": item.name, "description": item.description, "document_count": 0}


@router.patch("/knowledge-bases/{kb_id}", response_model=KBView)
async def update_kb(kb_id: UUID, body: KBCreate, request: Request, user: User = Depends(admin_user),
                    db: AsyncSession = Depends(get_db)):
    kb = await db.get(KnowledgeBase, str(kb_id))
    if not kb:
        raise AppError("not_found", "知识库不存在", 404)
    kb.name, kb.description = body.name, body.description
    audit(db, user.id, "knowledge_base.updated", kb.id, trace_id=request.state.trace_id)
    await db.commit()
    return {"id": kb.id, "name": kb.name, "description": kb.description}


@router.get("/knowledge-bases/{kb_id}/grants", response_model=GrantView)
async def get_grants(kb_id: UUID, user: User = Depends(admin_user), db: AsyncSession = Depends(get_db)):
    await require_kbs(db, user, [str(kb_id)])
    return {"user_ids": list((await db.scalars(select(Grant.user_id).where(Grant.kb_id == str(kb_id)))).all())}


@router.put("/knowledge-bases/{kb_id}/grants", response_model=GrantView)
async def set_grants(kb_id: UUID, body: GrantInput, request: Request, user: User = Depends(admin_user),
                     db: AsyncSession = Depends(get_db)):
    kb = await db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == str(kb_id)).with_for_update())
    if not kb:
        raise AppError("not_found", "知识库不存在", 404)
    ids = list({str(i) for i in body.user_ids})
    found = list((await db.scalars(select(User.id).where(User.id.in_(ids)))).all())
    if len(found) != len(ids):
        raise AppError("invalid_users", "授权列表包含不存在的用户")
    await db.execute(delete(Grant).where(Grant.kb_id == str(kb_id)))
    db.add_all([Grant(kb_id=str(kb_id), user_id=i) for i in ids])
    audit(db, user.id, "knowledge_base.grants", str(kb_id), {"user_ids": ids}, request.state.trace_id)
    await db.commit()
    return {"user_ids": ids}


@router.get("/documents", response_model=DocumentList)
async def documents(kb_id: UUID | None = None, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100),
                     user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    ids = await require_kbs(db, user, [str(kb_id)]) if kb_id else await permitted_kbs(db, user)
    condition = (Document.kb_id.in_(ids), Document.deleted_at.is_(None))
    total = await db.scalar(select(func.count()).select_from(Document).where(*condition))
    rows = (await db.execute(select(Document, DocumentVersion).outerjoin(
        DocumentVersion, Document.latest_version_id == DocumentVersion.id).where(*condition)
        .order_by(Document.created_at.desc()).offset(offset).limit(limit))).all()
    return {"total": total, "items": [{"id": d.id, "kb_id": d.kb_id, "title": d.title,
            "current_version_id": d.current_version_id, "latest_version_id": d.latest_version_id,
            "status": v.status if v else "queued", "chunk_count": v.chunk_count if v else 0,
            "warnings": v.warnings if v else [], "created_at": d.created_at} for d, v in rows]}


@router.post("/documents", status_code=202, response_model=UploadView)
async def upload_document(request: Request, kb_id: UUID = Form(...), file: UploadFile = File(...),
                          document_id: UUID | None = Form(None), user: User = Depends(admin_user),
                          db: AsyncSession = Depends(get_db)):
    settings, services = request.app.state.settings, request.app.state.services
    await require_kbs(db, user, [str(kb_id)])
    services.models.configured()
    services.storage.configured()
    services.index.configured()
    maximum = settings.upload_limit_mb * 1024 * 1024
    data = await file.read(maximum + 1)
    if len(data) > maximum:
        raise AppError("upload_too_large", f"单个文件不能超过 {settings.upload_limit_mb} MB", 413)
    filename = (file.filename or "document.txt").replace("\\", "/").rsplit("/", 1)[-1][:255]
    await asyncio.to_thread(validate_file, filename, data)
    digest = hashlib.sha256(data).hexdigest()
    # KB lock serializes new-document hash deduplication; updates also lock the document.
    await db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == str(kb_id)).with_for_update())
    if document_id:
        doc = await accessible_document(db, user, document_id, lock=True)
        if doc.kb_id != str(kb_id):
            raise AppError("knowledge_base_mismatch", "文档与知识库不匹配")
    else:
        doc = await db.scalar(select(Document).join(DocumentVersion, DocumentVersion.document_id == Document.id)
                              .where(Document.kb_id == str(kb_id), Document.deleted_at.is_(None),
                                     DocumentVersion.file_hash == digest,
                                     DocumentVersion.fingerprint == settings.processing_fingerprint).limit(1))
        if not doc:
            doc = Document(id=new_id(), kb_id=str(kb_id), title=filename)
            db.add(doc)
            await db.flush()
    existing = await db.scalar(select(DocumentVersion).where(DocumentVersion.document_id == doc.id,
        DocumentVersion.file_hash == digest, DocumentVersion.fingerprint == settings.processing_fingerprint))
    if existing:
        job = await db.scalar(select(Job).where(Job.version_id == existing.id, Job.kind == "ingest")
                               .order_by(Job.created_at.desc()).limit(1))
        return {"document_id": doc.id, "version_id": existing.id, "job_id": job.id if job else None,
                "deduplicated": True}
    version_id = new_id()
    key = f"documents/{doc.id}/{version_id}/original{Path(filename).suffix.lower()}"
    await services.storage.put(key, data, file.content_type or "application/octet-stream")
    version = DocumentVersion(id=version_id, document_id=doc.id, filename=filename, file_hash=digest,
        fingerprint=settings.processing_fingerprint, embedding_fingerprint=settings.embedding_fingerprint,
        object_key=key)
    db.add(version)
    await db.flush()
    doc.latest_version_id = version_id
    job = Job(document_id=doc.id, version_id=version_id)
    db.add(job)
    await db.flush()
    audit(db, user.id, "document.uploaded", doc.id, {"version_id": version_id}, request.state.trace_id)
    await db.commit()
    return {"document_id": doc.id, "version_id": version_id, "job_id": job.id, "deduplicated": False}


@router.get("/documents/{document_id}/versions", response_model=list[VersionView])
async def versions(document_id: UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    doc = await accessible_document(db, user, document_id)
    stmt = select(DocumentVersion).where(DocumentVersion.document_id == doc.id)
    if user.role != "admin":
        stmt = stmt.where(DocumentVersion.id == doc.current_version_id)
    return [{"id": v.id, "filename": v.filename, "status": v.status, "chunk_count": v.chunk_count,
             "warnings": v.warnings, "created_at": v.created_at, "current": v.id == doc.current_version_id}
            for v in (await db.scalars(stmt.order_by(DocumentVersion.created_at.desc()))).all()]


@router.get("/documents/{document_id}/download")
async def download(document_id: UUID, request: Request, version_id: UUID | None = None,
                   user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    doc = await accessible_document(db, user, document_id)
    selected = str(version_id) if version_id else doc.current_version_id
    if not selected or (user.role != "admin" and selected != doc.current_version_id):
        raise AppError("not_found", "文档版本尚不可访问", 404)
    version = await db.get(DocumentVersion, selected)
    if not version or version.document_id != doc.id:
        raise AppError("not_found", "文档版本不存在", 404)
    data = await request.app.state.services.storage.get(version.object_key)
    audit(db, user.id, "document.downloaded", doc.id, {"version_id": version.id}, request.state.trace_id)
    await db.commit()
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(version.filename)}"})


@router.delete("/documents/{document_id}", status_code=202)
async def delete_document(document_id: UUID, request: Request, user: User = Depends(admin_user),
                          db: AsyncSession = Depends(get_db)):
    doc = await accessible_document(db, user, document_id, lock=True)
    doc.deleted_at, doc.current_version_id, doc.latest_version_id = utcnow(), None, None
    job = Job(document_id=doc.id, kind="delete")
    db.add(job)
    audit(db, user.id, "document.deleted", doc.id, trace_id=request.state.trace_id)
    await db.commit()
    return {"job_id": job.id}


@router.get("/jobs", response_model=list[JobView])
async def jobs(document_id: UUID | None = None, limit: int = Query(100, ge=1, le=200),
               user: User = Depends(admin_user), db: AsyncSession = Depends(get_db)):
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if document_id:
        stmt = stmt.where(Job.document_id == str(document_id))
    return [job_view(job) for job in (await db.scalars(stmt)).all()]


@router.get("/jobs/{job_id}", response_model=JobView)
async def job_detail(job_id: UUID, user: User = Depends(admin_user), db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, str(job_id))
    if not job:
        raise AppError("not_found", "任务不存在", 404)
    return job_view(job)


@router.post("/jobs/{job_id}/retry", response_model=JobView)
async def retry_job(job_id: UUID, request: Request, user: User = Depends(admin_user),
                     db: AsyncSession = Depends(get_db)):
    job = await db.scalar(select(Job).where(Job.id == str(job_id)).with_for_update())
    if not job or job.status != "failed":
        raise AppError("job_not_retryable", "仅失败任务可以重试", 409)
    if job.kind == "ingest":
        doc = await accessible_document(db, user, job.document_id)
        if doc.latest_version_id != job.version_id:
            raise AppError("superseded", "已有更新的版本，请重试最新版本", 409)
    job.status, job.stage, job.attempts = "queued", "queued", 0
    job.error_code, job.error_message, job.available_at = None, None, utcnow()
    audit(db, user.id, "job.retried", job.id, trace_id=request.state.trace_id)
    await db.commit()
    return job_view(job)


@router.get("/chunks", response_model=ChunkList)
async def chunks(document_id: UUID, version_id: UUID | None = None, offset: int = Query(0, ge=0),
                 limit: int = Query(30, ge=1, le=100), user: User = Depends(current_user),
                 db: AsyncSession = Depends(get_db)):
    doc = await accessible_document(db, user, document_id)
    version = str(version_id) if version_id else doc.current_version_id
    if user.role != "admin" and version != doc.current_version_id:
        raise AppError("forbidden", "无权访问该版本", 403)
    rows = (await db.execute(select(Chunk, ChunkMetadata).join(ChunkMetadata)
        .where(Chunk.document_id == doc.id, Chunk.version_id == version)
        .order_by(Chunk.ordinal).offset(offset).limit(limit))).all()
    total = await db.scalar(select(func.count()).select_from(Chunk).where(
        Chunk.document_id == doc.id, Chunk.version_id == version))
    return {"total": total, "items": [chunk_view(c, m) for c, m in rows]}


def chunk_view(chunk, metadata):
    return {"chunk_id": chunk.id, "ordinal": chunk.ordinal, "version_id": chunk.version_id,
            "document_id": chunk.document_id, "body": chunk.body, "heading_path": chunk.heading_path,
            "sources": chunk.sources, "edits": chunk.edits,
            "metadata": {"summary": metadata.summary, "keywords": metadata.keywords,
                         "time_mentions": metadata.time_mentions, "locations": metadata.locations,
                         "model": metadata.model, "prompt_version": metadata.prompt_version}}


@router.get("/chunks/{chunk_id}", response_model=ChunkView)
async def chunk_detail(chunk_id: UUID, user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    chunk = await db.get(Chunk, str(chunk_id))
    if not chunk:
        raise AppError("not_found", "分块不存在", 404)
    doc = await accessible_document(db, user, chunk.document_id)
    if chunk.version_id != doc.current_version_id:
        raise AppError("source_changed", "此分块不属于当前文档版本", 409)
    metadata = await db.get(ChunkMetadata, chunk.id)
    return chunk_view(chunk, metadata)


@router.post("/retrieval/search", response_model=SearchView)
async def search(body: SearchInput, request: Request, user: User = Depends(current_user),
                 db: AsyncSession = Depends(get_db)):
    await check_rate(db, f"retrieval:{user.id}", limit=60)
    result = await request.app.state.services.retriever.search(db, user, body.question, [str(x) for x in body.kb_ids])
    audit(db, user.id, "retrieval.search", detail={"hits": len(result["hits"])}, trace_id=request.state.trace_id)
    await db.commit()
    return result


@router.get("/conversations", response_model=list[ConversationView])
async def conversations(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    items = list((await db.scalars(select(Conversation).where(Conversation.user_id == user.id)
                                  .order_by(Conversation.created_at.desc()).limit(100))).all())
    titles = await visible_conversation_titles(db, user, items)
    return [{"id": c.id, "title": titles[c.id], "created_at": c.created_at} for c in items]


@router.post("/conversations", status_code=201, response_model=ConversationView)
async def create_conversation(user: User = Depends(current_user), db: AsyncSession = Depends(get_db)):
    item = Conversation(user_id=user.id)
    db.add(item)
    await db.commit()
    return {"id": item.id, "title": item.title}


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def conversation_detail(conversation_id: UUID, user: User = Depends(current_user),
                               db: AsyncSession = Depends(get_db)):
    item = await db.get(Conversation, str(conversation_id))
    if not item or item.user_id != user.id:
        raise AppError("not_found", "对话不存在", 404)
    titles = await visible_conversation_titles(db, user, [item])
    return {"id": item.id, "title": titles[item.id], "turns": await visible_turns(db, user, item.id)}


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID, request: Request, user: User = Depends(current_user),
                              db: AsyncSession = Depends(get_db)):
    # Chat creation locks the same parent row, so a turn cannot start between this check and deletion.
    item = await db.scalar(select(Conversation).where(
        Conversation.id == str(conversation_id), Conversation.user_id == user.id).with_for_update())
    if not item:
        raise AppError("not_found", "对话不存在", 404)
    running = await db.scalar(select(Turn.id).where(Turn.conversation_id == item.id,
        Turn.status == "running", Turn.created_at > utcnow() - timedelta(minutes=10)).limit(1))
    if running:
        raise AppError("conversation_busy", "该对话仍有回答正在生成，请停止生成或稍后再删除", 409)
    await db.execute(delete(Turn).where(Turn.conversation_id == item.id))
    await db.delete(item)
    audit(db, user.id, "conversation.delete", target_id=str(conversation_id), trace_id=request.state.trace_id)
    await db.commit()
    return Response(status_code=204)


@router.post("/chat")
async def chat(body: ChatInput, request: Request, user: User = Depends(current_user),
               db: AsyncSession = Depends(get_db)):
    ids = [str(x) for x in body.kb_ids]
    await require_kbs(db, user, ids)
    item = await db.get(Conversation, str(body.conversation_id))
    if not item or item.user_id != user.id:
        raise AppError("not_found", "对话不存在", 404)
    await check_rate(db, f"chat:{user.id}", limit=30)
    return StreamingResponse(request.app.state.chat.events(user.id, str(body.conversation_id),
        body.question, ids, request.state.trace_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/audit")
async def audit_log(limit: int = Query(100, ge=1, le=200), user: User = Depends(admin_user),
                    db: AsyncSession = Depends(get_db)):
    rows = (await db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit))).all()
    return [{"id": a.id, "user_id": a.user_id, "action": a.action, "target_id": a.target_id,
             "detail": a.detail, "trace_id": a.trace_id, "created_at": a.created_at} for a in rows]


@router.get("/health/live")
async def live():
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request):
    state, checks = request.app.state, {}
    async def run(name, operation):
        try:
            await asyncio.wait_for(operation(), timeout=10)
            checks[name] = "ready"
        except Exception:
            checks[name] = "unavailable"

    async def database():
        async with state.db.sessions() as db:
            await db.execute(text("SELECT 1 FROM users LIMIT 1"))

    async def model_configuration():
        state.services.models.configured()

    await asyncio.gather(run("database", database), run("storage", state.services.storage.check),
                         run("milvus", state.services.index.check), run("model_configuration", model_configuration))
    ok = all(v == "ready" for v in checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@router.get("/metrics")
async def metrics(user: User = Depends(admin_user)):
    return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


def create_app(settings: Settings | None = None, services: Services | None = None, database: Database | None = None):
    settings = settings or get_settings()
    database, services = database or Database(settings), services or Services(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await services.close()
        await database.close()

    app = FastAPI(title="知衡 · 企业知识工作台", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.db, app.state.services = settings, database, services
    app.state.chat = ChatEngine(database, services, settings)
    app.add_middleware(CORSMiddleware, allow_origins=[settings.app_origin], allow_credentials=True,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                       allow_headers=["Content-Type", "X-CSRF-Token"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.trace_id = str(uuid4())
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.trace_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        route = getattr(request.scope.get("route"), "path", "unmatched")
        elapsed = time.monotonic() - started
        REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        LATENCY.labels(route).observe(elapsed)
        logger.info(json.dumps({"trace_id": request.state.trace_id, "route": route,
                               "status": response.status_code, "seconds": round(elapsed, 4)}))
        return response

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message,
                                      "trace_id": getattr(request.state, "trace_id", "")}}, status_code=exc.status)

    @app.exception_handler(IntegrityError)
    async def integrity_error(request, exc):
        return JSONResponse({"error": {"code": "conflict", "message": "记录已存在或数据状态发生变化，请刷新重试"}}, status_code=409)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        logger.error("database_error trace=%s type=%s", request.state.trace_id, type(exc).__name__)
        return JSONResponse({"error": {"code": "database_unavailable", "message": "数据库不可用或尚未执行迁移",
                                      "trace_id": request.state.trace_id}}, status_code=503)

    app.include_router(router)
    frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend.is_dir():
        @app.get("/{path:path}", include_in_schema=False)
        async def frontend_file(path: str):
            if path.startswith("api/"):
                raise AppError("not_found", "接口不存在", 404)
            target = (frontend / path).resolve()
            if not target.is_relative_to(frontend):
                raise AppError("not_found", "文件不存在", 404)
            return FileResponse(target if target.is_file() else frontend / "index.html")
    return app
