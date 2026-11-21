import asyncio
import json
import logging
import time
from contextlib import suppress
from datetime import timedelta
from uuid import UUID, uuid4, uuid5

from sqlalchemy import and_, func, or_, select, update

from rag.db import aware, utcnow
from rag.errors import AppError, LeaseLost
from rag.models import AuditEvent, Chunk, ChunkMetadata, Document, DocumentVersion, Job
from rag.processing import Parsed, make_chunks, parse_document, resolve_document, summarize
from rag.telemetry import JOBS, STAGES

logger = logging.getLogger("rag.worker")


async def claim_job(database, settings) -> tuple[str, str] | None:
    now = utcnow()
    async with database.sessions() as db, db.begin():
        job = await db.scalar(select(Job).where(or_(
            and_(Job.status == "queued", Job.available_at <= now),
            and_(Job.status == "running", Job.lease_until < now),
        )).order_by(Job.created_at).with_for_update(skip_locked=True).limit(1))
        if not job:
            return None
        if job.attempts >= 3:
            job.status, job.error_code, job.error_message = "failed", "attempts_exhausted", "任务重试次数已用尽"
            job.lease_token, job.lease_until, job.updated_at = None, None, now
            if job.version_id:
                version = await db.get(DocumentVersion, job.version_id)
                if version and version.status != "ready":
                    version.status = "failed"
            return None
        token = str(uuid4())
        job.status, job.lease_token = "running", token
        job.lease_until = now + timedelta(seconds=settings.job_lease_seconds)
        job.attempts += 1
        job.updated_at = now
        return job.id, token


async def locked_job(db, job_id, token) -> Job:
    job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
    if (not job or job.lease_token != token or job.status != "running"
            or not job.lease_until or aware(job.lease_until) <= utcnow()):
        raise LeaseLost()
    return job


class Pipeline:
    def __init__(self, database, services, settings):
        self.db, self.services, self.settings = database, services, settings

    async def stage(self, job_id, token, name):
        async with self.db.sessions() as db, db.begin():
            job = await locked_job(db, job_id, token)
            job.stage, job.updated_at = name, utcnow()
            if job.version_id:
                version = await db.get(DocumentVersion, job.version_id)
                if version.status != "ready":
                    version.status = name

    async def process(self, job_id: str, token: str):
        async with self.db.sessions() as db:
            job = await locked_job(db, job_id, token)
            doc = await db.get(Document, job.document_id)
            version = await db.get(DocumentVersion, job.version_id) if job.version_id else None
            kind = job.kind
        if kind == "delete":
            await self._delete(doc.id)
            await self._finish(job_id, token, "done")
            return
        if doc.deleted_at or doc.latest_version_id != version.id:
            await self._finish(job_id, token, "superseded")
            return
        if version.fingerprint != self.settings.processing_fingerprint:
            raise AppError("pipeline_configuration_changed", "任务配置已改变，请重新上传为新版本", 409)
        self.services.models.configured()
        await self.services.index.check()
        if version.processed_key:
            parsed = Parsed.deserialize(json.loads(await self.services.storage.get(version.processed_key)))
        else:
            await self.stage(job_id, token, "parsing")
            data = await self.services.storage.get(version.object_key)
            with STAGES.labels("parsing").time():
                parsed = await asyncio.to_thread(parse_document, version.filename, data)
            await self.stage(job_id, token, "coreference")
            with STAGES.labels("coreference").time():
                parsed = await resolve_document(parsed, self.services.models,
                    normalize_whitespace=self.settings.clean_normalize_whitespace,
                    remove_fillers=self.settings.clean_remove_fillers)
            key = f"documents/{doc.id}/{version.id}/processed.json"
            await self.services.storage.put(key, json.dumps(parsed.serialize(), ensure_ascii=False).encode(),
                                             "application/json")
            async with self.db.sessions() as db, db.begin():
                await locked_job(db, job_id, token)
                v = await db.get(DocumentVersion, version.id)
                v.processed_key, v.warnings = key, parsed.warnings
        await self.stage(job_id, token, "metadata")
        pieces = make_chunks(parsed.blocks, self.settings.chunk_target, self.settings.chunk_max,
                             self.settings.chunk_overlap)
        if not pieces:
            raise AppError("empty_document", "清洗后没有可入库正文")
        chunk_ids = []
        for ordinal, piece in enumerate(pieces):
            chunk_id = str(uuid5(UUID(version.id), str(ordinal)))
            chunk_ids.append(chunk_id)
            async with self.db.sessions() as db:
                exists = await db.get(ChunkMetadata, chunk_id)
            if exists:
                continue
            with STAGES.labels("metadata").time():
                card = await summarize(piece, self.services.models)
            async with self.db.sessions() as db, db.begin():
                await locked_job(db, job_id, token)
                if not await db.get(Chunk, chunk_id):
                    db.add(Chunk(id=chunk_id, kb_id=doc.kb_id, document_id=doc.id, version_id=version.id,
                                 ordinal=ordinal, body=piece.body, heading_path=piece.heading_path,
                                 sources=piece.sources, edits=piece.edits))
                    await db.flush()
                db.add(ChunkMetadata(chunk_id=chunk_id, **card.model_dump(), model=self.settings.deepseek_model))
        await self.stage(job_id, token, "indexing")
        for offset in range(0, len(pieces), 32):
            batch = pieces[offset:offset + 32]
            sparse_vectors = None
            if self.settings.embedding_store_sparse:
                embeddings = await self.services.models.embed_vectors([p.body for p in batch], include_sparse=True)
                vectors, sparse_vectors = embeddings.dense, embeddings.sparse
            else:
                vectors = await self.services.models.embed([p.body for p in batch])
            await self.services.index.upsert([
                {"chunk_id": chunk_ids[offset + i], "kb_id": doc.kb_id, "document_id": doc.id,
                 "version_id": version.id, "body": p.body, "dense": vectors[i],
                 **({"model_sparse": sparse_vectors[i]} if sparse_vectors is not None else {})}
                for i, p in enumerate(batch)])
        await self.services.index.verify(chunk_ids)
        # Serialize publication against document uploads/deletion; the lease token fences stale workers.
        async with self.db.sessions() as db, db.begin():
            job = await locked_job(db, job_id, token)
            current = await db.scalar(select(Document).where(Document.id == doc.id).with_for_update())
            v = await db.get(DocumentVersion, version.id)
            if current.deleted_at or current.latest_version_id != version.id:
                job.status, job.stage = "superseded", "superseded"
            else:
                count = await db.scalar(select(func.count()).select_from(ChunkMetadata).join(Chunk)
                                        .where(Chunk.version_id == version.id))
                if count != len(pieces):
                    raise AppError("metadata_incomplete", "分块元数据不完整，未发布文档", 502)
                current.current_version_id = version.id
                v.status, v.chunk_count = "ready", len(pieces)
                job.status, job.stage = "done", "done"
                db.add(AuditEvent(action="document.published", target_id=doc.id,
                                  detail={"version_id": version.id, "chunks": len(pieces)}))
            job.lease_token, job.lease_until, job.updated_at = None, None, utcnow()
            # Durable outbox: publication and its retryable cleanup task commit together.
            db.add(Job(document_id=doc.id, kind="gc", status="queued"))

    async def _delete(self, doc_id):
        async with self.db.sessions() as db:
            active = await db.scalar(select(Job.id).where(
                Job.document_id == doc_id, Job.kind == "ingest", Job.status == "running",
                Job.lease_until > utcnow(),
            ).limit(1))
            if active:
                raise AppError("cleanup_waiting", "等待当前入库任务结束后清理文件", 409)
        await self.services.index.delete_document(doc_id)
        async with self.db.sessions() as db:
            versions = list((await db.scalars(select(DocumentVersion).where(DocumentVersion.document_id == doc_id))).all())
        keys = [key for version in versions for key in (version.object_key, version.processed_key) if key]
        if keys:
            await self.services.storage.delete(keys)

    async def garbage_collect(self, job_id, token):
        async with self.db.sessions() as db:
            job = await locked_job(db, job_id, token)
            doc = await db.get(Document, job.document_id)
            ids = list((await db.scalars(select(DocumentVersion.id).where(
                DocumentVersion.document_id == doc.id,
                DocumentVersion.id.not_in([v for v in [doc.current_version_id, doc.latest_version_id] if v]),
            ))).all())
        await self.services.index.delete_versions(ids)
        await self._finish(job_id, token, "done")

    async def _finish(self, job_id, token, status):
        async with self.db.sessions() as db, db.begin():
            job = await locked_job(db, job_id, token)
            job.status, job.stage = status, status
            job.lease_token, job.lease_until, job.updated_at = None, None, utcnow()

    async def fail(self, job_id, token, exc):
        async with self.db.sessions() as db, db.begin():
            job = await db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if not job or job.lease_token != token:
                return
            code = exc.code if isinstance(exc, AppError) else "internal_error"
            message = exc.message if isinstance(exc, AppError) else "内部处理错误，请查看服务日志中的错误类型"
            retry = code in {"provider_unavailable", "storage_unavailable", "storage_delete_failed", "milvus_unavailable", "index_incomplete", "cleanup_waiting"}
            if code == "cleanup_waiting":
                job.attempts -= 1
            job.status = "queued" if retry and job.attempts < 3 else "failed"
            job.error_code, job.error_message = code, message
            job.available_at = utcnow() + timedelta(seconds=5 * 2 ** job.attempts)
            job.lease_token, job.lease_until, job.updated_at = None, None, utcnow()
            if job.version_id:
                version = await db.get(DocumentVersion, job.version_id)
                if version.status != "ready":
                    version.status = job.status
            logger.warning("job_failed id=%s code=%s type=%s", job_id, code, type(exc).__name__)

    async def execute(self, job_id, token):
        async def heartbeat():
            while True:
                await asyncio.sleep(self.settings.job_heartbeat_seconds)
                async with self.db.sessions() as db, db.begin():
                    result = await db.execute(update(Job).where(
                        Job.id == job_id, Job.lease_token == token, Job.status == "running",
                        Job.lease_until > utcnow(),
                    ).values(lease_until=utcnow() + timedelta(seconds=self.settings.job_lease_seconds),
                             updated_at=utcnow()))
                    if not result.rowcount:
                        raise LeaseLost()

        async def process():
            async with self.db.sessions() as db:
                kind = (await db.get(Job, job_id)).kind
            if kind == "gc":
                await self.garbage_collect(job_id, token)
            else:
                await self.process(job_id, token)

        work, beat = asyncio.create_task(process()), asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait({work, beat}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except Exception as exc:
            work.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await work
            await self.fail(job_id, token, exc)
        finally:
            for task in (work, beat):
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task


async def run_worker(database, services, settings):
    pipeline = Pipeline(database, services, settings)

    async def loop():
        last_metrics = 0
        while True:
            try:
                claim = await claim_job(database, settings)
                if claim:
                    await pipeline.execute(*claim)
                else:
                    await asyncio.sleep(1)
                if time.monotonic() - last_metrics > 15:
                    async with database.sessions() as db:
                        counts = dict((await db.execute(select(Job.status, func.count()).group_by(Job.status))).all())
                        for status in ("queued", "running", "failed", "done", "superseded"):
                            JOBS.labels(status).set(counts.get(status, 0))
                    last_metrics = time.monotonic()
            except Exception as exc:
                logger.error("worker_iteration_failed type=%s", type(exc).__name__)
                await asyncio.sleep(5)

    await asyncio.gather(*(loop() for _ in range(settings.worker_concurrency)))
