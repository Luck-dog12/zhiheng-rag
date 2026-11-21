"""Opt-in external capacity acceptance. --dry-run generates data without API calls."""
import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]


def document_text(run_id: str, index: int, characters: int) -> str:
    code = f"CAP-{run_id}-{index:06d}"
    text = f"# 容量测试设备 {code}\n\n设备编号 {code}，保修期为 {index % 36 + 1} 个月。\n\n"
    paragraph = "维护人员每月检查设备外观、供电和运行日志。发现异常时登记工单，并保留检查记录。维修结束后核验功能，再由值班人员确认。\n\n"
    while len(text) < characters:
        text += paragraph
    return text


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


class Journal:
    """Append-only progress; tolerate and truncate only an interrupted final record."""
    def __init__(self, path: Path, run_id: str, count: int, characters: int):
        self.path, self.records = path, []
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            lines = path.read_bytes().splitlines(keepends=True)
            good = 0
            for i, line in enumerate(lines):
                try:
                    self.records.append(json.loads(line))
                    good += len(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1:
                        raise ValueError("容量测试日志中间记录损坏，停止恢复") from None
                    with path.open("r+b") as file:
                        file.truncate(good)
            if good and not path.read_bytes().endswith(b"\n"):
                with path.open("ab") as file:
                    file.write(b"\n")
        header = {"event": "header", "run_id": run_id, "count": count, "characters": characters}
        if self.records and self.records[0] != header:
            raise ValueError("恢复参数与原测试不一致，请使用原 run-id、documents 和 characters")
        if not self.records:
            self.append(header)

    def append(self, value):
        with self.path.open("ab") as file:
            file.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
            file.flush()
            os.fsync(file.fileno())
        self.records.append(value)


class CapacityRunner:
    def __init__(self, client, journal: Journal, *, concurrency=4, poll_seconds=10):
        self.client, self.journal = client, journal
        self.config = journal.records[0]
        self.slots = asyncio.Semaphore(concurrency)
        self.poll_seconds = poll_seconds
        self.kb_id = next((r["kb_id"] for r in journal.records if r["event"] == "knowledge_base"), None)
        self.uploads = {r["index"]: r for r in journal.records if r["event"] == "uploaded"}
        self.finished = {r["index"]: r for r in journal.records if r["event"] == "finished"}
        self.upload_errors = {}

    async def call(self, method, path, **kwargs):
        response = await self.client.request(method, "/api/v1" + path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def prepare(self):
        await self.call("GET", "/health/ready")
        libraries = await self.call("GET", "/knowledge-bases")
        name = "RAG 容量测试 " + self.config["run_id"]
        if self.kb_id:
            if not any(k["id"] == self.kb_id and k["name"] == name for k in libraries):
                raise ValueError("原测试知识库不存在或无权访问；请核对服务地址和测试账号")
            return
        kb = next((k for k in libraries if k["name"] == name), None)
        if kb is None:
            kb = await self.call("POST", "/knowledge-bases", json={"name": name, "description": "合成容量测试数据，请勿作为业务依据"})
        self.kb_id = kb["id"]
        self.journal.append({"event": "knowledge_base", "kb_id": self.kb_id})

    async def upload_all(self):
        async def upload(index):
            async with self.slots:
                started = time.time()
                body = document_text(self.config["run_id"], index, self.config["characters"])
                try:
                    value = await self.call("POST", "/documents", data={"kb_id": self.kb_id},
                        files={"file": (f"capacity-{index:06d}.md", body.encode(), "text/markdown")})
                    record = {"event": "uploaded", "index": index, "submitted_at": started,
                              "upload_seconds": time.time() - started, **value}
                    self.journal.append(record)
                    self.uploads[index] = record
                    self.upload_errors.pop(index, None)
                except httpx.HTTPError as exc:
                    self.upload_errors[index] = type(exc).__name__
        # Bounded task creation even for 10,000 documents.
        for offset in range(0, self.config["count"], 100):
            await asyncio.gather(*(upload(i) for i in range(offset, min(offset + 100, self.config["count"]))
                                   if i not in self.uploads))

    async def wait_jobs(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            pending = [i for i in self.uploads if i not in self.finished]
            if not pending:
                return
            async def inspect(index):
                async with self.slots:
                    upload = self.uploads[index]
                    try:
                        job = await self.call("GET", "/jobs/" + upload["job_id"])
                    except httpx.HTTPError:
                        return  # Network failures remain pending, never count as successful jobs.
                    if job["status"] in ("done", "failed", "superseded"):
                        record = {"event": "finished", "index": index, "status": job["status"],
                            "error_code": job.get("error_code"), "observed_at": time.time(),
                            "observed_ingestion_seconds": time.time() - upload["submitted_at"]}
                        self.journal.append(record)
                        self.finished[index] = record
            for offset in range(0, len(pending), 100):
                if time.monotonic() >= deadline:
                    return
                await asyncio.gather(*(inspect(i) for i in pending[offset:offset + 100]))
            if len(self.finished) < len(self.uploads):
                await asyncio.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))

    async def measure_chats(self, samples=20):
        completed = sorted(i for i, value in self.finished.items() if value["status"] == "done")
        count = min(samples, len(completed))
        selected = [completed[round(i * (len(completed) - 1) / max(1, count - 1))] for i in range(count)]

        async def ask(index):
            started = time.perf_counter()
            record = {"index": index, "success": False, "citation_document_hit": False}
            try:
                conversation = await self.call("POST", "/conversations")
                response = await self.client.post("/api/v1/chat", json={
                    "conversation_id": conversation["id"], "kb_ids": [self.kb_id],
                    "question": f"编号 CAP-{self.config['run_id']}-{index:06d} 的设备保修期是多少个月？"})
                response.raise_for_status()
                events = response.text.replace("\r\n", "\n").split("\n\n")
                record["success"] = any(e.startswith("event: done\n") for e in events)
                for event in events:
                    if event.startswith("event: citations\ndata: "):
                        citations = json.loads(event.split("data: ", 1)[1])["citations"]
                        record["citation_document_hit"] = any(c["document_id"] == self.uploads[index]["document_id"] for c in citations)
                    if event.startswith("event: error\ndata: "):
                        record["error_code"] = json.loads(event.split("data: ", 1)[1]).get("code")
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                record["error_type"] = type(exc).__name__
            record["seconds"] = time.perf_counter() - started
            return record
        return await asyncio.gather(*(ask(i) for i in selected))

    def report(self, chats):
        durations = [r["seconds"] for r in chats]
        succeeded = sum(r["success"] for r in chats)
        return {"mode": "live_external_services", "run_id": self.config["run_id"], "kb_id": self.kb_id,
            "requested_documents": self.config["count"], "uploaded_documents": len(self.uploads),
            "ready_documents": sum(r["status"] == "done" for r in self.finished.values()),
            "failed_jobs": [r for r in self.finished.values() if r["status"] != "done"],
            "pending_jobs": len(self.uploads) - len(self.finished), "upload_errors": self.upload_errors,
            "concurrent_chats": len(chats), "successful_chats": succeeded,
            "chat_success_rate": succeeded / len(chats) if chats else None,
            "p50_seconds": percentile(durations, .5), "p95_seconds": percentile(durations, .95),
            "chats": chats, "limitation": "Synthetic corpus; citation document matching does not establish answer correctness. Ingestion times in the journal include polling delay."}


async def main(args):
    if args.dry_run:
        total = sum(len(document_text(args.run_id, i, args.characters).encode()) for i in range(args.documents))
        print(json.dumps({"mode": "dry_run_no_services_called", "documents": args.documents,
                          "total_utf8_bytes": total, "run_id": args.run_id}, indent=2))
        return 0
    missing = [key for key in ("EVAL_USERNAME", "EVAL_PASSWORD") if not os.environ.get(key)]
    if missing:
        raise ValueError("缺少测试账号配置：" + ", ".join(missing))
    journal = Journal(ROOT / "output/capacity" / f"{args.run_id}.jsonl", args.run_id, args.documents, args.characters)
    async with httpx.AsyncClient(base_url=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000"), timeout=300) as client:
        runner = CapacityRunner(client, journal, concurrency=args.upload_concurrency, poll_seconds=args.poll_seconds)
        login = await runner.call("POST", "/auth/login", json={"username": os.environ["EVAL_USERNAME"], "password": os.environ["EVAL_PASSWORD"]})
        client.headers["X-CSRF-Token"] = login["csrf_token"]
        await runner.prepare()
        print(f"run-id={args.run_id}; isolated knowledge base={runner.kb_id}", flush=True)
        await runner.upload_all()
        print(f"uploaded={len(runner.uploads)}; waiting for Worker", flush=True)
        await runner.wait_jobs(args.wait_minutes * 60)
        chats = await runner.measure_chats(20) if args.chat else []
        report = runner.report(chats)
        path = journal.path.with_suffix(".report.json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in ("chats", "failed_jobs", "upload_errors")}, ensure_ascii=False, indent=2))
        print(f"report={path}")
        return 0 if report["ready_documents"] == args.documents and all(r["success"] for r in chats) else 1


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=uuid4().hex[:12], help="Reuse the same ID and parameters to resume progress")
    parser.add_argument("--documents", type=int, default=10000)
    parser.add_argument("--characters", type=int, default=2400)
    parser.add_argument("--upload-concurrency", type=int, default=4, choices=range(1, 21))
    parser.add_argument("--poll-seconds", type=int, default=10, choices=range(1, 61))
    parser.add_argument("--wait-minutes", type=int, default=120)
    parser.add_argument("--chat", action="store_true", help="Run up to 20 simultaneous chats after ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Generate and count data without connecting to services")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]{1,40}", args.run_id):
        parser.error("run-id must be 1-40 letters, digits or hyphens")
    if not 1 <= args.documents <= 100000 or not 400 <= args.characters <= 20000 or args.wait_minutes < 1:
        parser.error("documents: 1-100000; characters: 400-20000; wait-minutes: positive")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(arguments())))
    except (ValueError, httpx.HTTPError) as error:
        print(f"Capacity run stopped: {type(error).__name__}. Check service configuration and the saved journal.")
        raise SystemExit(1) from None
