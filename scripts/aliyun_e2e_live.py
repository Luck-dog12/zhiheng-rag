"""Explicit real-service acceptance through the running API and independent worker.

Credentials are read from the ignored local output directory; never printed.
"""
import argparse
import asyncio
import hashlib
import json
import math
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/e2e-live"
STATE = OUT / "state.json"
LIFECYCLE_V1 = "# 星河测试饮料\n\n星河可乐是一种测试饮料。\n\n它的容量为330 ml，单价为5元，不含酒精。"
LIFECYCLE_V2 = "# 星河测试饮料\n\n星河可乐是一种测试饮料。\n\n它的容量为500 ml，单价为6元，不含酒精。"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Acceptance:
    def __init__(self):
        self.credentials = read(OUT / "credentials.json")
        self.state = read(STATE) if STATE.exists() else {"run_id": self.credentials["run_id"], "documents": {}}
        self.client = httpx.AsyncClient(base_url=self.credentials["base_url"], timeout=180)

    async def call(self, method, path, *, client=None, expected=None, **kwargs):
        response = await (client or self.client).request(method, "/api/v1" + path, **kwargs)
        if expected is not None:
            if response.status_code != expected:
                raise RuntimeError(f"Unexpected HTTP {response.status_code}: {method} {path}")
            return response
        if response.is_error:
            try:
                code = response.json().get("error", {}).get("code", "unknown")
            except ValueError:
                code = "non_json_error"
            raise RuntimeError(f"HTTP {response.status_code} {code}: {method} {path}")
        return response.json() if response.content else None

    async def login(self, client=None, credentials=None):
        client = client or self.client
        credentials = credentials or self.credentials
        result = await self.call("POST", "/auth/login", client=client,
            json={k: credentials[k] for k in ("username", "password")})
        client.headers["X-CSRF-Token"] = result["csrf_token"]
        return result

    async def upload(self):
        for key, name in (("kb_id", "阿里云文档真实验收"), ("lifecycle_kb_id", "版本与权限真实验收")):
            if key not in self.state:
                kb = await self.call("POST", "/knowledge-bases", json={"name": name + " " + self.state["run_id"],
                    "description": "本次真实端到端测试创建，保留供复核"})
                self.state[key] = kb["id"]
                save(STATE, self.state)
        source = ROOT / "test-documents/aliyun"
        documents = read(source / "manifest.json")["documents"]
        order = ["generate-temporary-api-key", "lifecycle", "error-code"]
        order.extend(d["id"] for d in documents if d["id"] not in order)
        for name in order:
            if name in self.state["documents"]:
                continue
            if name == "lifecycle":
                data, filename, kb_id = LIFECYCLE_V1.encode(), "lifecycle.md", self.state["lifecycle_kb_id"]
            else:
                document = next(d for d in documents if d["id"] == name)
                data = (source / document["markdown_file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != document["markdown_sha256"]:
                    raise ValueError("Snapshot checksum mismatch")
                filename, kb_id = name + ".md", self.state["kb_id"]
            result = await self.call("POST", "/documents", data={"kb_id": kb_id},
                files={"file": (filename, data, "text/markdown")})
            self.state["documents"][name] = {**result, "sha256": hashlib.sha256(data).hexdigest()}
            save(STATE, self.state)
            print(json.dumps({"uploaded": name, "job_id": result["job_id"]}), flush=True)

    async def status(self):
        result = []
        for name, doc in self.state["documents"].items():
            job = await self.call("GET", "/jobs/" + doc["job_id"])
            result.append({"document": name, **job})
        save(OUT / "ingestion-status.json", result)
        return result

    async def wait(self):
        deadline = time.monotonic() + 10800
        while True:
            statuses = await self.status()
            print(json.dumps([{k: x[k] for k in ("document", "status", "stage", "error_code")} for x in statuses],
                             ensure_ascii=False), flush=True)
            if any(x["status"] in ("failed", "superseded") for x in statuses):
                raise RuntimeError("An ingestion job failed; inspect ingestion-status.json")
            if all(x["status"] == "done" for x in statuses):
                return
            if time.monotonic() > deadline:
                raise TimeoutError("Ingestion deadline reached")
            await asyncio.sleep(30)

    async def chat(self, question, kb_id, *, client=None, conversation_id=None):
        client = client or self.client
        if conversation_id is None:
            conversation_id = (await self.call("POST", "/conversations", client=client))["id"]
        started = time.perf_counter()
        events, current_event = [], None
        async with client.stream("POST", "/api/v1/chat", json={"question": question,
                "kb_ids": [kb_id], "conversation_id": conversation_id}) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Chat HTTP {response.status_code}")
            trace = response.headers.get("X-Request-ID")
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:]
                elif line.startswith("data: "):
                    events.append({"event": current_event, "data": json.loads(line[6:])})
        errors = [e["data"] for e in events if e["event"] == "error"]
        if errors or not any(e["event"] == "done" for e in events):
            save(OUT / "last-chat-failure.json", {"question": question, "events": events, "trace_id": trace})
            raise RuntimeError("SSE chat failed: " + (errors[0]["code"] if errors else "missing_done"))
        detail = await self.call("GET", "/conversations/" + conversation_id, client=client)
        turn = detail["turns"][-1]
        if turn["status"] != "completed":
            raise RuntimeError("Conversation turn was not committed")
        for citation in turn["citations"]:
            chunk = await self.call("GET", "/chunks/" + citation["chunk_id"], client=client)
            if chunk["version_id"] != citation["version_id"] or not citation["sources"]:
                raise RuntimeError("Citation version or source mismatch")
        return {"question": question, "conversation_id": conversation_id, "turn": turn,
                "events": events, "seconds": time.perf_counter() - started, "trace_id": trace}

    async def wait_job(self, job_id):
        deadline = time.monotonic() + 10800
        while True:
            job = await self.call("GET", "/jobs/" + job_id)
            if job["status"] == "done":
                return job
            if job["status"] in ("failed", "superseded"):
                raise RuntimeError(f"Job {job['status']}: {job['error_code']}")
            if time.monotonic() > deadline:
                raise RuntimeError("Job deadline reached")
            await asyncio.sleep(5)

    async def new_member(self, kind):
        key = kind + "_credentials"
        if key not in self.credentials:
            creds = {"username": kind + "_" + self.state["run_id"], "password": secrets.token_urlsafe(24)}
            user = await self.call("POST", "/users", json={**creds, "display_name": "真实验收 " + kind,
                                                          "role": "member"})
            self.credentials[key] = {**creds, "user_id": user["id"]}
            save(OUT / "credentials.json", self.credentials)
        client = httpx.AsyncClient(base_url=self.credentials["base_url"], timeout=180)
        await self.login(client, self.credentials[key])
        return client, self.credentials[key]

    async def lifecycle(self):
        report = {"started_at": datetime.now(UTC).isoformat(), "checks": {}, "answers": []}
        path = OUT / "lifecycle.json"
        def passed(name):
            report["checks"][name] = True
            save(path, report)
            print(json.dumps({"passed": name}), flush=True)

        document = self.state["documents"]["lifecycle"]
        doc_id, kb_id = document["document_id"], self.state["lifecycle_kb_id"]
        member, creds = await self.new_member("member")
        try:
            await self.wait_job(document["job_id"])
            await self.call("POST", "/retrieval/search", client=member, expected=403,
                            json={"question": "星河可乐价格", "kb_ids": [kb_id]})
            await self.call("GET", "/documents/" + doc_id + "/download", client=member, expected=403)
            passed("unauthorized_retrieval_and_download_denied")
            await self.call("PUT", "/knowledge-bases/" + kb_id + "/grants", json={"user_ids": [creds["user_id"]]})
            chunks = await self.call("GET", "/chunks", params={"document_id": doc_id})
            first = chunks["items"][0]
            if not any(e["type"] == "coreference" for c in chunks["items"] for e in c["edits"]):
                raise RuntimeError("No coreference edit persisted for lifecycle document")
            if not any("它的容量" in s["original"] for c in chunks["items"] for s in c["sources"]):
                raise RuntimeError("Original pronoun evidence not preserved")
            passed("metadata_coreference_and_original_mapping")
            answer = await self.chat("星河可乐的容量和单价是多少？", kb_id, client=member)
            report["answers"].append(answer)
            if "330" not in answer["turn"]["answer"] or "5" not in answer["turn"]["answer"]:
                raise RuntimeError("Initial answer missed expected values")
            passed("initial_answer_and_citations")
            follow = await self.chat("它是否含酒精？", kb_id, client=member, conversation_id=answer["conversation_id"])
            report["answers"].append(follow)
            if not any(p in follow["turn"]["answer"] for p in ("不含", "不含有", "没有酒精", "无酒精")):
                raise RuntimeError("Follow-up failed to preserve negation")
            passed("followup_and_negation")
            download = await self.call("GET", "/documents/" + doc_id + "/download", client=member, expected=200)
            if hashlib.sha256(download.content).hexdigest() != document["sha256"]:
                raise RuntimeError("Downloaded original checksum differs")
            passed("original_object_download_checksum")
            await self.call("GET", "/conversations/" + answer["conversation_id"], expected=404)
            passed("cross_user_conversation_denied")
            await self.call("PUT", "/knowledge-bases/" + kb_id + "/grants", json={"user_ids": []})
            hidden = await self.call("GET", "/conversations/" + answer["conversation_id"], client=member)
            if any(t["status"] != "redacted" or t["answer"] or t["citations"] for t in hidden["turns"]):
                raise RuntimeError("Revoked history leaked evidence")
            await self.call("GET", "/chunks/" + first["chunk_id"], client=member, expected=403)
            await self.call("POST", "/chat", client=member, expected=403, json={"question": "继续介绍它",
                "kb_ids": [kb_id], "conversation_id": answer["conversation_id"]})
            passed("revocation_hides_history_and_blocks_followup_and_citations")
            await self.call("PUT", "/knowledge-bases/" + kb_id + "/grants", json={"user_ids": [creds["user_id"]]})
            update = await self.call("POST", "/documents", data={"kb_id": kb_id, "document_id": doc_id},
                files={"file": ("lifecycle.md", LIFECYCLE_V2.encode(), "text/markdown")})
            report["update"] = update
            save(path, report)
            await self.wait_job(update["job_id"])
            passed("updated_version_published")
            await self.call("GET", "/chunks/" + first["chunk_id"], client=member, expected=409)
            await self.call("GET", "/documents/" + doc_id + "/download", client=member, expected=404,
                            params={"version_id": document["version_id"]})
            hidden = await self.call("GET", "/conversations/" + answer["conversation_id"], client=member)
            if any(t["status"] != "redacted" for t in hidden["turns"]):
                raise RuntimeError("Historical answer still visible after source update")
            passed("old_versions_and_stale_citations_blocked")
            current = await self.chat("星河可乐目前的容量和单价是多少？", kb_id, client=member)
            report["answers"].append(current)
            if ("500" not in current["turn"]["answer"] or "6" not in current["turn"]["answer"]
                    or "330" in current["turn"]["answer"]
                    or any(c["version_id"] != update["version_id"] for c in current["turn"]["citations"])):
                raise RuntimeError("Updated answer used old content")
            passed("updated_answer_uses_only_current_version")
            deleted = await self.call("DELETE", "/documents/" + doc_id)
            report["delete"] = deleted
            await self.call("GET", "/documents/" + doc_id + "/download", client=member, expected=404)
            for c in current["turn"]["citations"]:
                await self.call("GET", "/chunks/" + c["chunk_id"], client=member, expected=404)
            hidden = await self.call("GET", "/conversations/" + current["conversation_id"], client=member)
            if any(t["status"] != "redacted" or t["answer"] for t in hidden["turns"]):
                raise RuntimeError("Deleted source leaked through history")
            result = await self.call("POST", "/retrieval/search", client=member,
                                    json={"question": "星河可乐", "kb_ids": [kb_id]})
            if result["hits"]:
                raise RuntimeError("Deleted document still retrieved")
            passed("delete_hides_download_citations_history_and_retrieval")
            await self.wait_job(deleted["job_id"])
            passed("async_index_and_object_cleanup_job_completed")
            await self.call("POST", "/auth/logout", client=member)
            await self.call("GET", "/auth/me", client=member, expected=401)
            passed("logout_invalidates_session")
            report["status"] = "completed"
            report["finished_at"] = datetime.now(UTC).isoformat()
            save(path, report)
        finally:
            await member.aclose()

    async def capacity(self):
        client, creds = await self.new_member("load")
        kb_id = self.state["kb_id"]
        previous = await self.call("GET", "/knowledge-bases/" + kb_id + "/grants")
        await self.call("PUT", "/knowledge-bases/" + kb_id + "/grants",
                        json={"user_ids": [*previous["user_ids"], creds["user_id"]]})
        try:
            conversations = [(await self.call("POST", "/conversations", client=client))["id"] for _ in range(20)]
            questions = ["临时 API Key 默认有效期是多少？", "临时 API Key 有效期的允许范围是多少？",
                         "临时 API Key 可以提前删除吗？", "临时 API Key 会继承原密钥的权限吗？"]
            async def one(i, conversation):
                started = time.perf_counter()
                try:
                    result = await self.chat(questions[i % 4], kb_id, client=client, conversation_id=conversation)
                    return {"success": True, **result}
                except Exception as exc:
                    return {"success": False, "seconds": time.perf_counter() - started,
                            "type": type(exc).__name__, "error": str(exc) if isinstance(exc, RuntimeError) else "request_failed"}
            results = await asyncio.gather(*(one(i, c) for i, c in enumerate(conversations)))
            times = sorted(r["seconds"] for r in results)
            report = {"mode": "real_external_services", "concurrency": 20, "requests": 20,
                      "successes": sum(r["success"] for r in results), "p50_seconds": times[9],
                      "p95_seconds": times[math.ceil(len(times) * .95) - 1], "results": results,
                      "timing_scope": "SSE request plus committed history and citation detail checks"}
            save(OUT / "concurrency-20.json", report)
            print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
            if report["successes"] != 20:
                raise RuntimeError("Real concurrent chat acceptance failed")
        finally:
            await client.aclose()

    async def evaluate(self, *, resume=False):
        await self.wait()
        report = {"mode": "live_api_postgresql_worker_s3_milvus_models", "status": "running",
                  "started_at": datetime.now(UTC).isoformat(), "documents": [], "retrieval": [], "answers": []}
        path = OUT / "evaluation.json"
        previous = read(path) if resume and path.exists() else None
        all_chunks = {}
        listed = await self.call("GET", "/documents", params={"kb_id": self.state["kb_id"], "limit": 100})
        if listed["total"] != 8 or any(d["status"] != "ready" for d in listed["items"]):
            raise RuntimeError("All eight source documents must be published")
        for name, doc in self.state["documents"].items():
            if name == "lifecycle":
                continue
            data = await self.call("GET", "/documents/" + doc["document_id"] + "/download", expected=200)
            if hashlib.sha256(data.content).hexdigest() != doc["sha256"]:
                raise RuntimeError("Original object checksum mismatch")
            chunks, offset = [], 0
            while True:
                page = await self.call("GET", "/chunks", params={"document_id": doc["document_id"],
                                                                "offset": offset, "limit": 100})
                chunks.extend(page["items"])
                if len(chunks) >= page["total"]:
                    break
                if not page["items"]:
                    raise RuntimeError("Missing metadata rows in chunk pagination")
                offset += len(page["items"])
            if (not chunks or any(len(c["body"]) > 2000 or not c["sources"] or not c["metadata"]["summary"].strip()
                                  or c["version_id"] != doc["version_id"] for c in chunks)):
                raise RuntimeError("Incomplete or invalid published chunks")
            all_chunks.update({c["chunk_id"]: c for c in chunks})
            report["documents"].append({"name": name, "document_id": doc["document_id"],
                "version_id": doc["version_id"], "chunks": len(chunks), "original_checksum_valid": True,
                "coreference_edits": sum(e["type"] == "coreference" for c in chunks for e in c["edits"])})
        questions = read(ROOT / "output/aliyun-live/questions.json")
        if previous:
            def versions(rows):
                return {(d["document_id"], d["version_id"], d["chunks"]) for d in rows}
            if versions(previous["documents"]) != versions(report["documents"]):
                raise RuntimeError("Cannot resume: published corpus changed")
            expected = {q["id"]: q for q in questions}
            for result in previous["retrieval"]:
                if result["id"] not in expected or any(result[k] != v for k, v in expected[result["id"]].items()):
                    raise RuntimeError("Cannot resume: evaluation questions changed")
            for result in previous["answers"]:
                if (result["case_id"] not in expected
                        or result["expected_answer"] != expected[result["case_id"]]["answer"]
                        or any(c["chunk_id"] not in all_chunks for c in result["turn"]["citations"])):
                    raise RuntimeError("Cannot resume: answer evidence changed")
            report = previous
            report.setdefault("resumed_at", []).append(datetime.now(UTC).isoformat())
            report["status"] = "running"
        save(path, report)
        save(OUT / "published-chunks.json", all_chunks)
        # Duplicate upload must reuse the published version and its original ingestion task.
        sample = self.state["documents"]["get-api-key"]
        duplicate = await self.call("POST", "/documents", data={"kb_id": self.state["kb_id"]},
            files={"file": ("get-api-key.md", (ROOT /
                   "test-documents/aliyun/documents/get-api-key.md").read_bytes(), "text/markdown")})
        if not duplicate["deduplicated"] or duplicate["version_id"] != sample["version_id"]:
            raise RuntimeError("Identical upload not deduplicated")
        report["duplicate_upload_idempotent"] = True
        for question in questions:
            if any(r["id"] == question["id"] for r in report["retrieval"]):
                continue
            start = time.perf_counter()
            result = await self.call("POST", "/retrieval/search", json={"question": question["question"],
                                                                       "kb_ids": [self.state["kb_id"]]})
            wanted = self.state["documents"][question["document_id"]]["document_id"]
            def relevant(hit):
                return hit["document_id"] == wanted and question["evidence"] in hit["body"]
            rank = next((i for i, hit in enumerate(result["hits"], 1) if relevant(hit)), None)
            if any(h["chunk_id"] not in all_chunks for h in result["candidates"]):
                raise RuntimeError("Retrieval exposed a chunk outside the published corpus")
            report["retrieval"].append({**question, "expected_document_id": wanted,
                "recall_candidates": any(relevant(h) for h in result["candidates"]), "recall_at_5": rank is not None,
                "ndcg_at_5": 1 / math.log2(rank + 1) if rank else 0,
                "seconds": time.perf_counter() - start, "hits": result["hits"]})
            save(path, report)
            print(f"API retrieval {len(report['retrieval'])}/{len(questions)}", flush=True)
        # One question per annotated fact; the other wording remains in retrieval testing.
        for question in questions[::2]:
            if any(r["case_id"] == question["id"] for r in report["answers"]):
                continue
            result = await self.chat(question["question"], self.state["kb_id"])
            result.update(case_id=question["id"], expected_answer=question["answer"],
                          expected_evidence=question["evidence"], document_id=question["document_id"])
            if not result["turn"]["citations"]:
                raise RuntimeError("Evidence-bearing question returned no citations")
            report["answers"].append(result)
            save(path, report)
            print(f"Grounded answer {len(report['answers'])}/25", flush=True)
        report.setdefault("unanswerable", [])
        for question in ("公司食堂每周三的午餐菜单是什么？", "星河工厂2029年的停机检修计划是什么？", "明天北京最高气温是多少？"):
            if any(r["turn"]["question"] == question for r in report["unanswerable"]):
                continue
            result = await self.chat(question, self.state["kb_id"])
            result["abstention_phrase_present"] = any(p in result["turn"]["answer"] for p in
                ("不足", "无法", "未提供", "未包含", "未提及", "没有", "不能确定", "不能回答"))
            report["unanswerable"].append(result)
            save(path, report)
        report.update(status="completed", finished_at=datetime.now(UTC).isoformat(),
            summary={"documents": len(report["documents"]), "chunks": len(all_chunks),
                     "questions": len(questions), "grounded_answers": len(report["answers"]),
                     "recall_at_5": sum(r["recall_at_5"] for r in report["retrieval"]) / len(questions),
                     "recall_candidates": sum(r["recall_candidates"] for r in report["retrieval"]) / len(questions),
                     "ndcg_at_5": sum(r["ndcg_at_5"] for r in report["retrieval"]) / len(questions),
                     "manual_answer_entailment_review_required": True})
        save(path, report)
        print(json.dumps(report["summary"]), flush=True)


async def main(phase, *, resume=False):
    run = Acceptance()
    try:
        await run.login()
        if phase == "upload":
            await run.upload()
        elif phase == "wait":
            await run.wait()
        elif phase == "status":
            print(json.dumps(await run.status(), ensure_ascii=False, indent=2))
        elif phase == "lifecycle":
            await run.lifecycle()
        elif phase == "capacity":
            await run.capacity()
        elif phase == "evaluate":
            await run.evaluate(resume=resume)
    finally:
        await run.client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["upload", "wait", "status", "lifecycle", "capacity", "evaluate"])
    parser.add_argument("--resume", action="store_true", help="Resume evaluation only when corpus and questions match")
    try:
        args = parser.parse_args()
        asyncio.run(main(args.phase, resume=args.resume))
    except Exception as exc:
        # No exception traceback/local variables: credentials must not appear in logs.
        print(json.dumps({"status": "failed", "type": type(exc).__name__,
                          "message": str(exc) if isinstance(exc, RuntimeError) else "Acceptance step failed."}))
        raise SystemExit(1) from None
