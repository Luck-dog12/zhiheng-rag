import json

import pytest

from scripts.capacity_live import CapacityRunner, Journal, document_text


def test_capacity_journal_recovers_only_interrupted_tail_and_rejects_parameter_changes(tmp_path):
    path = tmp_path / "progress.jsonl"
    journal = Journal(path, "demo", 4, 400)
    journal.append({"event": "knowledge_base", "kb_id": "example"})
    with path.open("ab") as file:
        file.write(b'{"event":"uploaded"')
    restored = Journal(path, "demo", 4, 400)
    assert restored.records == journal.records
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    with pytest.raises(ValueError, match="恢复参数"):
        Journal(path, "demo", 5, 400)


def test_capacity_data_is_deterministic_and_unique():
    assert document_text("sample", 9, 2400) == document_text("sample", 9, 2400)
    assert len({document_text("sample", i, 400) for i in range(10000)}) == 10000
    assert "CAP-sample-000009" in document_text("sample", 9, 2400)
    assert len(document_text("sample", 9, 2400)) >= 2400


async def test_capacity_runner_resume_and_chat_metrics(harness, tmp_path):
    h = harness
    journal = Journal(tmp_path / "progress.jsonl", "local-test", 4, 400)
    runner = CapacityRunner(h.client, journal, concurrency=1, poll_seconds=.001)
    await runner.prepare()
    await runner.upload_all()
    assert len(runner.uploads) == 4
    await h.drain(max_jobs=20)
    await runner.wait_jobs(10)
    report = runner.report(await runner.measure_chats(4))
    assert report["ready_documents"] == 4 and report["successful_chats"] == 4
    assert report["pending_jobs"] == 0 and not report["upload_errors"]
    count = len(h.services.storage.objects)
    restored = CapacityRunner(h.client, Journal(journal.path, "local-test", 4, 400), concurrency=1)
    await restored.prepare()
    await restored.upload_all()
    assert len(h.services.storage.objects) == count
    assert restored.report([])["ready_documents"] == 4


async def test_capacity_pending_jobs_do_not_become_successes(harness, tmp_path):
    runner = CapacityRunner(harness.client, Journal(tmp_path / "progress.jsonl", "pending", 1, 400), poll_seconds=.001)
    await runner.prepare()
    await runner.upload_all()
    await runner.wait_jobs(0)
    report = runner.report(await runner.measure_chats())
    assert report["ready_documents"] == 0 and report["pending_jobs"] == 1
    assert report["concurrent_chats"] == 0 and report["chat_success_rate"] is None
