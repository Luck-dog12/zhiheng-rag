import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import MetaData, create_engine, inspect, select

ROOT = Path(__file__).resolve().parents[1]


def test_migration_upgrade_schema_check_and_downgrade(tmp_path):
    path = tmp_path / "migration.db"
    env = {**os.environ, "APP_ENV": "test", "DATABASE_URL": f"sqlite+aiosqlite:///{path}"}
    for args in [("upgrade", "head"), ("check",), ("downgrade", "base")]:
        result = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    engine = create_engine(f"sqlite:///{path}")
    assert inspect(engine).get_table_names() == ["alembic_version"]
    engine.dispose()


def test_upgrade_preserves_existing_failed_turn_with_empty_recovery_state(tmp_path):
    path = tmp_path / "existing.db"
    env = {**os.environ, "APP_ENV": "test", "DATABASE_URL": f"sqlite+aiosqlite:///{path}"}
    def migrate(revision):
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", revision], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    migrate("0001")
    engine = create_engine(f"sqlite:///{path}")
    metadata = MetaData()
    metadata.reflect(bind=engine)
    now = datetime.now()
    with engine.begin() as conn:
        conn.execute(metadata.tables["users"].insert().values(id="user", username="test", display_name="测试",
            password_hash="test", role="admin", active=True, created_at=now))
        conn.execute(metadata.tables["conversations"].insert().values(id="conversation", user_id="user",
            title="如何调用模型", created_at=now))
        conn.execute(metadata.tables["conversation_turns"].insert().values(id="turn", conversation_id="conversation",
            question="指导我调用模型", resolved_question="", answer="", status="failed", citations=[], scope_kb_ids=[],
            error_code="generation_incomplete", created_at=now))
    migrate("head")
    metadata = MetaData()
    metadata.reflect(bind=engine)
    with engine.connect() as conn:
        turn = conn.execute(select(metadata.tables["conversation_turns"])).mappings().one()
        assert turn["question"] == "指导我调用模型" and turn["error_code"] == "generation_incomplete"
        assert turn["agent_state"] == {}
    engine.dispose()
