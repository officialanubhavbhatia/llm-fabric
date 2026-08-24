"""Current-release PostgreSQL backup/restore against live Compose Postgres."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from llm_fabric.errors import ConfigurationError
from llm_fabric.observability.usage_event import TokenSource, UsageEvent
from llm_fabric.storage.postgres import create_database_engine, probe_database
from llm_fabric.storage.records import ConversationMessage, PromptDefinition
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.storage.schema import EXPECTED_HEAD, current_revision
from llm_fabric.storage.usage import UsageLedger
from llm_fabric.tenancy.scope import TenantScope

REPO = Path(__file__).resolve().parents[2]


def _admin() -> str:
    return os.environ.get(
        "LLM_FABRIC_TEST_DATABASE_URL",
        "postgresql://fabric:fabric@127.0.0.1:5432/fabric",
    )


def _require_live() -> None:
    try:
        probe_database(_admin(), timeout_s=2)
    except ConfigurationError:
        pytest.fail("live PostgreSQL required for pg_dump/pg_restore")


def _pg(container: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", container, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _container() -> str | None:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    for name in result.stdout.splitlines():
        if "postgres" in name:
            return name
    return None


def test_pg_dump_restore_preserves_schema_rls_and_rows(tmp_path: Path) -> None:
    _require_live()
    container = _container()
    if container is None:
        pytest.fail("docker postgres container required for pg_dump/pg_restore")

    source = f"fabric_p17_{uuid.uuid4().hex[:8]}"
    restored = f"fabric_p17r_{uuid.uuid4().hex[:8]}"
    dump = tmp_path / "fabric.dump"
    admin_engine = create_database_engine(_admin())
    try:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {source}"))
            connection.execute(text(f"DROP DATABASE IF EXISTS {restored}"))
            connection.execute(text(f"CREATE DATABASE {source}"))
        src_url = _admin().rsplit("/", 1)[0] + f"/{source}"
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_FABRIC_")}
        env["LLM_FABRIC_DATABASE_URL"] = src_url
        env["PYTHONPATH"] = str(REPO / "src")
        upgraded = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert upgraded.returncode == 0, upgraded.stderr + upgraded.stdout

        src = create_database_engine(src_url)
        suffix = uuid.uuid4().hex[:8]
        tenant_a = TenantScope(tenant_id=f"tenant-a-{suffix}", user_id="alice")
        tenant_b = TenantScope(tenant_id=f"tenant-b-{suffix}", user_id="bob")
        stores = TenantStores(engine=src)
        stores.conversations.create(
            tenant_a, title="keep-me", messages=(ConversationMessage(role="user", content="a"),)
        )
        stores.prompts.publish(
            tenant_a,
            PromptDefinition(
                tenant_id=tenant_a.tenant_id,
                prompt_id="sys",
                version=1,
                owner="alice",
                purpose="chat",
                template="hello",
            ),
        )
        ledger = UsageLedger(src)
        ledger.insert(
            UsageEvent(
                event_id=f"evt-{suffix}",
                invocation_id=f"inv-{suffix}",
                request_id=f"req-{suffix}",
                tenant_id=tenant_a.tenant_id,
                provider="mock",
                model="cheap",
                prompt_tokens=4,
                completion_tokens=1,
                token_source=TokenSource.PROVIDER_MEASURED.value,
                started_at=1.0,
                completed_at=2.0,
                status="success",
            )
        )
        with src.begin() as connection:
            connection.execute(
                text("INSERT INTO tenants (tenant_id, name, created_at) VALUES (:id, 'A', :ts)"),
                {"id": tenant_a.tenant_id, "ts": time.time()},
            )
            connection.execute(
                text(
                    "INSERT INTO users (tenant_id, user_id, roles, created_at) "
                    "VALUES (:tid, 'alice', 'owner', :ts)"
                ),
                {"tid": tenant_a.tenant_id, "ts": time.time()},
            )
        src.dispose()

        dump_cmd = _pg(
            container,
            "pg_dump",
            "-U",
            "fabric",
            "-Fc",
            "-f",
            f"/tmp/{source}.dump",
            source,
        )
        assert dump_cmd.returncode == 0, dump_cmd.stderr
        copied = subprocess.run(
            ["docker", "cp", f"{container}:/tmp/{source}.dump", str(dump)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert copied.returncode == 0, copied.stderr
        size = dump.stat().st_size
        assert size > 0

        started = time.perf_counter()
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text(f"CREATE DATABASE {restored}"))
        subprocess.run(
            ["docker", "cp", str(dump), f"{container}:/tmp/{restored}.dump"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        restore = _pg(
            container,
            "pg_restore",
            "-U",
            "fabric",
            "-d",
            restored,
            f"/tmp/{restored}.dump",
        )
        duration = time.perf_counter() - started
        assert restore.returncode == 0, restore.stderr + restore.stdout

        restored_url = _admin().rsplit("/", 1)[0] + f"/{restored}"
        restored_engine = create_database_engine(restored_url)
        try:
            assert current_revision(restored_engine) == EXPECTED_HEAD
        finally:
            restored_engine.dispose()
        app_url = restored_url.replace("://fabric:", "://fabric_app:", 1)
        app_engine = create_database_engine(app_url)
        try:
            restored_stores = TenantStores(engine=app_engine)
            assert restored_stores.conversations.list(tenant_b) == []
            kept = restored_stores.conversations.list(tenant_a)
            assert {row.title for row in kept} == {"keep-me"}
            restored_ledger = UsageLedger(app_engine)
            assert restored_ledger.totals(tenant_id=tenant_a.tenant_id).prompt_tokens == 4
            assert restored_ledger.totals(tenant_id=tenant_b.tenant_id).prompt_tokens == 0
        finally:
            app_engine.dispose()
        (tmp_path / "backup-report.txt").write_text(
            f"backup_size_bytes={size}\nrestore_duration_s={duration:.3f}\n",
            encoding="utf-8",
        )
        print(f"backup_size_bytes={size} restore_duration_s={duration:.3f}")
    finally:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            for name in (source, restored):
                connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = :name AND pid <> pg_backend_pid()"
                    ),
                    {"name": name},
                )
                connection.execute(text(f"DROP DATABASE IF EXISTS {name}"))
        admin_engine.dispose()
