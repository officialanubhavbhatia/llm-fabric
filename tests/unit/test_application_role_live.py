"""Live PostgreSQL: fabric_app is DML-only; migrations use the owner role."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError
from llm_fabric.observability.usage_event import TokenSource, UsageEvent
from llm_fabric.runtime import initialize_runtime
from llm_fabric.storage.postgres import (
    APPLICATION_ROLE,
    create_database_engine,
    current_role_schema_privileges,
    probe_database,
)
from llm_fabric.storage.records import ConversationMessage
from llm_fabric.storage.redis import probe_redis
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.storage.schema import EXPECTED_HEAD, current_revision
from llm_fabric.storage.usage import UsageLedger
from llm_fabric.tenancy.scope import TenantScope

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
API_KEY = "production-test-key-16"

REGISTRY = """
models:
  - id: cheap
    provider: mock
    provider_model: cheap-v1
    context_window: 8192
    capabilities: [chat, streaming]
    enabled: true
"""


def _admin_url() -> str:
    return os.environ.get(
        "LLM_FABRIC_TEST_DATABASE_URL",
        "postgresql://fabric:fabric@127.0.0.1:5432/fabric",
    )


def _redis_url() -> str:
    return os.environ.get(
        "LLM_FABRIC_TEST_REDIS_URL",
        "redis://127.0.0.1:6379/0",
    )


def _app_url(admin_url: str) -> str:
    if "://fabric:" not in admin_url:
        raise AssertionError("live tests expect admin URL user 'fabric'")
    return admin_url.replace("://fabric:", f"://{APPLICATION_ROLE}:", 1)


def _require_live() -> None:
    try:
        probe_database(_admin_url(), timeout_s=2)
        probe_redis(_redis_url(), timeout_s=2)
    except ConfigurationError:
        pytest.fail(f"live PostgreSQL and Redis required; tried {_admin_url()}")


def _db_url(name: str) -> str:
    return f"{_admin_url().rsplit('/', 1)[0]}/{name}"


def _create_database(name: str) -> str:
    if not name.replace("_", "").isalnum() or not name.startswith("fabric_p12_"):
        raise AssertionError(f"refusing to create database '{name}'")
    engine = create_database_engine(_admin_url())
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {name}"))
            connection.execute(text(f"CREATE DATABASE {name}"))
    finally:
        engine.dispose()
    return _db_url(name)


def _drop_database(name: str) -> None:
    engine = create_database_engine(_admin_url())
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": name},
            )
            connection.execute(text(f"DROP DATABASE IF EXISTS {name}"))
    finally:
        engine.dispose()


def _alembic(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["LLM_FABRIC_DATABASE_URL"] = url
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _production(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        api_keys=[API_KEY],
        database_url=database_url,
        redis_url=_redis_url(),
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _clean_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["PYTHONPATH"] = str(SRC)
    env["HOME"] = str(tmp_path)
    env["LLM_FABRIC_ENVIRONMENT"] = "production"
    env["LLM_FABRIC_ALLOW_ANONYMOUS"] = "false"
    env["LLM_FABRIC_API_KEYS"] = API_KEY
    env["LLM_FABRIC_HOST"] = "127.0.0.1"
    env["LLM_FABRIC_REGISTRY_PATH"] = str(REPO / "config" / "models.yaml")
    env.update(overrides)
    return env


def _must_deny_sql(engine: Engine, statement: str, match: str | None = None) -> None:
    pattern = match if match is not None else r"."
    with pytest.raises(ProgrammingError, match=pattern), engine.begin() as connection:
        connection.execute(text(statement))


def test_alembic_as_fabric_app_cannot_create_schema() -> None:
    _require_live()
    name = f"fabric_p12_{uuid.uuid4().hex[:10]}"
    admin_url = _create_database(name)
    app_url = _app_url(admin_url)
    try:
        admin = create_database_engine(admin_url)
        try:
            with admin.begin() as connection:
                connection.execute(text(f"GRANT CONNECT ON DATABASE {name} TO fabric_app"))
                connection.execute(text("GRANT USAGE ON SCHEMA public TO fabric_app"))
                connection.execute(text("REVOKE CREATE ON SCHEMA public FROM fabric_app"))
        finally:
            admin.dispose()
        failed = _alembic(app_url, "upgrade", "head")
        assert failed.returncode != 0, failed.stdout + failed.stderr
        combined = (failed.stdout + failed.stderr).lower()
        assert "permission denied" in combined or "must be owner" in combined
        upgraded = _alembic(admin_url, "upgrade", "head")
        assert upgraded.returncode == 0, upgraded.stderr + upgraded.stdout
        engine = create_database_engine(admin_url)
        try:
            assert current_revision(engine) == EXPECTED_HEAD
        finally:
            engine.dispose()
    finally:
        _drop_database(name)


def test_fabric_app_cannot_ddl_and_can_serve() -> None:
    _require_live()
    name = f"fabric_p12_{uuid.uuid4().hex[:10]}"
    admin_url = _create_database(name)
    app_url = _app_url(admin_url)
    worker: subprocess.Popen[str] | None = None
    try:
        upgraded = _alembic(admin_url, "upgrade", "head")
        assert upgraded.returncode == 0, upgraded.stderr + upgraded.stdout

        app = create_database_engine(app_url)
        try:
            privileges = current_role_schema_privileges(app)
            assert privileges["role"] == APPLICATION_ROLE
            assert privileges["create_on_public"] is False
            assert privileges["create_on_database"] is False
            assert privileges["createdb"] is False
            assert privileges["createrole"] is False
            assert privileges["superuser"] is False
            assert privileges["usage_on_public"] is True

            _must_deny_sql(
                app, "CREATE TABLE p12_should_fail (id integer)", match="permission denied"
            )
            _must_deny_sql(app, "ALTER TABLE usage_events ADD COLUMN p12_should_fail integer")
            _must_deny_sql(app, "DROP TABLE usage_events")
            _must_deny_sql(app, "CREATE SCHEMA p12_should_fail")
            _must_deny_sql(app, "CREATE ROLE p12_should_fail_role")
            _must_deny_sql(app, "DROP ROLE fabric_app")
        finally:
            app.dispose()

        initialize_runtime(_production(app_url))

        suffix = uuid.uuid4().hex[:8]
        tenant_a = TenantScope(tenant_id=f"tenant-a-{suffix}", user_id="alice")
        tenant_b = TenantScope(tenant_id=f"tenant-b-{suffix}", user_id="bob")
        stores = TenantStores(engine=create_database_engine(app_url))
        conv = stores.conversations.create(
            tenant_a,
            title="owned-by-a",
            messages=(ConversationMessage(role="user", content="secret-a"),),
        )
        assert stores.conversations.get(tenant_b, conv.conversation_id) is None
        assert stores.conversations.require(tenant_a, conv.conversation_id).title == "owned-by-a"

        ledger = UsageLedger(create_database_engine(app_url))
        event = UsageEvent(
            event_id=f"evt-{suffix}",
            invocation_id=f"inv-{suffix}",
            request_id=f"req-{suffix}",
            tenant_id=tenant_a.tenant_id,
            provider="mock",
            model="cheap",
            prompt_tokens=7,
            completion_tokens=2,
            token_source=TokenSource.PROVIDER_MEASURED.value,
            started_at=1.0,
            completed_at=2.0,
            status="success",
        )
        assert ledger.insert(event).inserted is True
        assert ledger.get(event.event_id, tenant_id=tenant_a.tenant_id) is not None
        assert ledger.get(event.event_id, tenant_id=tenant_b.tenant_id) is None
        assert ledger.totals(tenant_id=tenant_a.tenant_id).prompt_tokens == 7
        assert ledger.totals(tenant_id=tenant_b.tenant_id).prompt_tokens == 0

        import tempfile

        tmp = Path(tempfile.mkdtemp())
        registry = tmp / "models.yaml"
        registry.write_text(REGISTRY, encoding="utf-8")
        port = _free_port()
        worker = subprocess.Popen(
            [sys.executable, "-m", "llm_fabric"],
            cwd=str(tmp),
            env=_clean_env(
                tmp,
                LLM_FABRIC_DATABASE_URL=app_url,
                LLM_FABRIC_REDIS_URL=_redis_url(),
                LLM_FABRIC_PORT=str(port),
                LLM_FABRIC_REGISTRY_PATH=str(registry),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + 20
        healthy = False
        while time.time() < deadline:
            if worker.poll() is not None:
                stderr = worker.stderr.read() if worker.stderr else ""
                pytest.fail(f"gateway exited {worker.returncode}: {stderr[-2000:]}")
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
                if response.status_code == 200:
                    healthy = True
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        assert healthy, "production gateway as fabric_app did not become healthy"

        chat = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "cheap", "messages": [{"role": "user", "content": "p12-chat"}]},
            timeout=15.0,
        )
        assert chat.status_code == 200, chat.text
        default_ledger = UsageLedger(create_database_engine(app_url))
        totals = default_ledger.totals(tenant_id="default")
        assert totals.requests >= 1
        ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
        assert ready.status_code == 200, ready.text
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
        _drop_database(name)
