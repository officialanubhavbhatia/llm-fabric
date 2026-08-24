"""Live PostgreSQL: empty DB fails production start; Alembic head serves."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError
from llm_fabric.runtime import initialize_runtime
from llm_fabric.storage.postgres import create_database_engine, probe_database
from llm_fabric.storage.redis import probe_redis
from llm_fabric.storage.schema import EXPECTED_HEAD, assert_schema_revision, current_revision

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
API_KEY = "production-test-key-16"


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


def _require_live() -> None:
    try:
        probe_database(_admin_url(), timeout_s=2)
        probe_redis(_redis_url(), timeout_s=2)
    except ConfigurationError:
        pytest.fail(f"live PostgreSQL and Redis required; tried {_admin_url()}")


def _db_url(name: str) -> str:
    base = _admin_url().rsplit("/", 1)[0]
    return f"{base}/{name}"


def _create_database(name: str) -> str:
    if not name.replace("_", "").isalnum() or not name.startswith("fabric_p11_"):
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


def _upgrade(url: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["LLM_FABRIC_DATABASE_URL"] = url
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _stamp(url: str, revision: str) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["LLM_FABRIC_DATABASE_URL"] = url
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "stamp", revision],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def _production(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        api_keys=[API_KEY],
        database_url=database_url,
        redis_url=_redis_url(),
    )


def test_empty_database_refuses_production_startup() -> None:
    _require_live()
    name = f"fabric_p11_{uuid.uuid4().hex[:10]}"
    url = _create_database(name)
    try:
        with pytest.raises(ConfigurationError, match="no Alembic"):
            initialize_runtime(_production(url))
        with pytest.raises(ConfigurationError, match="no Alembic"):
            assert_schema_revision(url)

        import tempfile

        tmp = Path(tempfile.mkdtemp())
        port = _free_port()
        result = subprocess.run(
            [sys.executable, "-m", "llm_fabric"],
            cwd=str(tmp),
            env=_clean_env(
                tmp,
                LLM_FABRIC_DATABASE_URL=url,
                LLM_FABRIC_REDIS_URL=_redis_url(),
                LLM_FABRIC_PORT=str(port),
            ),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert "no Alembic" in combined or "production startup validation failed" in combined
    finally:
        _drop_database(name)


def test_alembic_upgrade_then_production_starts() -> None:
    _require_live()
    name = f"fabric_p11_{uuid.uuid4().hex[:10]}"
    url = _create_database(name)
    try:
        upgraded = _upgrade(url)
        assert upgraded.returncode == 0, upgraded.stderr + upgraded.stdout
        engine = create_database_engine(url)
        try:
            assert current_revision(engine) == EXPECTED_HEAD
        finally:
            engine.dispose()
        initialize_runtime(_production(url))
    finally:
        _drop_database(name)


def test_schema_behind_head_refuses_production_startup() -> None:
    _require_live()
    name = f"fabric_p11_{uuid.uuid4().hex[:10]}"
    url = _create_database(name)
    try:
        assert _upgrade(url).returncode == 0
        _stamp(url, "0001_usage_events")
        with pytest.raises(ConfigurationError, match="is not head"):
            initialize_runtime(_production(url))

        import tempfile

        tmp = Path(tempfile.mkdtemp())
        port = _free_port()
        result = subprocess.run(
            [sys.executable, "-m", "llm_fabric"],
            cwd=str(tmp),
            env=_clean_env(
                tmp,
                LLM_FABRIC_DATABASE_URL=url,
                LLM_FABRIC_REDIS_URL=_redis_url(),
                LLM_FABRIC_PORT=str(port),
            ),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode != 0
        assert "is not head" in result.stderr + result.stdout
    finally:
        _drop_database(name)


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


def test_four_production_workers_start_without_ddl() -> None:
    _require_live()
    name = f"fabric_p11_{uuid.uuid4().hex[:10]}"
    url = _create_database(name)
    workers: list[subprocess.Popen[str]] = []
    try:
        upgraded = _upgrade(url)
        assert upgraded.returncode == 0, upgraded.stderr + upgraded.stdout
        before = create_database_engine(url)
        try:
            with before.begin() as connection:
                table_count = connection.execute(
                    text("SELECT count(*) FROM pg_tables WHERE schemaname = 'public'")
                ).scalar()
                # create_all is a no-op when tables already exist, so a table
                # count is not proof. An event trigger aborts any worker DDL.
                connection.execute(
                    text(
                        """
                        CREATE FUNCTION p11_reject_ddl() RETURNS event_trigger
                        LANGUAGE plpgsql AS $$
                        BEGIN
                          RAISE EXCEPTION 'worker ddl forbidden';
                        END;
                        $$
                        """
                    )
                )
                connection.execute(
                    text(
                        "CREATE EVENT TRIGGER p11_reject_ddl ON ddl_command_start "
                        "EXECUTE FUNCTION p11_reject_ddl()"
                    )
                )
            assert current_revision(before) == EXPECTED_HEAD
        finally:
            before.dispose()

        import tempfile
        import urllib.error
        import urllib.request

        tmp = Path(tempfile.mkdtemp())
        ports: list[int] = []
        for _ in range(4):
            port = _free_port()
            ports.append(port)
            proc = subprocess.Popen(
                [sys.executable, "-m", "llm_fabric"],
                cwd=str(tmp),
                env=_clean_env(
                    tmp,
                    LLM_FABRIC_DATABASE_URL=url,
                    LLM_FABRIC_REDIS_URL=_redis_url(),
                    LLM_FABRIC_PORT=str(port),
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            workers.append(proc)

        deadline = time.time() + 25
        ready: set[int] = set()
        while time.time() < deadline and len(ready) < 4:
            for proc in workers:
                if proc.poll() is not None:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    pytest.fail(f"worker exited {proc.returncode}: {stderr[-2000:]}")
            for port in ports:
                if port in ready:
                    continue
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/healthz", timeout=1
                    ) as response:
                        if response.status == 200:
                            ready.add(port)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                    pass
            time.sleep(0.2)
        assert ready == set(ports), f"workers not healthy: {set(ports) - ready}"

        after = create_database_engine(url)
        try:
            with after.connect() as connection:
                after_count = connection.execute(
                    text("SELECT count(*) FROM pg_tables WHERE schemaname = 'public'")
                ).scalar()
            assert current_revision(after) == EXPECTED_HEAD
            assert after_count == table_count
        finally:
            after.dispose()
    finally:
        for proc in workers:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        _drop_database(name)
