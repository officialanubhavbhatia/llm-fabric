"""Real process startup paths share initialize_runtime.

Dead-dependency cases do not need a live cluster: they point at a closed port.
Healthy cases use LLM_FABRIC_TEST_* URLs when set, otherwise the Compose
defaults on localhost.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]
DEAD_POSTGRES = "postgresql://fabric:supersecret@127.0.0.1:1/fabric"
DEAD_REDIS = "redis://127.0.0.1:1/0"
API_KEY = "production-test-key-16"


def _live_postgres_url() -> str:
    return os.environ.get(
        "LLM_FABRIC_TEST_DATABASE_URL",
        "postgresql://fabric:fabric@127.0.0.1:5432/fabric",
    )


def _live_redis_url() -> str:
    return os.environ.get(
        "LLM_FABRIC_TEST_REDIS_URL",
        "redis://127.0.0.1:6379/0",
    )


def _ensure_migrated(url: str) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["LLM_FABRIC_DATABASE_URL"] = url
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def _live_dependencies_available() -> bool:
    try:
        probe_database(_live_postgres_url(), timeout_s=2)
        probe_redis(_live_redis_url(), timeout_s=2)
    except ConfigurationError:
        return False
    return True


def _clean_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["PYTHONPATH"] = str(SRC)
    env["HOME"] = str(tmp_path)
    env["LLM_FABRIC_ENVIRONMENT"] = "production"
    env["LLM_FABRIC_ALLOW_ANONYMOUS"] = "false"
    env["LLM_FABRIC_API_KEYS"] = API_KEY
    env["LLM_FABRIC_HOST"] = "127.0.0.1"
    env["LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED"] = "true"
    env["LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER"] = "true"
    env["LLM_FABRIC_REGISTRY_PATH"] = str(REPO / "config" / "models.yaml")
    env.update(overrides)
    return env


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _nothing_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return True
        return False


def _run(tmp_path: Path, args: list[str], **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=tmp_path,
        env=_clean_env(tmp_path, **overrides),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_factory_refuses_when_postgres_is_unreachable(tmp_path: Path) -> None:
    port = _free_port()
    result = _run(
        tmp_path,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "llm_fabric.gateway.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        LLM_FABRIC_DATABASE_URL=DEAD_POSTGRES,
        LLM_FABRIC_REDIS_URL=DEAD_REDIS,
        LLM_FABRIC_PORT=str(port),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert (
        "PostgreSQL is unreachable" in combined
        or "production startup validation failed" in combined
    )
    assert "supersecret" not in combined
    assert _nothing_listening(port)


def test_factory_refuses_when_redis_is_unreachable(tmp_path: Path) -> None:
    if not _live_dependencies_available():
        pytest.fail(
            "live PostgreSQL is required to isolate a Redis startup failure; "
            f"tried {_live_postgres_url()}"
        )
    port = _free_port()
    result = _run(
        tmp_path,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "llm_fabric.gateway.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        LLM_FABRIC_DATABASE_URL=_live_postgres_url(),
        LLM_FABRIC_REDIS_URL=DEAD_REDIS,
        LLM_FABRIC_PORT=str(port),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "Redis is unreachable" in combined or "production startup validation failed" in combined
    assert _nothing_listening(port)


def test_cli_refuses_when_postgres_is_unreachable(tmp_path: Path) -> None:
    port = _free_port()
    result = _run(
        tmp_path,
        [sys.executable, "-m", "llm_fabric"],
        LLM_FABRIC_DATABASE_URL=DEAD_POSTGRES,
        LLM_FABRIC_REDIS_URL=DEAD_REDIS,
        LLM_FABRIC_PORT=str(port),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert (
        "PostgreSQL is unreachable" in combined
        or "production startup validation failed" in combined
    )
    assert "supersecret" not in combined
    assert _nothing_listening(port)


def test_cli_refuses_when_redis_is_unreachable(tmp_path: Path) -> None:
    if not _live_dependencies_available():
        pytest.fail(
            "live PostgreSQL is required to isolate a Redis startup failure; "
            f"tried {_live_postgres_url()}"
        )
    port = _free_port()
    result = _run(
        tmp_path,
        [sys.executable, "-m", "llm_fabric"],
        LLM_FABRIC_DATABASE_URL=_live_postgres_url(),
        LLM_FABRIC_REDIS_URL=DEAD_REDIS,
        LLM_FABRIC_PORT=str(port),
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "Redis is unreachable" in combined or "production startup validation failed" in combined
    assert _nothing_listening(port)


def test_cli_refuses_when_production_auth_is_missing(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        [sys.executable, "-m", "llm_fabric"],
        LLM_FABRIC_API_KEYS="",
        LLM_FABRIC_DATABASE_URL=DEAD_POSTGRES,
        LLM_FABRIC_REDIS_URL=DEAD_REDIS,
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "ALLOW_ANONYMOUS" in combined or "without authentication" in combined


def test_multiworker_factory_does_not_serve_when_postgres_is_unreachable(
    tmp_path: Path,
) -> None:
    """Uvicorn's supervisor may bind before workers load.

    Workers still run `create_app` → `initialize_runtime`. A dead Postgres
    must mean no successful production response, even if a socket exists.
    """
    if not _live_dependencies_available():
        pytest.fail(
            "live Redis is required to start a production multi-worker process; "
            f"tried {_live_redis_url()}"
        )
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "llm_fabric.gateway.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "2",
        ],
        cwd=tmp_path,
        env=_clean_env(
            tmp_path,
            LLM_FABRIC_DATABASE_URL=DEAD_POSTGRES,
            LLM_FABRIC_REDIS_URL=_live_redis_url(),
            LLM_FABRIC_PORT=str(port),
            LLM_FABRIC_WORKERS="2",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(3)
        import urllib.error
        import urllib.request

        try:
            url = f"http://127.0.0.1:{port}/healthz"
            with urllib.request.urlopen(url, timeout=1) as response:
                pytest.fail(
                    f"multi-worker process served HTTP {response.status} with Postgres down"
                )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        stderr = ""
        if proc.poll() is not None and proc.stderr is not None:
            stderr = proc.stderr.read()
        if stderr:
            assert "supersecret" not in stderr
            assert (
                "PostgreSQL is unreachable" in stderr
                or "production startup validation failed" in stderr
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_factory_serves_when_dependencies_are_healthy(tmp_path: Path) -> None:
    if not _live_dependencies_available():
        pytest.fail(
            "live PostgreSQL and Redis are required for factory healthy-path verification; "
            f"tried {_live_postgres_url()} and {_live_redis_url()}"
        )
    _ensure_migrated(_live_postgres_url())
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "llm_fabric.gateway.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=tmp_path,
        env=_clean_env(
            tmp_path,
            LLM_FABRIC_DATABASE_URL=_live_postgres_url(),
            LLM_FABRIC_REDIS_URL=_live_redis_url(),
            LLM_FABRIC_PORT=str(port),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 15
        import urllib.error
        import urllib.request

        last_error = "not contacted"
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                pytest.fail(f"factory exited {proc.returncode} before serving: {stderr}")
            try:
                url = f"http://127.0.0.1:{port}/healthz"
                with urllib.request.urlopen(url, timeout=1) as response:
                    assert response.status == 200
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = str(exc)
                time.sleep(0.2)
        pytest.fail(f"factory did not become healthy: {last_error}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_cli_serves_when_dependencies_are_healthy(tmp_path: Path) -> None:
    if not _live_dependencies_available():
        pytest.fail(
            "live PostgreSQL and Redis are required for CLI healthy-path verification; "
            f"tried {_live_postgres_url()} and {_live_redis_url()}"
        )
    _ensure_migrated(_live_postgres_url())
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_fabric"],
        cwd=tmp_path,
        env=_clean_env(
            tmp_path,
            LLM_FABRIC_DATABASE_URL=_live_postgres_url(),
            LLM_FABRIC_REDIS_URL=_live_redis_url(),
            LLM_FABRIC_PORT=str(port),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 15
        import urllib.error
        import urllib.request

        last_error = "not contacted"
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                pytest.fail(f"CLI exited {proc.returncode} before serving: {stderr}")
            try:
                url = f"http://127.0.0.1:{port}/healthz"
                with urllib.request.urlopen(url, timeout=1) as response:
                    assert response.status == 200
                    return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = str(exc)
                time.sleep(0.2)
        pytest.fail(f"CLI did not become healthy: {last_error}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
