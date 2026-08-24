"""Production quotas are finite and shared across workers via Redis."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]
API_KEY = "quota-live-test-key1"

REGISTRY = """
models:
  - id: cheap
    provider: mock
    provider_model: cheap-v1
    context_window: 8192
    capabilities: [chat]
    enabled: true
"""


def _postgres() -> str:
    return os.environ.get(
        "LLM_FABRIC_TEST_DATABASE_URL",
        "postgresql://fabric:fabric@127.0.0.1:5432/fabric",
    )


def _redis() -> str:
    return os.environ.get("LLM_FABRIC_TEST_REDIS_URL", "redis://127.0.0.1:6379/0")


def _require_live() -> None:
    try:
        probe_database(_postgres(), timeout_s=2)
        probe_redis(_redis(), timeout_s=2)
    except ConfigurationError:
        pytest.fail("live PostgreSQL and Redis required for global quota proof")


def _ensure_migrated() -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
    env["LLM_FABRIC_DATABASE_URL"] = _postgres()
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_four_workers_share_finite_production_quota(tmp_path: Path) -> None:
    _require_live()
    _ensure_migrated()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    redis_url = _redis().rstrip("/") + "/14"
    tenant_id = f"quota-{uuid.uuid4().hex[:8]}"
    credentials = json.dumps([{"key": API_KEY, "tenant_id": tenant_id, "user_id": "alice"}])
    ports: list[int] = []
    workers: list[subprocess.Popen[str]] = []
    try:
        for _ in range(4):
            port = _free_port()
            ports.append(port)
            env = {
                key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")
            }
            env.update(
                {
                    "PYTHONPATH": str(SRC),
                    "HOME": str(tmp_path),
                    "LLM_FABRIC_ENVIRONMENT": "production",
                    "LLM_FABRIC_ALLOW_ANONYMOUS": "false",
                    "LLM_FABRIC_API_CREDENTIALS": credentials,
                    "LLM_FABRIC_HOST": "127.0.0.1",
                    "LLM_FABRIC_PORT": str(port),
                    "LLM_FABRIC_DATABASE_URL": (
                        _postgres().replace("://fabric:", "://fabric_app:", 1)
                        if "://fabric:" in _postgres()
                        else _postgres()
                    ),
                    "LLM_FABRIC_REDIS_URL": redis_url,
                    "LLM_FABRIC_REGISTRY_PATH": str(registry),
                    "LLM_FABRIC_QUOTA_TENANT_REQUESTS_PER_MINUTE": "5",
                    "LLM_FABRIC_QUOTA_TENANT_MAX_CONCURRENCY": "8",
                }
            )
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-m", "llm_fabric"],
                    cwd=str(tmp_path),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
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
                    response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1)
                    if response.status_code == 200:
                        ready.add(port)
                except httpx.HTTPError:
                    pass
            time.sleep(0.2)
        assert ready == set(ports)

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        body = {"model": "cheap", "messages": [{"role": "user", "content": "quota"}]}
        codes: list[int] = []
        for index in range(12):
            port = ports[index % 4]
            response = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=10,
            )
            codes.append(response.status_code)
            if response.status_code == 429:
                assert response.json()["error"]["type"] == "quota_exceeded"
            elif response.status_code == 200:
                pass
            else:
                pytest.fail(f"unexpected {response.status_code}: {response.text}")
        assert codes.count(200) == 5
        assert codes.count(429) == 7
        assert 500 not in codes
        assert 503 not in codes

        from llm_fabric.storage.postgres import create_database_engine
        from llm_fabric.storage.usage import UsageLedger

        engine = create_database_engine(_postgres())
        try:
            totals = UsageLedger(engine).totals(tenant_id=tenant_id, observe=False)
            assert totals.requests == 5
            assert totals.invocations == 5
        finally:
            engine.dispose()
    finally:
        for proc in workers:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
