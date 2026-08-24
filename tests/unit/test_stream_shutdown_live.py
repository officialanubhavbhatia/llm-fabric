"""SIGTERM drains an in-flight SSE response instead of cutting it immediately."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]
API_KEY = "stream-live-test-key"

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
        pytest.fail("live PostgreSQL and Redis required for stream drain proof")


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


def test_sigterm_drains_in_flight_mock_stream(tmp_path: Path) -> None:
    _require_live()
    _ensure_migrated()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    port = _free_port()
    credentials = json.dumps([{"key": API_KEY, "tenant_id": "stream-live", "user_id": "alice"}])
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")}
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
            "LLM_FABRIC_REDIS_URL": _redis().rstrip("/") + "/12",
            "LLM_FABRIC_REGISTRY_PATH": str(registry),
            "LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED": "true",
            "LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER": "true",
            "LLM_FABRIC_MOCK_DELAY_S": "0.4",
            "LLM_FABRIC_GRACEFUL_SHUTDOWN_TIMEOUT_S": "20",
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_fabric"],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    collected: dict[str, object] = {}

    def _read_stream() -> None:
        with (
            httpx.Client(timeout=30) as client,
            client.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "cheap",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": "alpha beta gamma delta epsilon zeta",
                        }
                    ],
                },
            ) as response,
        ):
            collected["status"] = response.status_code
            chunks: list[str] = []
            for line in response.iter_lines():
                chunks.append(line)
                if len(chunks) == 2 and proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
            collected["body"] = "\n".join(chunks)

    try:
        deadline = time.time() + 25
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                pytest.fail(f"worker exited {proc.returncode}: {stderr[-2000:]}")
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("gateway never became healthy")

        reader = threading.Thread(target=_read_stream, daemon=True)
        reader.start()
        reader.join(timeout=25)
        assert not reader.is_alive()
        assert collected.get("status") == 200
        body = str(collected.get("body") or "")
        assert "[DONE]" in body
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
