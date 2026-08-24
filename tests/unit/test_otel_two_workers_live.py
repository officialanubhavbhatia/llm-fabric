"""Two gateway processes export traces to one shared OTLP HTTP destination."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]
API_KEY = "otel-live-test-key1"

REGISTRY = """
models:
  - id: cheap
    provider: mock
    provider_model: cheap-v1
    context_window: 8192
    capabilities: [chat]
    enabled: true
"""


class _Collector(BaseHTTPRequestHandler):
    posts: list[bytes] = []

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        _Collector.posts.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")


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
        pytest.fail("live PostgreSQL and Redis required for OTLP replica proof")


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


def test_two_workers_export_traces_to_shared_otlp_sink(tmp_path: Path) -> None:
    _require_live()
    _ensure_migrated()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    credentials = json.dumps([{"key": API_KEY, "tenant_id": "otel-live", "user_id": "alice"}])
    collector_port = _free_port()
    _Collector.posts = []
    server = ThreadingHTTPServer(("127.0.0.1", collector_port), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    workers: list[subprocess.Popen[str]] = []
    ports: list[int] = []
    try:
        for name in ("worker-a", "worker-b"):
            port = _free_port()
            ports.append(port)
            env = {
                key: value for key, value in os.environ.items() if not key.startswith("LLM_FABRIC_")
            }
            env.update(
                {
                    "PYTHONPATH": str(SRC),
                    "HOME": str(tmp_path),
                    "HOSTNAME": name,
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
                    "LLM_FABRIC_REDIS_URL": _redis().rstrip("/") + "/13",
                    "LLM_FABRIC_REGISTRY_PATH": str(registry),
                    "LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED": "true",
                    "LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER": "true",
                    "LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT": (
                        f"http://127.0.0.1:{collector_port}"
                    ),
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
        while time.time() < deadline and len(ready) < 2:
            for proc in workers:
                if proc.poll() is not None:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    pytest.fail(f"worker exited {proc.returncode}: {stderr[-2000:]}")
            for port in ports:
                if port in ready:
                    continue
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                        ready.add(port)
                except httpx.HTTPError:
                    pass
            time.sleep(0.2)
        assert ready == set(ports)

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        body = {"model": "cheap", "messages": [{"role": "user", "content": "trace me"}]}
        for port in ports:
            response = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=10,
            )
            assert response.status_code == 200, response.text
    finally:
        for proc in workers:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        deadline = time.time() + 8
        while time.time() < deadline and len(_Collector.posts) < 2:
            time.sleep(0.2)
        server.shutdown()

    blob = b"".join(_Collector.posts)
    assert len(_Collector.posts) >= 2, f"OTLP posts={len(_Collector.posts)}"
    assert b"worker-a" in blob
    assert b"worker-b" in blob
    assert b"llm-fabric" in blob or b"llm_fabric" in blob
