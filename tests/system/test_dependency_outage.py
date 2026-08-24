"""Live P0-FIX-4: dependency outage, admission, recovery. Not mocked."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis

pytestmark = [
    pytest.mark.system,
    pytest.mark.skipif(
        os.environ.get("LLM_FABRIC_SYSTEM_TEST") != "1",
        reason="live dependency outage tests require LLM_FABRIC_SYSTEM_TEST=1",
    ),
]

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "deployments" / "docker" / "docker-compose.yml"
API_KEY = "usage-live-test-key-16"

REGISTRY = """
models:
  - id: cheap
    provider: mock
    provider_model: cheap-v1
    context_window: 8192
    capabilities: [chat, streaming]
    enabled: true
"""


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


def _require_live() -> None:
    try:
        probe_database(_live_postgres_url(), timeout_s=2)
        probe_redis(_live_redis_url(), timeout_s=2)
    except ConfigurationError:
        pytest.fail(
            "live PostgreSQL and Redis are required for P0-FIX-4 outage tests; "
            f"tried {_live_postgres_url()} and {_live_redis_url()}"
        )
    _ensure_migrated(_live_postgres_url())


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
    env["LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED"] = "true"
    env["LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER"] = "true"
    env["LLM_FABRIC_DATABASE_URL"] = _live_postgres_url()
    env["LLM_FABRIC_REDIS_URL"] = _live_redis_url()
    env["LLM_FABRIC_HEALTH_PROBE_INTERVAL_S"] = "1"
    env["LLM_FABRIC_HEALTH_PROBE_TIMEOUT_S"] = "0.5"
    env["LLM_FABRIC_HEALTH_FAIL_THRESHOLD"] = "2"
    env["LLM_FABRIC_HEALTH_RECOVERY_THRESHOLD"] = "2"
    env.update(overrides)
    return env


def _start_worker(tmp_path: Path, port: int, registry: Path, **env: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "llm_fabric"],
        cwd=tmp_path,
        env=_clean_env(
            tmp_path,
            LLM_FABRIC_PORT=str(port),
            LLM_FABRIC_REGISTRY_PATH=str(registry),
            **env,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_http(
    url: str,
    proc: subprocess.Popen[str],
    *,
    status: int = 200,
    timeout_s: float = 25.0,
) -> httpx.Response:
    deadline = time.time() + timeout_s
    last = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=2)
            raise AssertionError(
                f"gateway exited {proc.returncode}: {stderr[-2000:]}\n{stdout[-500:]}"
            )
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == status:
                return response
            last = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.2)
    raise AssertionError(f"{url} never reached HTTP {status}: {last}")


def _stop(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _chat(port: int, *, client: httpx.Client | None = None) -> httpx.Response:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {"model": "cheap", "messages": [{"role": "user", "content": "hi"}]}
    if client is None:
        return httpx.post(url, headers=_headers(), json=payload, timeout=10.0)
    return client.post(url, headers=_headers(), json=payload, timeout=10.0)


def _metric_count(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        total += float(line.rsplit(" ", 1)[-1])
    return total


def _invocations(port: int) -> float:
    metrics = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5.0).text
    return _metric_count(metrics, "fabric_route_decisions_total")


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(COMPOSE.parent),
    )


def _restore_service(name: str) -> None:
    _compose("start", name)
    deadline = time.time() + 40
    while time.time() < deadline:
        if name == "postgres":
            try:
                probe_database(_live_postgres_url(), timeout_s=2)
                return
            except ConfigurationError:
                time.sleep(0.5)
        elif name == "redis":
            try:
                probe_redis(_live_redis_url(), timeout_s=2)
                return
            except ConfigurationError:
                time.sleep(0.5)
        else:
            time.sleep(0.5)
            if time.time() > deadline - 1:
                return
    pytest.fail(f"could not restore compose service {name}")


def _pid_listening(port: int) -> int | None:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    line = result.stdout.strip().splitlines()
    if not line:
        return None
    try:
        return int(line[0])
    except ValueError:
        return None


@pytest.fixture
def live_registry(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(REGISTRY, encoding="utf-8")
    return path


def test_compose_postgres_outage_rejects_without_inference(
    tmp_path: Path, live_registry: Path
) -> None:
    _require_live()
    port = _free_port()
    worker = _start_worker(tmp_path, port, live_registry)
    try:
        _wait_http(f"http://127.0.0.1:{port}/readyz", worker, status=200)
        healthy = _chat(port)
        assert healthy.status_code == 200, healthy.text
        before = _invocations(port)
        pid = _pid_listening(port)
        assert pid is not None

        stopped = _compose("stop", "postgres")
        assert stopped.returncode == 0, stopped.stderr
        try:
            started = time.monotonic()
            deadline = started + 12
            ready = None
            while time.monotonic() < deadline:
                ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
                if ready.status_code == 503:
                    break
                time.sleep(0.2)
            detection_s = time.monotonic() - started
            assert ready is not None
            assert ready.status_code == 503
            live = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
            assert live.status_code == 200
            assert worker.poll() is None
            assert _pid_listening(port) == pid

            chat = _chat(port)
            assert chat.status_code == 503, chat.text
            assert chat.json()["error"]["type"] == "dependency_unavailable"
            after = _invocations(port)
            assert after == before

            for _ in range(20):
                denied = _chat(port)
                assert denied.status_code == 503
            assert _invocations(port) == before
        finally:
            _restore_service("postgres")

        recovered_at = None
        recover_started = time.monotonic()
        while time.monotonic() - recover_started < 20:
            ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
            if ready.status_code == 200:
                recovered_at = time.monotonic() - recover_started
                break
            time.sleep(0.2)
        assert recovered_at is not None, "readiness did not recover without restart"
        assert _pid_listening(port) == pid
        ok = _chat(port)
        assert ok.status_code == 200, ok.text
        print(
            f"postgres_outage detection_s={detection_s:.3f} recovery_s={recovered_at:.3f} pid={pid}"
        )
    finally:
        _stop(worker)


def test_compose_redis_outage_rejects_without_inference(
    tmp_path: Path, live_registry: Path
) -> None:
    _require_live()
    port = _free_port()
    worker = _start_worker(tmp_path, port, live_registry)
    try:
        _wait_http(f"http://127.0.0.1:{port}/readyz", worker, status=200)
        assert _chat(port).status_code == 200
        before = _invocations(port)
        pid = _pid_listening(port)
        stopped = _compose("stop", "redis")
        assert stopped.returncode == 0, stopped.stderr
        try:
            started = time.monotonic()
            ready = None
            while time.monotonic() - started < 12:
                ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
                if ready.status_code == 503:
                    break
                time.sleep(0.2)
            detection_s = time.monotonic() - started
            assert ready is not None and ready.status_code == 503
            assert httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0).status_code == 200
            assert worker.poll() is None
            assert _pid_listening(port) == pid
            chat = _chat(port)
            assert chat.status_code == 503, chat.text
            assert _invocations(port) == before
        finally:
            _restore_service("redis")

        recover_started = time.monotonic()
        recovered = False
        while time.monotonic() - recover_started < 20:
            if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0).status_code == 200:
                recovered = True
                break
            time.sleep(0.2)
        assert recovered
        assert _pid_listening(port) == pid
        assert _chat(port).status_code == 200
        print(f"redis_outage detection_s={detection_s:.3f} pid={pid}")
    finally:
        _stop(worker)


def test_otel_outage_keeps_ready_and_chat(tmp_path: Path, live_registry: Path) -> None:
    _require_live()
    port = _free_port()
    worker = _start_worker(
        tmp_path,
        port,
        live_registry,
        LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4318",
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/readyz", worker, status=200)
        assert _chat(port).status_code == 200
        _compose("stop", "otel-collector")
        try:
            time.sleep(2)
            assert httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0).status_code == 200
            ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
            assert ready.status_code == 200, ready.text
            assert _chat(port).status_code == 200
            assert worker.poll() is None
        finally:
            _compose("start", "otel-collector")
        assert httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0).status_code == 200
    finally:
        _stop(worker)


def test_direct_and_keepalive_cannot_bypass_admission(tmp_path: Path, live_registry: Path) -> None:
    _require_live()
    port = _free_port()
    worker = _start_worker(tmp_path, port, live_registry)
    try:
        _wait_http(f"http://127.0.0.1:{port}/readyz", worker, status=200)
        with httpx.Client(http2=False, timeout=10.0) as client:
            first = _chat(port, client=client)
            assert first.status_code == 200
            before = _invocations(port)
            _compose("stop", "postgres")
            try:
                deadline = time.time() + 12
                while time.time() < deadline:
                    if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0).status_code == 503:
                        break
                    time.sleep(0.2)
                else:
                    pytest.fail("readyz did not become 503")
                # Direct process address, same keep-alive client.
                reused = _chat(port, client=client)
                direct = _chat(port)
                assert reused.status_code == 503, reused.text
                assert direct.status_code == 503, direct.text
                assert _invocations(port) == before
            finally:
                _restore_service("postgres")
    finally:
        _stop(worker)


def test_four_workers_converge_to_reject(tmp_path: Path, live_registry: Path) -> None:
    _require_live()
    ports: list[int] = []
    workers: list[subprocess.Popen[str]] = []
    try:
        for _ in range(4):
            port = _free_port()
            worker = _start_worker(tmp_path, port, live_registry)
            _wait_http(f"http://127.0.0.1:{port}/readyz", worker, status=200)
            ports.append(port)
            workers.append(worker)
        for port in ports:
            assert _chat(port).status_code == 200
        _compose("stop", "postgres")
        try:
            started = time.monotonic()
            deadline = started + 12
            rejected: dict[int, float] = {}
            while time.monotonic() < deadline and len(rejected) < 4:
                for port in ports:
                    if port in rejected:
                        continue
                    ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
                    chat = _chat(port)
                    if ready.status_code == 503 and chat.status_code == 503:
                        rejected[port] = time.monotonic() - started
                time.sleep(0.2)
            assert len(rejected) == 4, f"workers still admitting: {set(ports) - set(rejected)}"
            print(f"multi_worker convergence_s={rejected}")
            for worker in workers:
                assert worker.poll() is None
        finally:
            _restore_service("postgres")
        for port, worker in zip(ports, workers, strict=True):
            deadline = time.time() + 20
            while time.time() < deadline:
                if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0).status_code == 200:
                    break
                time.sleep(0.2)
            else:
                pytest.fail(f"worker on {port} did not recover")
            assert _chat(port).status_code == 200
            assert worker.poll() is None
    finally:
        for worker in workers:
            _stop(worker)
