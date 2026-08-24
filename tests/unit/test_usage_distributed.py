"""Distributed usage accounting: 4 workers, real Postgres, real Redis, TCP."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.observability.usage_event import TokenSource, UsageEvent
from llm_fabric.storage.postgres import create_database_engine, init_schema, probe_database
from llm_fabric.storage.redis import probe_redis
from llm_fabric.storage.usage import UsageLedger

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[2]
API_KEY = "usage-live-test-key-16"

REGISTRY = """
models:
  - id: cheap
    provider: mock
    provider_model: cheap-v1
    context_window: 8192
    input_cost_per_mtok: 0.1
    output_cost_per_mtok: 0.2
    capabilities: [chat, streaming]
    enabled: true
    fallbacks: [premium]
  - id: premium
    provider: mock
    provider_model: premium-v1
    context_window: 8192
    input_cost_per_mtok: 1.0
    output_cost_per_mtok: 2.0
    capabilities: [chat, streaming]
    enabled: true
  - id: broken
    provider: failing
    provider_model: broken-v1
    capabilities: [chat]
    enabled: true
    fallbacks: [cheap]
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


def _live_dependencies_available() -> bool:
    try:
        probe_database(_live_postgres_url(), timeout_s=2)
        probe_redis(_live_redis_url(), timeout_s=2)
    except ConfigurationError:
        return False
    return True


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
    if not _live_dependencies_available():
        pytest.fail(
            "live PostgreSQL and Redis are required for distributed usage verification; "
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
    env["LLM_FABRIC_DATABASE_URL"] = _live_postgres_url()
    env["LLM_FABRIC_REDIS_URL"] = _live_redis_url()
    env.update(overrides)
    return env


def _start_worker(tmp_path: Path, port: int, registry: Path) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_fabric"],
        cwd=tmp_path,
        env=_clean_env(
            tmp_path,
            LLM_FABRIC_PORT=str(port),
            LLM_FABRIC_REGISTRY_PATH=str(registry),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def _wait_healthy(port: int, proc: subprocess.Popen[str], timeout_s: float = 25.0) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/healthz"
    last_error = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=2)
            raise AssertionError(
                f"gateway on {port} exited {proc.returncode}: {stderr[-2000:]}\n{stdout[-500:]}"
            )
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.2)
    stdout, stderr = "", ""
    if proc.poll() is not None:
        stdout, stderr = proc.communicate(timeout=2)
    raise AssertionError(f"gateway on {port} never became healthy: {last_error}\n{stderr[-2000:]}")


def _start_healthy_workers(
    tmp_path: Path, registry: Path, count: int = 4
) -> tuple[list[int], list[subprocess.Popen[str]]]:
    ports: list[int] = []
    workers: list[subprocess.Popen[str]] = []
    try:
        for _ in range(count):
            port = _free_port()
            worker = _start_worker(tmp_path, port, registry)
            _wait_healthy(port, worker)
            ports.append(port)
            workers.append(worker)
    except Exception:
        for worker in workers:
            _stop(worker)
        raise
    return ports, workers


def _stop(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _chat(port: int, model: str, content: str, *, stream: bool = False) -> httpx.Response:
    return httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers=_headers(),
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        },
        timeout=15.0,
    )


def _ledger() -> UsageLedger:
    engine = create_database_engine(_live_postgres_url())
    init_schema(engine)
    return UsageLedger(engine)


def test_four_workers_share_one_usage_ledger(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    tenant = "default"
    before = _ledger().totals(tenant_id=tenant)

    ports, workers = _start_healthy_workers(tmp_path, registry, 4)
    try:
        prompt = "count-me-please-for-usage"
        request_count = 40
        for index in range(request_count):
            port = ports[index % 4]
            response = _chat(port, "cheap", prompt)
            assert response.status_code == 200, response.text
        sample = _chat(ports[0], "cheap", prompt)
        body = sample.json()
        prompt_tokens = body["usage"]["prompt_tokens"]
        completion_tokens = body["usage"]["completion_tokens"]

        after = _ledger().totals(tenant_id=tenant)
        assert after.requests - before.requests == request_count + 1
        assert after.invocations - before.invocations == request_count + 1
        assert after.prompt_tokens - before.prompt_tokens == prompt_tokens * (request_count + 1)
        assert after.completion_tokens - before.completion_tokens == completion_tokens * (
            request_count + 1
        )
    finally:
        for worker in workers:
            _stop(worker)


def test_worker_restart_does_not_reset_usage(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    tenant = "default"
    ports, workers = _start_healthy_workers(tmp_path, registry, 4)
    try:
        for index in range(8):
            assert _chat(ports[index % 4], "cheap", "before-restart").status_code == 200
        mid = _ledger().totals(tenant_id=tenant)
        _stop(workers[1])
        workers[1] = _start_worker(tmp_path, ports[1], registry)
        _wait_healthy(ports[1], workers[1])
        for index in range(8):
            assert _chat(ports[index % 4], "cheap", "after-restart").status_code == 200
        after = _ledger().totals(tenant_id=tenant)
        assert after.invocations >= mid.invocations + 8
        assert after.prompt_tokens >= mid.prompt_tokens
    finally:
        for worker in workers:
            _stop(worker)


def test_duplicate_event_live_insert(tmp_path: Path) -> None:
    _require_live()
    del tmp_path
    ledger = _ledger()
    event_id = f"dup-{int(time.time() * 1000)}"
    event = UsageEvent(
        event_id=event_id,
        invocation_id=event_id,
        request_id=event_id,
        tenant_id="default",
        provider="mock",
        model="cheap",
        prompt_tokens=3,
        completion_tokens=1,
        token_source=TokenSource.PROVIDER_MEASURED.value,
        started_at=time.time(),
        completed_at=time.time(),
        status="success",
    )
    first = ledger.insert(event)
    before = ledger.totals(tenant_id="default")
    second = ledger.insert(event)
    after = ledger.totals(tenant_id="default")
    assert first.inserted is True
    assert second.duplicate is True
    assert after.invocations == before.invocations
    assert after.prompt_tokens == before.prompt_tokens


def test_fallback_records_both_invocations(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    port = _free_port()
    worker = _start_worker(tmp_path, port, registry)
    try:
        _wait_healthy(port, worker)
        response = _chat(port, "broken", "fallback-please")
        assert response.status_code == 200, response.text
        assert response.headers["x-fabric-served-model"] == "cheap"
        assert response.headers["x-fabric-failovers"] == "1"
        request_id = response.headers["x-fabric-request-id"]
        events = _ledger().list_events(tenant_id="default", request_id=request_id)
        assert len(events) == 2
        models = {event.model for event in events}
        assert models == {"broken", "cheap"}
        prompt_sum = sum(event.prompt_tokens for event in events)
        completion_sum = sum(event.completion_tokens for event in events)
        visible = response.json()["usage"]
        assert prompt_sum >= visible["prompt_tokens"]
        assert completion_sum >= visible["completion_tokens"]
    finally:
        _stop(worker)


def test_failed_requests_do_not_invent_usage(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    port = _free_port()
    worker = _start_worker(tmp_path, port, registry)
    try:
        _wait_healthy(port, worker)
        before = _ledger().totals(tenant_id="default")
        unauth = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "nope"}]},
            timeout=10.0,
        )
        assert unauth.status_code in {401, 403}
        rail = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers=_headers(),
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 99_999,
            },
            timeout=10.0,
        )
        assert rail.status_code == 400
        after = _ledger().totals(tenant_id="default")
        assert after.invocations == before.invocations
        assert _chat(port, "cheap", "after-failures").status_code == 200
        assert _ledger().totals(tenant_id="default").invocations == before.invocations + 1
    finally:
        _stop(worker)


def test_concurrent_tcp_usage_writes(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    ports, workers = _start_healthy_workers(tmp_path, registry, 4)
    try:
        before = _ledger().totals(tenant_id="default")

        def hit(index: int) -> int:
            port = ports[index % 4]
            return _chat(port, "cheap", f"concurrent-{index}").status_code

        with ThreadPoolExecutor(max_workers=20) as pool:
            codes = list(pool.map(hit, range(60)))
        assert codes.count(200) == 60
        after = _ledger().totals(tenant_id="default")
        assert after.invocations - before.invocations == 60
    finally:
        for worker in workers:
            _stop(worker)


def test_stream_success_is_metered(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    port = _free_port()
    worker = _start_worker(tmp_path, port, registry)
    try:
        _wait_healthy(port, worker)
        response = _chat(port, "cheap", "stream-me", stream=True)
        assert response.status_code == 200
        request_id = response.headers["x-fabric-request-id"]
        # Allow the generator to persist after the body is consumed.
        _ = response.text
        deadline = time.time() + 5
        events = []
        while time.time() < deadline:
            events = _ledger().list_events(tenant_id="default", request_id=request_id)
            if events:
                break
            time.sleep(0.1)
        assert len(events) == 1
        assert events[0].streaming is True
        assert events[0].token_source == TokenSource.LOCAL_TOKENIZER_ESTIMATE.value
    finally:
        _stop(worker)


def test_postgres_outage_does_not_unbounded_queue_and_request_still_returns(
    tmp_path: Path,
) -> None:
    """Unit-level: DurableMeter backpressure. Live outage is P0-FIX-4."""
    from llm_fabric.observability.metering import DurableMeter
    from llm_fabric.storage.postgres import create_database_engine, init_schema

    engine = create_database_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    init_schema(engine)
    meter = DurableMeter(engine, retry_buffer=3)
    meter._ledger.insert = lambda event: (_ for _ in ()).throw(RuntimeError("postgres down"))  # type: ignore[method-assign]
    results = meter.record_events(
        [
            UsageEvent(
                event_id=f"o{i}",
                invocation_id=f"o{i}",
                request_id=f"r{i}",
                tenant_id="acme",
                provider="mock",
                model="cheap",
                prompt_tokens=1,
                completion_tokens=1,
                token_source="UNAVAILABLE",
                started_at=1.0,
                completed_at=1.0,
                status="success",
            )
            for i in range(8)
        ]
    )
    assert meter.dropped_events == 5
    assert len(results) == 8


def test_usage_persist_overhead_is_measured(tmp_path: Path) -> None:
    _require_live()
    registry = tmp_path / "models.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    port = _free_port()
    worker = _start_worker(tmp_path, port, registry)
    try:
        _wait_healthy(port, worker)
        # Warmup
        for _ in range(5):
            assert _chat(port, "cheap", "warmup").status_code == 200
        started = time.perf_counter()
        count = 20
        for _ in range(count):
            assert _chat(port, "cheap", "measure-overhead").status_code == 200
        elapsed_ms = (time.perf_counter() - started) * 1000
        per_request = elapsed_ms / count
        artifact = tmp_path / "usage-overhead.json"
        artifact.write_text(
            json.dumps(
                {
                    "requests": count,
                    "total_ms": elapsed_ms,
                    "per_request_ms": per_request,
                    "note": "full HTTP path including DurableMeter sync insert",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert per_request > 0
        # Bound: a local mock+Postgres insert must finish. This is a measurement,
        # not a performance SLO.
        assert elapsed_ms < 60_000
    finally:
        _stop(worker)
