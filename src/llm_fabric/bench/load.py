"""A reproducible load harness for the gateway.

The constitution requires k6 "or an equivalent reproducible harness", and
requires that no RPS number is ever published without the workload that produced
it. This module is that harness.

**Why it speaks HTTP directly.** A load generator written on top of a
general-purpose client library tends to become the bottleneck before the server
does, and then the number measured is the client's ceiling wearing the server's
name. This opens plain sockets, reuses them, and pre-encodes each request once,
so the generator does as little work per request as possible. It still has a
ceiling — see `calibrate()`, which measures it so a run can be checked against
it rather than trusted.

**Closed loop by default, open loop on request.** With `--concurrency` alone the
generator keeps a fixed number of requests in flight, which measures capacity: a
slower server simply receives less load. That is the right shape for "how much
can it do", and the wrong shape for "what is latency at 1000 RPS", because a
struggling server is never actually offered 1000 RPS. Passing `--rate` switches
to an open loop that offers a fixed arrival rate regardless of how the server is
coping, and reports the queueing that results.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_fabric.bench.resources import elapsed_cpu, sample_process

#: Read cap for a single response. Large enough for any gateway reply, small
#: enough that a runaway stream fails the run instead of exhausting memory.
_READ_LIMIT = 4 * 1024 * 1024

_CRLF = b"\r\n"
_HEADER_END = b"\r\n\r\n"


# --------------------------------------------------------------------------
# workloads


@dataclass(frozen=True, slots=True)
class Workload:
    """One named, reproducible request shape.

    The constitution forbids publishing an RPS number without saying what was
    being served, so a result always carries the workload that produced it.
    """

    name: str
    description: str
    method: str
    path: str
    body: dict[str, Any] | None = None
    streaming: bool = False
    requires_auth: bool = True

    def encode(self, host_header: str, token: str | None) -> bytes:
        """Pre-build the request bytes, so the hot loop only writes them."""
        payload = b"" if self.body is None else json.dumps(self.body).encode()
        lines = [
            f"{self.method} {self.path} HTTP/1.1".encode(),
            f"host: {host_header}".encode(),
            b"connection: keep-alive",
            b"accept-encoding: identity",
        ]
        if payload:
            lines.append(b"content-type: application/json")
            lines.append(f"content-length: {len(payload)}".encode())
        if token and self.requires_auth:
            lines.append(f"authorization: Bearer {token}".encode())
        return _CRLF.join(lines) + _HEADER_END + payload


_SHORT_PROMPT = "Summarise the following in one sentence: the meeting was long."
_LONG_PROMPT = " ".join(["Explain the trade-offs in distributed rate limiting."] * 40)


def _chat(content: str, *, stream: bool = False, model: str = "auto") -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if stream:
        body["stream"] = True
    return body


#: The workload set. The constitution names more (agent workloads, real
#: generation); those are absent because the subsystems they exercise are not
#: built, and a workload that does not exercise anything is worse than none.
WORKLOADS: dict[str, Workload] = {
    "liveness": Workload(
        name="liveness",
        description=(
            "GET /healthz. No auth, no routing. The ASGI and event-loop floor: "
            "no gateway workload can exceed this on the same hardware."
        ),
        method="GET",
        path="/healthz",
        requires_auth=False,
    ),
    "models": Workload(
        name="models",
        description="GET /v1/models. Auth, tenancy and registry serialisation, no routing.",
        method="GET",
        path="/v1/models",
    ),
    "route-preview": Workload(
        name="route-preview",
        description=(
            "POST /v1/routes/preview. The full planner — filtering, scoring, "
            "explanation — with no inference. Isolates routing cost."
        ),
        method="POST",
        path="/v1/routes/preview",
        body=_chat(_SHORT_PROMPT),
    ),
    "chat-short": Workload(
        name="chat-short",
        description=(
            "POST /v1/chat/completions against the mock provider. The whole "
            "serving path minus provider latency: auth, quota, routing, "
            "adapter, metering."
        ),
        method="POST",
        path="/v1/chat/completions",
        body=_chat(_SHORT_PROMPT),
    ),
    "chat-long": Workload(
        name="chat-long",
        description="As chat-short, with a longer prompt, to expose token-counting cost.",
        method="POST",
        path="/v1/chat/completions",
        body=_chat(_LONG_PROMPT),
    ),
    "chat-stream": Workload(
        name="chat-stream",
        description="Streaming SSE to completion, measuring whole-response time.",
        method="POST",
        path="/v1/chat/completions",
        body=_chat(_SHORT_PROMPT, stream=True),
        streaming=True,
    ),
    "chat-pinned": Workload(
        name="chat-pinned",
        description="A pinned model rather than an alias, so no ranking runs.",
        method="POST",
        path="/v1/chat/completions",
        body=_chat(_SHORT_PROMPT, model="mock-small"),
    ),
}


# --------------------------------------------------------------------------
# settings and results


@dataclass(frozen=True, slots=True)
class LoadSettings:
    workload: Workload
    host: str = "127.0.0.1"
    port: int = 8000
    duration_s: float = 20.0
    warmup_s: float = 3.0
    connections: int = 64
    processes: int = 4
    token: str | None = None
    #: Requests per second to offer. `None` runs a closed loop instead.
    rate: float | None = None

    @property
    def host_header(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(slots=True)
class _Samples:
    """Raw observations from one generator process."""

    latencies_ms: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    #: Open loop only: how late each request was actually sent.
    schedule_delay_ms: list[float] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "latencies_ms": self.latencies_ms,
            "statuses": self.statuses,
            "errors": self.errors,
            "schedule_delay_ms": self.schedule_delay_ms,
            "started": self.started,
            "finished": self.finished,
        }


@dataclass(frozen=True, slots=True)
class LoadResult:
    settings: LoadSettings
    requests: int
    duration_s: float
    achieved_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    statuses: dict[int, int]
    errors: dict[str, int]
    schedule_delay_p99_ms: float | None
    environment: dict[str, Any]
    resources: dict[str, Any] = field(default_factory=dict)
    tokens_per_s: float | None = None

    @property
    def successes(self) -> int:
        return sum(count for status, count in self.statuses.items() if 200 <= status < 400)

    @property
    def error_rate(self) -> float:
        total = self.requests + sum(self.errors.values())
        return 0.0 if total == 0 else 1.0 - (self.successes / total)

    @property
    def offered_load_was_met(self) -> bool | None:
        """Open loop only: whether the generator kept to its schedule.

        A large schedule delay means the *generator* fell behind, so the server
        was never offered the requested rate and the run proves nothing about
        that rate.
        """
        if self.settings.rate is None or self.schedule_delay_p99_ms is None:
            return None
        return self.schedule_delay_p99_ms < 50.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload": {
                "name": self.settings.workload.name,
                "description": self.settings.workload.description,
                "method": self.settings.workload.method,
                "path": self.settings.workload.path,
                "streaming": self.settings.workload.streaming,
                "request_body": self.settings.workload.body,
            },
            "load": {
                "mode": "open-loop" if self.settings.rate else "closed-loop",
                "target_rps": self.settings.rate,
                "connections": self.settings.connections,
                "generator_processes": self.settings.processes,
                "duration_s": round(self.duration_s, 3),
                "warmup_s": self.settings.warmup_s,
            },
            "results": {
                "requests": self.requests,
                "achieved_rps": round(self.achieved_rps, 1),
                "error_rate": round(self.error_rate, 6),
                "p50_ms": round(self.p50_ms, 3),
                "p95_ms": round(self.p95_ms, 3),
                "p99_ms": round(self.p99_ms, 3),
                "max_ms": round(self.max_ms, 3),
                "statuses": self.statuses,
                "errors": self.errors,
                "schedule_delay_p99_ms": (
                    round(self.schedule_delay_p99_ms, 3)
                    if self.schedule_delay_p99_ms is not None
                    else None
                ),
                "offered_load_was_met": self.offered_load_was_met,
                "tokens_per_s": (
                    round(self.tokens_per_s, 1) if self.tokens_per_s is not None else None
                ),
            },
            "resources": self.resources,
            "environment": self.environment,
        }


# --------------------------------------------------------------------------
# the generator


class _Connection:
    """One keep-alive HTTP/1.1 connection.

    Deliberately minimal. It understands exactly the response shapes the gateway
    produces — `content-length` bodies and `chunked` SSE — and raises on anything
    else rather than guessing, because a generator that silently mis-parses a
    response reports throughput for work that never completed.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    @classmethod
    async def open(cls, host: str, port: int) -> _Connection:
        reader, writer = await asyncio.open_connection(host, port, limit=_READ_LIMIT)
        sock = writer.get_extra_info("socket")
        if sock is not None:
            # Nagle would batch small requests and invent latency that the
            # server is not responsible for.
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(reader, writer)

    async def request(self, payload: bytes) -> int:
        self._writer.write(payload)
        await self._writer.drain()
        return await self._read_response()

    async def _read_response(self) -> int:
        head = await self._reader.readuntil(_HEADER_END)
        status_line, _, raw_headers = head.partition(_CRLF)
        status = int(status_line.split(b" ", 2)[1])

        headers: dict[bytes, bytes] = {}
        for line in raw_headers.split(_CRLF):
            if not line:
                continue
            key, _, value = line.partition(b":")
            headers[key.strip().lower()] = value.strip()

        if (length := headers.get(b"content-length")) is not None:
            await self._reader.readexactly(int(length))
            return status
        if headers.get(b"transfer-encoding", b"").lower() == b"chunked":
            await self._drain_chunked()
            return status
        raise RuntimeError(f"unsupported response framing in {headers!r}")

    async def _drain_chunked(self) -> None:
        while True:
            size_line = await self._reader.readuntil(_CRLF)
            size = int(size_line.strip().split(b";")[0], 16)
            if size == 0:
                await self._reader.readuntil(_CRLF)
                return
            await self._reader.readexactly(size + 2)

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(OSError, asyncio.CancelledError):
            await self._writer.wait_closed()


async def _drive(
    settings: LoadSettings,
    payload: bytes,
    samples: _Samples,
    deadline: float,
    record_after: float,
    per_process_rate: float | None,
    start_at: float,
) -> None:
    """Run one process's share of the load."""
    connections = max(1, settings.connections // settings.processes)
    conns = await asyncio.gather(
        *(_Connection.open(settings.host, settings.port) for _ in range(connections))
    )

    async def closed_loop(conn: _Connection) -> None:
        while time.monotonic() < deadline:
            await _one(conn, scheduled=None)

    async def open_loop(conn: _Connection, offset: int) -> None:
        assert per_process_rate is not None
        # Each connection fires every `interval` seconds. The stagger must
        # spread the connections *within* one interval — offsetting by whole
        # intervals would leave every connection firing at the same instants,
        # turning a smooth arrival rate into a burst of `connections` requests
        # every `interval`, which measures queueing the server never caused.
        interval = connections / per_process_rate
        tick = start_at + offset * (interval / connections)
        while tick < deadline:
            now = time.monotonic()
            if tick > now:
                await asyncio.sleep(tick - now)
            await _one(conn, scheduled=tick)
            tick += interval

    async def _one(conn: _Connection, *, scheduled: float | None) -> None:
        sent = time.monotonic()
        try:
            status = await conn.request(payload)
        except Exception as exc:  # a failed request is data, not a crash
            name = type(exc).__name__
            samples.errors[name] = samples.errors.get(name, 0) + 1
            return
        done = time.monotonic()
        if done < record_after:
            return
        samples.latencies_ms.append((done - sent) * 1000.0)
        samples.statuses[status] = samples.statuses.get(status, 0) + 1
        if scheduled is not None:
            samples.schedule_delay_ms.append(max(0.0, (sent - scheduled) * 1000.0))

    try:
        if per_process_rate is None:
            await asyncio.gather(*(closed_loop(conn) for conn in conns))
        else:
            await asyncio.gather(*(open_loop(conn, offset) for offset, conn in enumerate(conns)))
    finally:
        samples.finished = time.monotonic()
        await asyncio.gather(*(conn.close() for conn in conns))


def _process_entry(settings: LoadSettings, payload: bytes, queue: Any) -> None:
    """Body of one generator process."""
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

    samples = _Samples()

    async def main() -> None:
        start = time.monotonic()
        samples.started = start
        await _drive(
            settings,
            payload,
            samples,
            deadline=start + settings.warmup_s + settings.duration_s,
            record_after=start + settings.warmup_s,
            per_process_rate=(
                None if settings.rate is None else settings.rate / settings.processes
            ),
            start_at=start,
        )

    asyncio.run(main())
    queue.put(samples.as_dict())


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _environment() -> dict[str, Any]:
    """Everything a benchmark report must name to be reproducible."""
    import importlib.metadata as md

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    return {
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "software": {
            "python": sys.version.split()[0],
            "llm_fabric": version("llm-fabric"),
            "uvicorn": version("uvicorn"),
            "fastapi": version("fastapi"),
            "pydantic": version("pydantic"),
            "uvloop": version("uvloop"),
            "httptools": version("httptools"),
        },
        "note": (
            "A laptop under a desktop OS is not a server. Absolute numbers "
            "measured here bound nothing about production hardware; they are "
            "useful as a before/after comparison on the same machine."
        ),
    }


def run_load(settings: LoadSettings) -> LoadResult:
    """Run one measurement and return it. Blocks for warmup plus duration."""
    payload = settings.workload.encode(settings.host_header, settings.token)
    before = sample_process()

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    workers = [
        ctx.Process(target=_process_entry, args=(settings, payload, queue))
        for _ in range(settings.processes)
    ]
    for worker in workers:
        worker.start()

    collected = [queue.get() for _ in workers]
    for worker in workers:
        worker.join()

    latencies: list[float] = []
    delays: list[float] = []
    statuses: dict[int, int] = {}
    errors: dict[str, int] = {}
    spans: list[float] = []

    for part in collected:
        latencies.extend(part["latencies_ms"])
        delays.extend(part["schedule_delay_ms"])
        for status, count in part["statuses"].items():
            statuses[int(status)] = statuses.get(int(status), 0) + count
        for name, count in part["errors"].items():
            errors[name] = errors.get(name, 0) + count
        spans.append(part["finished"] - part["started"] - settings.warmup_s)

    latencies.sort()
    delays.sort()
    # Processes run concurrently, so the wall clock of the run is the longest
    # span, not their sum.
    duration = max(spans) if spans else settings.duration_s

    after = sample_process()
    resources = {
        **elapsed_cpu(before, after),
        "rss_bytes": after.rss_bytes,
        "gpu": after.gpu,
        "queue_depth": None,
        "note": (
            "CPU is the generator parent plus its waited-for worker children. "
            "RSS is the parent only. The server is a separate process and is "
            "not sampled here. GPU is absent unless nvidia-smi reports a "
            "device. Server queue depth is not polled during a run."
        ),
    }
    return LoadResult(
        settings=settings,
        requests=len(latencies),
        duration_s=duration,
        achieved_rps=len(latencies) / duration if duration > 0 else 0.0,
        p50_ms=_percentile(latencies, 0.50),
        p95_ms=_percentile(latencies, 0.95),
        p99_ms=_percentile(latencies, 0.99),
        max_ms=latencies[-1] if latencies else 0.0,
        statuses=statuses,
        errors=errors,
        schedule_delay_p99_ms=_percentile(delays, 0.99) if delays else None,
        environment=_environment(),
        resources=resources,
        tokens_per_s=None,
    )


def calibrate(settings: LoadSettings) -> dict[str, Any]:
    """Measure the generator's own ceiling, to check a run against it.

    Runs the cheapest workload the server has. If a measured result approaches
    this number, the result is the *generator's* limit and says nothing about
    the server.
    """
    probe = LoadSettings(
        workload=WORKLOADS["liveness"],
        host=settings.host,
        port=settings.port,
        duration_s=min(5.0, settings.duration_s),
        warmup_s=1.0,
        connections=settings.connections,
        processes=settings.processes,
    )
    result = run_load(probe)
    return {
        "generator_ceiling_rps": round(result.achieved_rps, 1),
        "measured_on": probe.workload.name,
        "note": (
            "The floor workload's throughput. A result near this figure is "
            "bounded by the generator, not by the server."
        ),
    }


def summarise(result: LoadResult) -> str:
    """A human-readable block, stating the workload before the numbers."""
    settings = result.settings
    mode = f"open loop at {settings.rate:g} rps" if settings.rate else "closed loop"
    lines = [
        f"Workload      {settings.workload.name} — {settings.workload.description}",
        f"Load          {mode}, {settings.connections} connections, "
        f"{settings.processes} generator processes",
        f"Duration      {result.duration_s:.1f}s measured, {settings.warmup_s:g}s warmup discarded",
        "",
        f"Achieved      {result.achieved_rps:,.0f} req/s over {result.requests:,} requests",
        f"Latency       p50 {result.p50_ms:.2f}ms   p95 {result.p95_ms:.2f}ms   "
        f"p99 {result.p99_ms:.2f}ms   max {result.max_ms:.2f}ms",
        f"Errors        {result.error_rate:.4%}  statuses={result.statuses}",
    ]
    if result.errors:
        lines.append(f"Failures      {result.errors}")
    if result.offered_load_was_met is False:
        lines.append(
            f"WARNING       the generator fell behind its schedule by "
            f"{result.schedule_delay_p99_ms:.0f}ms at p99, so the server was "
            f"never offered {settings.rate:g} rps. This run does not measure that rate."
        )
    lines.append("")
    lines.append(
        "These are measurements of this build on this machine under this "
        "workload. They are not a comparison against any other system."
    )
    return "\n".join(lines)
