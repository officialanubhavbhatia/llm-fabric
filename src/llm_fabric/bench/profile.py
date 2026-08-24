"""Where the CPU goes on one request.

The load harness says how fast the gateway is. This says *why*, by driving the
ASGI application in-process under `cProfile` with no socket, no HTTP parsing and
no client in the way. Those are real costs at load, but they are not costs this
repository can change, and leaving them in the profile buries the ones it can.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import pstats
import statistics
import time
from typing import Any

from llm_fabric.bench.load import WORKLOADS, Workload
from llm_fabric.config import Settings
from llm_fabric.gateway.app import create_app
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.adapters.mock import MockProvider

_REGISTRY: dict[str, Any] = {
    "models": [
        {
            "id": "mock-small",
            "provider": "mock",
            "provider_model": "mock-small-v1",
            "context_window": 8192,
            "input_cost_per_mtok": 0.1,
            "output_cost_per_mtok": 0.2,
            "capabilities": ["chat"],
            "placement": {"locality": "local"},
            "fallbacks": ["mock-large"],
        },
        {
            "id": "mock-large",
            "provider": "mock",
            "provider_model": "mock-large-v1",
            "context_window": 32768,
            "input_cost_per_mtok": 3.0,
            "output_cost_per_mtok": 9.0,
            "capabilities": ["chat", "reasoning"],
            "placement": {"locality": "local"},
        },
    ],
    "aliases": [{"id": "auto", "policy": "cost_first", "candidates": ["mock-small", "mock-large"]}],
}


def _scope(workload: Workload, body: bytes) -> dict[str, Any]:
    headers = [(b"host", b"localhost"), (b"accept", b"*/*")]
    if body:
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body)).encode()))
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": workload.method,
        "scheme": "http",
        "path": workload.path,
        "raw_path": workload.path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 51000),
        "server": ("127.0.0.1", 8000),
    }


async def _drive(app: Any, workload: Workload, iterations: int) -> None:
    body = b"" if workload.body is None else json.dumps(workload.body).encode()

    for _ in range(iterations):
        sent = False

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            del message

        await app(_scope(workload, body), receive, send)


def profile(workload: Workload, iterations: int, top: int) -> str:
    app = create_app(
        settings=Settings(api_keys=[], log_level="ERROR"),
        registry=ModelRegistry.from_mapping(_REGISTRY),
        provider_overrides={"mock": MockProvider()},
    )

    async def run() -> None:
        # Run the real lifespan; skipping it would profile a different app from
        # the one that serves traffic.
        async with app.router.lifespan_context(app):
            await _drive(app, workload, iterations=50)  # warm caches and imports
            profiler.enable()
            await _drive(app, workload, iterations)
            profiler.disable()

    profiler = cProfile.Profile()
    asyncio.run(run())

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(top)
    return stream.getvalue()


def time_only(workload: Workload, iterations: int, repeats: int) -> list[float]:
    """Wall-clock microseconds per request, with no profiler attached.

    `cProfile` charges per function call, so it exaggerates any path that is
    call-heavy rather than work-heavy — which is exactly how a thread-dispatch
    path looks. Comparing two implementations under the profiler can therefore
    show a large win where there is none. This measures the same loop without
    it, and repeats so the spread is visible rather than a single sample.
    """
    app = create_app(
        settings=Settings(api_keys=[], log_level="ERROR"),
        registry=ModelRegistry.from_mapping(_REGISTRY),
        provider_overrides={"mock": MockProvider()},
    )
    timings: list[float] = []

    async def run() -> None:
        async with app.router.lifespan_context(app):
            await _drive(app, workload, iterations=200)
            for _ in range(repeats):
                start = time.perf_counter()
                await _drive(app, workload, iterations)
                elapsed = time.perf_counter() - start
                timings.append(elapsed / iterations * 1e6)

    asyncio.run(run())
    return timings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-profile",
        description="Profile one request path in-process, without sockets or HTTP parsing.",
    )
    parser.add_argument("--workload", default="chat-short", choices=sorted(WORKLOADS))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--time-only",
        action="store_true",
        help="Report microseconds per request without cProfile, which distorts call-heavy paths.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    if args.time_only:
        timings = time_only(WORKLOADS[args.workload], args.iterations, args.repeats)
        best = min(timings)
        print(
            f"{args.workload}: {best:.1f} us/request best of {args.repeats} "
            f"(median {statistics.median(timings):.1f}, worst {max(timings):.1f})"
        )
        return 0

    print(profile(WORKLOADS[args.workload], args.iterations, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
