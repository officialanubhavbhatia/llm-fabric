"""Entry point: `python -m llm_fabric` or the `llm-fabric` console script.

Serving configuration lives here rather than in a shell command so that the way
the gateway runs in production is version-controlled and reviewable, and so a
benchmark can name the configuration it measured.
"""

from __future__ import annotations

import sys

import uvicorn

from llm_fabric.config import Settings, get_settings
from llm_fabric.errors import ConfigurationError
from llm_fabric.observability.logging import request_logger
from llm_fabric.runtime import initialize_runtime


def _loop_and_protocol() -> tuple[str, str]:
    """Prefer the fast event loop and HTTP parser when they are installed.

    Both are optional dependencies. Asking uvicorn for `uvloop` when it is
    absent is a startup failure, so availability is checked rather than assumed;
    the gateway runs either way, more slowly on the pure-Python path.
    """
    try:
        import uvloop  # noqa: F401

        loop = "uvloop"
    except ImportError:
        loop = "asyncio"
    try:
        import httptools  # noqa: F401

        http = "httptools"
    except ImportError:
        http = "h11"
    return loop, http


def uvicorn_config(settings: Settings) -> dict[str, object]:
    """The serving configuration, as a dict so a test can assert on it.

    Refuses a multi-worker configuration that has not been explicitly
    acknowledged. Raising here rather than warning is deliberate: the failure it
    prevents is silent. Nothing errors, no test fails, and quotas simply stop
    binding at the number the operator configured.
    """
    if (
        settings.workers
        and settings.workers > 1
        and not settings.allow_unsafe_multiworker
        and (not settings.redis_url or not settings.database_url)
    ):
        raise ConfigurationError(
            f"refusing to start {settings.workers} workers: the quota ledger, "
            "circuit breakers and caches need Redis, and the usage ledger needs "
            f"PostgreSQL, so every quota would be enforced {settings.workers} "
            "times over and usage would stay process-local. Configure "
            "LLM_FABRIC_REDIS_URL and LLM_FABRIC_DATABASE_URL, or set "
            "LLM_FABRIC_ALLOW_UNSAFE_MULTIWORKER=true to proceed anyway."
        )

    loop, http = _loop_and_protocol()
    config: dict[str, object] = {
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level.lower(),
        "loop": loop,
        "http": http,
        "backlog": settings.backlog,
        # The gateway emits its own structured JSON logs. Uvicorn's per-request
        # access log duplicates them in a different format and costs measurable
        # throughput at load, so it is off.
        "access_log": False,
        # Long-lived SSE responses must not be severed by a keep-alive timer.
        "timeout_keep_alive": 15,
        # Drain in-flight work, including SSE, then exit before kube SIGKILL.
        "timeout_graceful_shutdown": settings.graceful_shutdown_timeout_s,
    }
    if settings.workers and settings.workers > 1:
        config["workers"] = settings.workers
    if settings.max_requests_per_worker:
        config["limit_max_requests"] = settings.max_requests_per_worker
    return config


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "doctor":
        from llm_fabric.doctor import run_doctor

        try:
            raise SystemExit(run_doctor())
        except ConfigurationError as exc:
            sys.stderr.write(exc.message + "\n")
            raise SystemExit(1) from exc

    if len(sys.argv) >= 2 and sys.argv[1] == "route":
        from llm_fabric.router.cli import main as route_main

        raise SystemExit(route_main(sys.argv[2:]))

    if len(sys.argv) >= 2 and sys.argv[1] == "model":
        from llm_fabric.models.cli import main as model_main

        raise SystemExit(model_main(sys.argv[2:]))

    if len(sys.argv) >= 2 and sys.argv[1] == "eval":
        rest = sys.argv[2:]
        if rest and rest[0] == "models":
            from llm_fabric.models.cli import main as model_main

            raise SystemExit(model_main(["eval", *rest[1:]]))
        if rest and rest[0] == "routing":
            from llm_fabric.eval.routing_quality import write_routing_artifacts

            paths = write_routing_artifacts()
            for label, path in paths.items():
                sys.stderr.write(f"wrote {label}: {path}\n")
            raise SystemExit(0)
        from llm_fabric.eval.cli import main as eval_main

        raise SystemExit(eval_main(rest))

    try:
        settings = get_settings()
        # Parent process: configuration + dependency probes. create_app repeats
        # the same initialize_runtime so `uvicorn --factory` cannot skip them.
        initialize_runtime(settings)
        config = uvicorn_config(settings)
    except ConfigurationError as exc:
        sys.stderr.write(exc.message + "\n")
        raise SystemExit(1) from exc

    if settings.workers and settings.workers > 1 and not settings.redis_url:
        request_logger().warning(
            "serving with multiple workers: per-process state is now per worker",
            extra={
                "workers": settings.workers,
                "consequence": (
                    "quota limits, circuit-breaker state, usage records and every "
                    "cache are per worker, so limits multiply by the worker count"
                ),
                "hint": "configure LLM_FABRIC_REDIS_URL for shared limits",
            },
        )

    try:
        uvicorn.run("llm_fabric.gateway.app:create_app", factory=True, **config)  # type: ignore[arg-type]
    except ConfigurationError as exc:
        sys.stderr.write(exc.message + "\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
