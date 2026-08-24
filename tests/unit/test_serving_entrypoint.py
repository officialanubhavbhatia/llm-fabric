"""How the gateway is configured to run.

Serving configuration is in Python rather than a shell command so it can be
asserted on. The multi-worker guard is the reason these tests exist: the failure
it prevents is silent, so nothing else in the suite would catch its removal.
"""

from __future__ import annotations

import pytest

from llm_fabric.__main__ import uvicorn_config
from llm_fabric.config import Settings, validate_startup
from llm_fabric.errors import ConfigurationError


def test_a_single_worker_is_the_default() -> None:
    config = uvicorn_config(Settings())
    # Absent rather than 1: uvicorn's single-process path is not the same code
    # as its supervisor, and the default should be the simpler one.
    assert "workers" not in config


def test_multiple_workers_are_refused_without_an_explicit_acknowledgement() -> None:
    with pytest.raises(ConfigurationError) as raised:
        uvicorn_config(Settings(workers=4))

    message = str(raised.value)
    assert "quota" in message
    assert "4 times over" in message
    assert "LLM_FABRIC_ALLOW_UNSAFE_MULTIWORKER" in message or "REDIS" in message


def test_multiple_workers_start_once_acknowledged() -> None:
    config = uvicorn_config(Settings(workers=4, allow_unsafe_multiworker=True))
    assert config["workers"] == 4


def test_multiple_workers_are_refused_without_postgres_even_with_redis() -> None:
    with pytest.raises(ConfigurationError, match="PostgreSQL|usage ledger"):
        uvicorn_config(Settings(workers=4, redis_url="redis://127.0.0.1:6379/0"))
    config = uvicorn_config(
        Settings(
            workers=4,
            redis_url="redis://127.0.0.1:6379/0",
            database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
        )
    )
    assert config["workers"] == 4


def test_one_worker_needs_no_acknowledgement() -> None:
    # Setting workers=1 explicitly is not the unsafe case.
    assert "workers" not in uvicorn_config(Settings(workers=1))


def test_the_uvicorn_access_log_is_off() -> None:
    # The gateway emits its own structured JSON logs. Uvicorn's access log
    # duplicates every line in a second format and costs throughput.
    assert uvicorn_config(Settings())["access_log"] is False


def test_the_fast_event_loop_and_parser_are_requested_when_available() -> None:
    config = uvicorn_config(Settings())
    assert config["loop"] in {"uvloop", "asyncio"}
    assert config["http"] in {"httptools", "h11"}


def test_keep_alive_outlives_a_streaming_response() -> None:
    # A keep-alive timeout shorter than a generation would sever SSE mid-stream.
    assert uvicorn_config(Settings())["timeout_keep_alive"] >= 5


def test_graceful_shutdown_is_finite_and_below_helm_pod_grace() -> None:
    config = uvicorn_config(Settings())
    timeout = config["timeout_graceful_shutdown"]
    assert isinstance(timeout, int)
    assert 0 < timeout < 30


def test_the_listen_backlog_is_bounded_and_configurable() -> None:
    assert uvicorn_config(Settings())["backlog"] == 2048
    assert uvicorn_config(Settings(backlog=128))["backlog"] == 128


def test_worker_recycling_is_off_unless_asked_for() -> None:
    assert "limit_max_requests" not in uvicorn_config(Settings())
    assert uvicorn_config(Settings(max_requests_per_worker=10_000))["limit_max_requests"] == 10_000


def test_host_and_port_come_from_settings() -> None:
    config = uvicorn_config(Settings(host="0.0.0.0", port=9999))  # noqa: S104
    assert config["host"] == "0.0.0.0"  # noqa: S104
    assert config["port"] == 9999


def test_production_cannot_acknowledge_unsafe_multiworker() -> None:
    """The development escape hatch is not a production configuration."""
    with pytest.raises(ConfigurationError, match="UNSAFE_MULTIWORKER|multiple workers"):
        validate_startup(
            Settings(
                _env_file=None,
                environment="production",
                allow_anonymous=False,
                api_keys=["a-long-enough-api-key"],
                workers=4,
                allow_unsafe_multiworker=True,
            )
        )
