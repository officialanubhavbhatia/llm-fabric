"""Authoritative process initialization.

Every supported serving entrypoint constructs the ASGI application through
`create_app`. `create_app` always calls `initialize_runtime`. There is no
second production validation path in the CLI, the container CMD, or a
Makefile: those processes either invoke `create_app` or they do not serve.

Supported serving entrypoints:

- `python -m llm_fabric` / the `llm-fabric` console script (uvicorn factory)
- `uvicorn llm_fabric.gateway.app:create_app --factory`
- the production image `CMD ["python", "-m", "llm_fabric"]`
- tests that call `create_app`

Unsupported: a module-level `app` object, gunicorn, or any ASGI server that
imports the gateway without calling `create_app`. Those are not given a
weaker guarantee; they are not a supported way to serve.

Configuration validity (`validate_startup`) and dependency reachability
(`probe_distributed_state`) both happen here. Ongoing readiness after the
process is serving is a different question and is not decided here.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_fabric.config import Settings, validate_startup
from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.runtime import probe_distributed_state


@dataclass(frozen=True)
class Runtime:
    """The result of one successful initialization against these settings."""

    settings: Settings


def initialize_runtime(settings: Settings) -> Runtime:
    """Refuse to proceed when this process is not allowed to serve.

    Production always probes PostgreSQL and Redis with the same URLs the
    serving runtime will use. Development and test do not require those
    dependencies: in-memory stores remain valid when the environment name is
    an explicit non-production value.
    """
    validate_startup(settings)
    if settings.environment == "production":
        try:
            probe_distributed_state(settings)
        except ConfigurationError as exc:
            message = exc.message
            if not message.startswith("production startup validation failed"):
                message = f"production startup validation failed: {message}"
            raise ConfigurationError(message) from None
    return Runtime(settings=settings)
