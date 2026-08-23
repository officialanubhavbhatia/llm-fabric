"""Application assembly.

This is the only place where the layers are wired to each other. Each collaborator
can be supplied explicitly, which is what the test suite does to run the whole
gateway against injected providers with no network access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_fabric import __version__
from llm_fabric.config import Settings, get_settings
from llm_fabric.errors import FabricError
from llm_fabric.gateway.routes import chat, health, models, usage
from llm_fabric.observability.logging import configure_logging, request_logger
from llm_fabric.observability.metering import InMemoryMeter
from llm_fabric.router.engine import Router
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.base import Provider
from llm_fabric.serving.factory import ProviderFactory

DESCRIPTION = """
An LLM gateway with policy-based routing across providers.

Send an OpenAI-compatible chat completion to `/v1/chat/completions`. Name a model
directly to pin it, or use an alias such as `auto` to let the fabric choose under
a routing policy. Every response reports which model actually served it in the
`x-fabric-served-model` header.
"""


def _error_response(request_id: str | None, exc: FabricError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "request_id": request_id,
            }
        },
    )


def create_app(
    *,
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    provider_overrides: dict[str, Provider] | None = None,
    meter: InMemoryMeter | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logger = configure_logging(settings.log_level)

    registry = registry or ModelRegistry.from_yaml(settings.registry_path)
    providers = ProviderFactory(settings, overrides=provider_overrides)
    meter = meter or InMemoryMeter()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not settings.auth_required:
            logger.warning(
                "gateway is running without authentication",
                extra={"hint": "set LLM_FABRIC_API_KEYS to require a client key"},
            )
        logger.info(
            "fabric ready",
            extra={
                "version": __version__,
                "enabled_models": len(registry.enabled_models()),
                "providers": sorted(registry.providers_in_use()),
                "default_policy": settings.default_policy,
            },
        )
        try:
            yield
        finally:
            await providers.aclose()

    app = FastAPI(
        title="MyVista LLM Fabric",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    app.state.settings = settings
    app.state.registry = registry
    app.state.providers = providers
    app.state.meter = meter
    app.state.router = Router(
        registry,
        providers,
        default_policy=settings.default_policy,
        max_attempts=settings.max_attempts,
    )

    @app.exception_handler(FabricError)
    async def _handle_fabric_error(request: Request, exc: FabricError) -> JSONResponse:
        request_id = request.headers.get("x-request-id")
        request_logger().warning(
            "request rejected",
            extra={
                "path": request.url.path,
                "error_type": exc.error_type,
                "status_code": exc.status_code,
                "request_id": request_id,
            },
        )
        return _error_response(request_id, exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Reshaped into the same error envelope as everything else, so a client
        # has exactly one error format to parse. Only the three fields a caller
        # can act on are copied across: pydantic also attaches the original
        # exception object, which is neither serializable nor useful to a client.
        details = [
            {
                "location": [str(part) for part in error.get("loc", ())],
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "request failed validation",
                    "type": "invalid_request_error",
                    "request_id": request.headers.get("x-request-id"),
                    "details": details,
                }
            },
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(usage.router)

    return app
