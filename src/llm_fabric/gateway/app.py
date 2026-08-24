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
from llm_fabric.deps.health import DependencyHealth
from llm_fabric.deps.monitor import DependencyMonitor
from llm_fabric.errors import ConfigurationError, FabricError
from llm_fabric.gateway.middleware import DEFAULT_PUBLIC_PATHS, AuthenticationMiddleware
from llm_fabric.gateway.routes import (
    chat,
    dev_auth,
    evals,
    health,
    intents,
    models,
    observability,
    routing,
    usage,
)
from llm_fabric.heal.controls import OperationalControls
from llm_fabric.heal.engine import HealController
from llm_fabric.identity.apikey import ApiKeyVerifier
from llm_fabric.identity.dev import DevIdentityProvider
from llm_fabric.identity.oidc import OIDCTokenVerifier
from llm_fabric.identity.revocation import (
    RevokingVerifier,
    TokenRevocationStore,
)
from llm_fabric.identity.verifier import TokenVerifier
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.intent.embeddings import HashingEmbedder, resolve_embedder
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.observability.asgi import RequestTelemetryMiddleware
from llm_fabric.observability.langfuse import build_langfuse
from llm_fabric.observability.logging import configure_logging, request_logger
from llm_fabric.observability.metering import UsageMeter, build_meter
from llm_fabric.observability.otel import FabricTracer, try_otlp_exporter
from llm_fabric.observability.telemetry import Telemetry
from llm_fabric.router.engine import Router
from llm_fabric.router.fallback import FallbackBudget
from llm_fabric.router.health import BreakerPolicy, HealthTracker
from llm_fabric.router.plan import TenantRoutingPolicies
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.runtime import initialize_runtime
from llm_fabric.serving.base import Provider
from llm_fabric.serving.factory import ProviderFactory
from llm_fabric.storage.redis import RedisCache
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.storage.runtime import (
    build_analytics,
    build_engine,
    build_health,
    build_quota,
    build_redis,
    build_revocation,
)
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.quota import QuotaLedger

DESCRIPTION = """
An LLM gateway with policy-based routing across providers.

Send an OpenAI-compatible chat completion to `/v1/chat/completions`. Name a model
directly to pin it, or use an alias such as `auto` to let the fabric choose under
a routing policy. Every response reports which model actually served it in the
`x-fabric-served-model` header.
"""


def build_breaker_policy(settings: Settings) -> BreakerPolicy:
    """Circuit-breaker thresholds from configuration.

    Assembled here rather than on `Settings` so that configuration stays a leaf
    module: the router already depends on config, and the reverse edge would
    close a cycle.
    """
    return BreakerPolicy(
        consecutive_failures=settings.breaker_consecutive_failures,
        error_rate=settings.breaker_error_rate,
        minimum_samples=settings.breaker_minimum_samples,
        open_duration_s=settings.breaker_open_duration_s,
        half_open_successes=settings.breaker_half_open_successes,
        max_concurrency=settings.effective_breaker_max_concurrency,
    )


def build_fallback_budget(settings: Settings) -> FallbackBudget:
    """Fallback ceilings. Depth follows `max_attempts`, which counts the primary."""
    return FallbackBudget(
        max_depth=max(0, settings.max_attempts - 1),
        max_cost_usd=settings.fallback_max_cost_usd,
        max_latency_ms=settings.fallback_max_latency_ms,
    )


def build_verifier(settings: Settings) -> TokenVerifier | None:
    """Construct the identity source named by configuration.

    Returns `None` only when authentication is switched off. Production
    never reaches this branch: `validate_startup` refuses first.
    """
    mode = settings.effective_auth_mode

    if mode == "disabled":
        return None

    if mode == "api_key":
        credentials = settings.resolved_credentials()
        if not credentials:
            raise ConfigurationError("auth_mode is 'api_key' but no credentials are configured")
        return ApiKeyVerifier(credentials)

    if mode == "dev":
        if not settings.dev_auth_secret:
            raise ConfigurationError("auth_mode is 'dev' but LLM_FABRIC_DEV_AUTH_SECRET is unset")
        return DevIdentityProvider(
            secret=settings.dev_auth_secret,
            mapping=settings.claims_mapping,
        )

    if not settings.oidc_issuer or not settings.oidc_audience:
        raise ConfigurationError(
            "auth_mode is 'oidc' but LLM_FABRIC_OIDC_ISSUER or LLM_FABRIC_OIDC_AUDIENCE is unset"
        )
    return OIDCTokenVerifier(
        issuer=settings.oidc_issuer,
        audience=settings.oidc_audience,
        jwks_uri=settings.oidc_jwks_uri,
        mapping=settings.claims_mapping,
        jwks_cache_seconds=settings.oidc_jwks_cache_seconds,
    )


def _error_response(
    request_id: str | None, exc: FabricError, *, trace_id: str | None = None
) -> JSONResponse:
    headers: dict[str, str] = {}
    retry_after = getattr(exc, "retry_after_s", None)
    if isinstance(retry_after, int):
        headers["retry-after"] = str(retry_after)
    payload: dict[str, object] = {
        "message": exc.message,
        "type": exc.error_type,
        "request_id": request_id,
    }
    if getattr(exc, "retryable", None) is True:
        payload["retryable"] = True
        if trace_id:
            payload["trace_id"] = trace_id
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload},
        headers=headers or None,
    )


def _otel_headers(raw: str | None) -> dict[str, str] | None:
    """Parse `k=v,k2=v2` from the secret env var. Never log the values."""
    if not raw or not raw.strip():
        return None
    parsed: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and value:
            parsed[key] = value
    return parsed or None


def create_app(
    *,
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    provider_overrides: dict[str, Provider] | None = None,
    meter: UsageMeter | None = None,
    verifier: TokenVerifier | None = None,
    stores: TenantStores | None = None,
    cache: TenantScopedCache | None = None,
    quota: QuotaLedger | None = None,
    health_tracker: HealthTracker | None = None,
    tenant_routing: TenantRoutingPolicies | None = None,
    intent: IntentCascade | None = None,
    telemetry: Telemetry | None = None,
    revocation_store: TokenRevocationStore | None = None,
    dependency_health: DependencyHealth | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    runtime = initialize_runtime(settings)
    logger = configure_logging(settings.log_level)

    registry = registry or ModelRegistry.from_yaml(settings.registry_path)
    providers = ProviderFactory(settings, overrides=provider_overrides)
    telemetry = telemetry or Telemetry(
        tracer=FabricTracer(
            exporter=try_otlp_exporter(
                settings.otel_exporter_otlp_endpoint,
                headers=_otel_headers(settings.otel_exporter_otlp_headers),
                certificate_file=settings.otel_exporter_otlp_certificate,
            )
        ),
        langfuse=build_langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        ),
    )
    engine = None
    if stores is None:
        engine = build_engine(settings) if settings.database_url else None
        stores = TenantStores(engine=engine)
    else:
        engine = getattr(stores, "engine", None)
    needs_redis = (
        quota is None
        or revocation_store is None
        or health_tracker is None
        or cache is None
        or (meter is None and engine is not None)
    )
    redis_client = build_redis(settings) if settings.redis_url and needs_redis else None
    if meter is None:
        meter = build_meter(engine=engine, redis_client=redis_client)
    if dependency_health is None:
        dependency_health = DependencyHealth(
            postgres=engine is not None,
            redis=redis_client is not None,
            telemetry=bool(settings.otel_exporter_otlp_endpoint),
            fail_threshold=settings.health_fail_threshold,
            recovery_threshold=settings.health_recovery_threshold,
            metrics=telemetry.metrics,
        )
    else:
        dependency_health.bind_metrics(telemetry.metrics)
    bind_health = getattr(meter, "bind_dependency_health", None)
    if callable(bind_health):
        bind_health(dependency_health)
    cache = cache or TenantScopedCache(
        audit=stores.audit,
        redis_cache=RedisCache(redis_client) if redis_client is not None else None,
    )
    quota = quota or build_quota(settings, redis_client=redis_client)
    inner_verifier = verifier if verifier is not None else build_verifier(settings)
    revocation = revocation_store or build_revocation(settings, redis_client=redis_client)
    bind_health = getattr(quota, "bind_dependency_health", None)
    if callable(bind_health):
        bind_health(dependency_health)
    bind_health = getattr(revocation, "bind_dependency_health", None)
    if callable(bind_health):
        bind_health(dependency_health)
    resolved_verifier: TokenVerifier | None = (
        RevokingVerifier(
            inner_verifier,
            revocation,
            required_scopes=settings.required_scopes,
        )
        if inner_verifier is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Same function as create_app. A process that reached lifespan without
        # going through initialize_runtime still cannot serve production traffic.
        initialize_runtime(settings)
        if settings.environment == "production":
            # Defense in depth: production must never reach "running without
            # authentication". Raise rather than warn, even if validation above
            # was somehow skipped.
            if not settings.auth_required or resolved_verifier is None:
                raise ConfigurationError("production refused to start without authentication")
            if isinstance(inner_verifier, OIDCTokenVerifier):
                await inner_verifier.warmup()
        elif not settings.auth_required:
            logger.warning(
                "gateway is running without authentication",
                extra={
                    "environment": settings.environment,
                    "hint": "set LLM_FABRIC_OIDC_ISSUER, LLM_FABRIC_DEV_AUTH_SECRET, "
                    "or LLM_FABRIC_API_CREDENTIALS, or keep this only for "
                    "development and test",
                    "anonymous_tenant": settings.anonymous_tenant,
                },
            )
        if settings.effective_auth_mode == "dev":
            if settings.environment == "production":
                raise ConfigurationError(
                    "production refused to start: the development identity provider is forbidden"
                )
            logger.warning(
                "development identity provider is enabled",
                extra={"hint": "never enable LLM_FABRIC_DEV_AUTH_SECRET in production"},
            )

        unready: list[str] = []
        for name in sorted(registry.providers_in_use()):
            if providers.constructible(name):
                continue
            unready.append(name)
            logger.warning(
                "provider is not constructible",
                extra={"provider": name, "hint": "check API keys for enabled models"},
            )
        logger.info(
            "fabric ready",
            extra={
                "version": __version__,
                "enabled_models": len(registry.enabled_models()),
                "providers": sorted(registry.providers_in_use()),
                "unready_providers": unready,
                "default_policy": settings.default_policy,
                **settings.describe_auth(),
            },
        )
        monitor = DependencyMonitor(
            dependency_health,
            database_url=settings.database_url if engine is not None else None,
            redis_url=settings.redis_url if redis_client is not None else None,
            interval_s=settings.health_probe_interval_s,
            timeout_s=settings.health_probe_timeout_s,
        )
        app.state.dependency_monitor = monitor
        await monitor.start()
        try:
            yield
        finally:
            await monitor.stop()
            await providers.aclose()
            await telemetry.aclose()
            if resolved_verifier is not None:
                await resolved_verifier.aclose()

    production = settings.environment == "production"
    app = FastAPI(
        title="MyVista LLM Fabric",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None,
        openapi_url=None if production else "/openapi.json",
    )

    app.state.settings = settings
    app.state.runtime = runtime
    app.state.registry = registry
    app.state.providers = providers
    app.state.meter = meter
    app.state.telemetry = telemetry
    app.state.stores = stores
    app.state.cache = cache
    app.state.quota = quota
    app.state.verifier = resolved_verifier
    app.state.revocation = revocation
    app.state.health = health_tracker or build_health(
        settings, redis_client=redis_client, policy=build_breaker_policy(settings)
    )
    app.state.analytics = build_analytics(settings)
    app.state.redis = redis_client
    app.state.dependency_health = dependency_health
    app.state.tenant_routing = tenant_routing or TenantRoutingPolicies()
    app.state.controls = OperationalControls()
    app.state.router = Router(
        registry,
        providers,
        default_policy=settings.default_policy,
        max_attempts=settings.max_attempts,
        health=app.state.health,
        tenant_policies=app.state.tenant_routing,
        fallback_budget=build_fallback_budget(settings),
        controls=app.state.controls,
    )
    # Built only when enabled: an unused cascade would still hold a taxonomy and
    # an embedder, and would appear in health output as if it were serving.
    if intent is not None:
        app.state.intent = intent
    elif settings.intent_classification_enabled or settings.intent_shadow:
        try:
            embedder = resolve_embedder(settings.intent_embedder)
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "intent embedder unavailable; using HashingEmbedder",
                extra={"embedder": settings.intent_embedder, "error": str(exc)},
            )
            embedder = HashingEmbedder()
        app.state.intent = build_offline_cascade(
            bootstrap_taxonomy(),
            cache,
            embedder=embedder,
            l4_rerank=settings.intent_l4_rerank,
        )
    else:
        app.state.intent = None
    if app.state.intent is not None:
        app.state.controls.classifiers.pin(app.state.intent)

    def _install_cascade(cascade: IntentCascade) -> None:
        app.state.intent = cascade

    app.state.heal = HealController(
        controls=app.state.controls,
        health=app.state.health,
        registry=registry,
        cache=cache,
        prompts=stores.prompts,
        incidents=stores.incidents,
        remediations=stores.remediations,
        jobs=stores.learning_jobs,
        baselines=stores.drift_baselines,
        install_cascade=_install_cascade,
    )

    public_paths = set(DEFAULT_PUBLIC_PATHS)
    public_paths.add("/metrics")
    if settings.effective_auth_mode == "dev":
        # Obtaining a development token cannot itself require a token.
        public_paths.add(dev_auth.DEV_TOKEN_PATH)

    app.add_middleware(
        AuthenticationMiddleware,
        verifier=resolved_verifier,
        auth_required=settings.auth_required,
        quota_ledger=quota,
        public_paths=public_paths,
        anonymous_tenant=settings.anonymous_tenant,
        telemetry=telemetry,
        max_request_bytes=settings.max_request_bytes,
        trusted_proxies=tuple(settings.trusted_proxies),
        dependency_health=dependency_health,
    )
    if settings.cors_origins:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Api-Key"],
        )
    # Outer: times the full request and binds Telemetry for the router.
    app.add_middleware(RequestTelemetryMiddleware, telemetry=telemetry)

    @app.exception_handler(FabricError)
    async def _handle_fabric_error(request: Request, exc: FabricError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )
        trace = getattr(request.state, "trace", None)
        trace_id = getattr(trace, "trace_id", None)
        request_logger().warning(
            "request rejected",
            extra={
                "path": request.url.path,
                "error_type": exc.error_type,
                "status_code": exc.status_code,
                "request_id": request_id,
            },
        )
        return _error_response(request_id, exc, trace_id=trace_id)

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
                    "request_id": getattr(request.state, "request_id", None)
                    or request.headers.get("x-request-id"),
                    "details": details,
                }
            },
        )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(routing.router)
    app.include_router(intents.router)
    app.include_router(evals.router)
    app.include_router(usage.router)
    app.include_router(observability.router)
    if settings.effective_auth_mode == "dev":
        app.include_router(dev_auth.router)

    return app
