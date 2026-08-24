"""Request-scoped dependencies.

Shared components live on `app.state`; per-request identity is put on the
request by `AuthenticationMiddleware` before any route runs. Routes reach both
only through these helpers, so no endpoint invents its own idea of who is
calling or which tenant it may read.

**Every dependency here is `async def`, and must stay that way.** FastAPI runs a
*sync* dependency in a worker thread, on the assumption that it might block.
None of these block — they are attribute lookups and `isinstance` checks — so
each thread hop would cost far more than the work it dispatches. A profile of
the chat path found 24 such hops per request, dominating the request cost. Add a
new dependency as `def` and that regression comes straight back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from fastapi import Request

from llm_fabric.config import Settings
from llm_fabric.deps.health import DependencyHealth
from llm_fabric.errors import AuthenticationError, AuthorizationError, ConfigurationError
from llm_fabric.identity.claims import Principal
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.observability.metering import UsageMeter
from llm_fabric.observability.telemetry import Telemetry
from llm_fabric.observability.trace import TraceContext
from llm_fabric.router.engine import Router
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.factory import ProviderFactory
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.quota import QuotaLedger
from llm_fabric.tenancy.scope import TenantScope


def _from_state[T](request: Request, name: str, expected: type[T]) -> T:
    """Fetch a component from application state, checking its type.

    Starlette's `state` is an untyped bag. Verifying the type here turns a
    mis-wired application into a clear failure at the first request rather than
    an `AttributeError` somewhere deeper.
    """
    value = getattr(request.app.state, name, None)
    if not isinstance(value, expected):
        raise ConfigurationError(
            f"application state '{name}' is missing or is not a {expected.__name__}"
        )
    return value


async def get_settings(request: Request) -> Settings:
    return _from_state(request, "settings", Settings)


async def get_registry(request: Request) -> ModelRegistry:
    return _from_state(request, "registry", ModelRegistry)


async def get_router(request: Request) -> Router:
    return _from_state(request, "router", Router)


async def get_meter(request: Request) -> UsageMeter:
    return _from_state(request, "meter", UsageMeter)


async def get_telemetry(request: Request) -> Telemetry:
    return _from_state(request, "telemetry", Telemetry)


async def get_providers(request: Request) -> ProviderFactory:
    return _from_state(request, "providers", ProviderFactory)


async def get_dependency_health(request: Request) -> DependencyHealth:
    return _from_state(request, "dependency_health", DependencyHealth)


async def get_stores(request: Request) -> TenantStores:
    return _from_state(request, "stores", TenantStores)


async def get_cache(request: Request) -> TenantScopedCache:
    return _from_state(request, "cache", TenantScopedCache)


async def get_quota(request: Request) -> QuotaLedger:
    return _from_state(request, "quota", QuotaLedger)


async def get_intent_cascade(request: Request) -> IntentCascade | None:
    """The classifier, or `None` when intent classification is switched off.

    Optional by design: the cascade is only built when
    `LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED` is set, so callers must handle
    its absence rather than assume every request carries an intent.
    """
    cascade = getattr(request.app.state, "intent", None)
    return cascade if isinstance(cascade, IntentCascade) else None


async def get_classify_cascade(request: Request) -> IntentCascade:
    """Cascade for `POST /v1/intents/classify`.

    Uses the serving-path cascade when it is on. Otherwise builds a separate
    offline cascade so classification stays available without putting it in
    front of every chat request.
    """
    serving = await get_intent_cascade(request)
    if serving is not None:
        return serving
    existing = getattr(request.app.state, "classify_cascade", None)
    if isinstance(existing, IntentCascade):
        return existing
    from llm_fabric.intent.bootstrap import bootstrap_taxonomy
    from llm_fabric.intent.embeddings import HashingEmbedder, resolve_embedder
    from llm_fabric.intent.factory import build_offline_cascade

    settings = getattr(request.app.state, "settings", None)
    embedder = HashingEmbedder()
    l4_rerank = False
    if settings is not None:
        try:
            embedder = resolve_embedder(settings.intent_embedder)
        except (ValueError, RuntimeError):
            embedder = HashingEmbedder()
        l4_rerank = bool(settings.intent_l4_rerank)
    cascade = build_offline_cascade(
        bootstrap_taxonomy(),
        await get_cache(request),
        embedder=embedder,
        l4_rerank=l4_rerank,
    )
    request.app.state.classify_cascade = cascade
    return cascade


async def get_request_id(request: Request) -> str:
    """The correlation id the middleware assigned, echoing any caller value."""
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str):
        return request_id
    return request.headers.get("x-request-id") or ""


async def get_trace(request: Request) -> TraceContext:
    trace = getattr(request.state, "trace", None)
    if not isinstance(trace, TraceContext):
        raise AuthenticationError("request has no trace context")
    return trace


async def get_optional_principal(request: Request) -> Principal | None:
    principal = getattr(request.state, "principal", None)
    return principal if isinstance(principal, Principal) else None


async def get_principal(request: Request) -> Principal:
    principal = await get_optional_principal(request)
    if principal is None:
        raise AuthenticationError("this endpoint requires an authenticated caller")
    return principal


async def get_tenant_scope(request: Request) -> TenantScope:
    scope = getattr(request.state, "tenant_scope", None)
    if isinstance(scope, TenantScope):
        return scope
    return TenantScope.from_principal(await get_principal(request))


def require_scopes(*required: str) -> Callable[[Request], Awaitable[Principal]]:
    """Refuse a caller whose token lacks every one of `required`."""
    needed: Sequence[str] = required

    async def dependency(request: Request) -> Principal:
        principal = await get_principal(request)
        missing = [scope for scope in needed if not principal.has_scope(scope)]
        if missing:
            raise AuthorizationError(f"missing required scope(s): {', '.join(sorted(missing))}")
        return principal

    return dependency


def require_roles(*required: str) -> Callable[[Request], Awaitable[Principal]]:
    """Refuse a caller holding none of `required`."""
    needed: Sequence[str] = required

    async def dependency(request: Request) -> Principal:
        principal = await get_principal(request)
        if not principal.has_any_role(needed):
            raise AuthorizationError(f"requires one of role(s): {', '.join(sorted(needed))}")
        return principal

    return dependency
