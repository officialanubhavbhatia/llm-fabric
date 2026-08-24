"""Authentication, tenant resolution and admission control.

Implemented as raw ASGI rather than Starlette's `BaseHTTPMiddleware` because the
gateway streams SSE. `BaseHTTPMiddleware` wraps the response body in an extra
task, which complicates cancellation and cleanup for long-lived streams — the
exact path where getting cleanup wrong leaks provider connections.

Order matters and is fixed: identify the caller, resolve the tenant, then admit
or shed. Nothing downstream runs for an unauthenticated request, and quota is
charged against a validated identity rather than a claimed one.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping
from typing import Any

from llm_fabric.deps.health import INFERENCE_PATHS, DependencyHealth
from llm_fabric.errors import (
    AuthenticationError,
    FabricError,
    InvalidRequestError,
)
from llm_fabric.identity.claims import Principal
from llm_fabric.identity.verifier import TokenVerifier
from llm_fabric.observability.logging import request_logger
from llm_fabric.observability.telemetry import Telemetry
from llm_fabric.observability.trace import TraceContext, bind_trace, reset_trace
from llm_fabric.tenancy.quota import QuotaLedger
from llm_fabric.tenancy.scope import TenantScope

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Probes and generated documentation never require a credential.
DEFAULT_PUBLIC_PATHS: frozenset[str] = frozenset(
    {"/healthz", "/readyz", "/docs", "/redoc", "/openapi.json"}
)

#: Header a delegation-capable caller uses to act for another tenant. It is
#: honoured only when the validated token carries the delegation scope; for
#: every other caller it is ignored entirely, never trusted.
TENANT_DELEGATION_HEADER = "x-tenant-id"

ANONYMOUS_ISSUER = "urn:myvista:anonymous"


def anonymous_principal(tenant_id: str = "public") -> Principal:
    """The identity used when authentication is switched off for local work."""
    return Principal(
        tenant_id=tenant_id,
        user_id="anonymous",
        subject=f"{ANONYMOUS_ISSUER}:{tenant_id}",
        issuer=ANONYMOUS_ISSUER,
    )


class AuthenticationMiddleware:
    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        *,
        verifier: TokenVerifier | None,
        auth_required: bool,
        quota_ledger: QuotaLedger | None = None,
        public_paths: Iterable[str] = DEFAULT_PUBLIC_PATHS,
        anonymous_tenant: str = "public",
        telemetry: Telemetry | None = None,
        max_request_bytes: int = 1_048_576,
        trusted_proxies: tuple[str, ...] = (),
        dependency_health: DependencyHealth | None = None,
    ) -> None:
        self._app = app
        self._verifier = verifier
        self._auth_required = auth_required
        self._quota = quota_ledger
        self._public_paths = frozenset(public_paths)
        self._anonymous_tenant = anonymous_tenant
        self._telemetry = telemetry
        self._max_request_bytes = max_request_bytes
        self._trusted_proxies = trusted_proxies
        self._dependency_health = dependency_health

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = _headers(scope)
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["client_ip"] = _client_ip(scope, headers, self._trusted_proxies)

        length = headers.get("content-length")
        if length is not None:
            try:
                size = int(length)
            except ValueError:
                size = -1
            if size > self._max_request_bytes:
                await _send_error(
                    send,
                    InvalidRequestError("request body exceeds the configured maximum"),
                    request_id,
                    TraceContext.start(request_id=request_id),
                )
                return

        trace = TraceContext.start(
            request_id=request_id,
            traceparent=headers.get("traceparent"),
        )
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        token = bind_trace(trace)

        try:
            if path in self._public_paths:
                principal = None
            else:
                if _is_inference(method, path):
                    self._admit_inference(path)
                if self._telemetry is not None:
                    with self._telemetry.span("auth"):
                        principal = await self._identify(headers)
                        trace = trace.with_principal(principal)
                        reset_trace(token)
                        token = bind_trace(trace)
                else:
                    principal = await self._identify(headers)
                    trace = trace.with_principal(principal)
                    reset_trace(token)
                    token = bind_trace(trace)
        except FabricError as exc:
            self._log_rejection(path, request_id, exc)
            await _send_error(send, exc, request_id, trace)
            reset_trace(token)
            return

        state["request_id"] = request_id
        state["trace"] = trace
        state["principal"] = principal
        state["tenant_scope"] = TenantScope.from_principal(principal) if principal else None

        admitted: TenantScope | None = None
        if principal is not None and self._quota is not None:
            try:
                admitted = TenantScope.from_principal(principal)
                self._quota.acquire_concurrency(admitted)
                self._quota.admit(admitted)
            except FabricError as exc:
                if admitted is not None:
                    self._quota.release_concurrency(admitted)
                    admitted = None
                self._log_rejection(path, request_id, exc, principal=principal)
                await _send_error(send, exc, request_id, trace)
                reset_trace(token)
                return

        try:
            await self._app(scope, receive, _with_trace_headers(send, request_id, trace))
        finally:
            if admitted is not None and self._quota is not None:
                self._quota.release_concurrency(admitted)
            reset_trace(token)

    async def _identify(self, headers: Mapping[str, str]) -> Principal:
        if not self._auth_required or self._verifier is None:
            return anonymous_principal(self._anonymous_tenant)

        credential = _presented_credential(headers)
        if credential is None:
            raise AuthenticationError(
                "missing credential: supply 'Authorization: Bearer <token>' or 'x-api-key'"
            )

        principal = await self._verifier.verify(credential)

        requested_tenant = headers.get(TENANT_DELEGATION_HEADER)
        if requested_tenant and requested_tenant != principal.tenant_id:
            # `delegate_to` refuses unless the validated token carries the
            # delegation scope, so an ordinary caller cannot reach another
            # tenant by setting a header.
            principal = principal.delegate_to(requested_tenant.strip())
        return principal

    def _admit_inference(self, path: str) -> None:
        """Refuse new provider work from cached health. No per-request probes."""
        health = self._dependency_health
        if health is None or health.allows_new_serving():
            return
        down = [
            name
            for name, snap in health.snapshots().items()
            if snap.required and snap.status.value in {"unhealthy", "recovering"}
        ]
        label = down[0] if down else "other"
        if self._telemetry is not None:
            with self._telemetry.span("admission.reject", path=path, dependency=label):
                self._telemetry.metrics.note_admission_rejection(label)
        refusal = health.refusal()
        if refusal is not None:
            raise refusal

    def _log_rejection(
        self,
        path: str,
        request_id: str,
        exc: FabricError,
        *,
        principal: Principal | None = None,
    ) -> None:
        extra: dict[str, Any] = {
            "path": path,
            "request_id": request_id,
            "error_type": exc.error_type,
            "status_code": exc.status_code,
        }
        if principal is not None:
            extra.update(principal.audit_fields())
        request_logger().warning("request rejected at the gateway edge", extra=extra)


def _client_ip(scope: Scope, headers: Mapping[str, str], trusted_proxies: tuple[str, ...]) -> str:
    """Peer address. `X-Forwarded-For` is ignored unless the peer is trusted."""
    peer = ""
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        peer = str(client[0] or "")
    if not trusted_proxies or peer not in trusted_proxies:
        return peer
    forwarded = headers.get("x-forwarded-for")
    if not forwarded:
        return peer
    return forwarded.split(",", 1)[0].strip() or peer


def _is_inference(method: str, path: str) -> bool:
    return method.upper() == "POST" and path in INFERENCE_PATHS


def _headers(scope: Scope) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


def _presented_credential(headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("authorization")
    if authorization:
        auth_scheme, _, value = authorization.partition(" ")
        if auth_scheme.lower() == "bearer" and value.strip():
            return value.strip()
    api_key = headers.get("x-api-key")
    return api_key.strip() if api_key and api_key.strip() else None


def _with_trace_headers(send: Send, request_id: str, trace: TraceContext) -> Send:
    """Stamp correlation headers onto the response without buffering the body."""

    async def wrapped(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            headers = list(message.get("headers", []))
            # A route may already have stamped the correlation id. Appending a
            # second copy would produce a comma-joined header value, so each is
            # added only when absent.
            present = {key.lower() for key, _ in headers}
            if b"x-fabric-request-id" not in present:
                headers.append((b"x-fabric-request-id", request_id.encode("latin-1")))
            if b"traceparent" not in present:
                headers.append((b"traceparent", trace.to_traceparent().encode("latin-1")))
            security = (
                (b"x-content-type-options", b"nosniff"),
                (b"x-frame-options", b"DENY"),
                (b"referrer-policy", b"no-referrer"),
                (b"x-permitted-cross-domain-policies", b"none"),
            )
            for name, value in security:
                if name not in present:
                    headers.append((name, value))
            message = {**message, "headers": headers}
        await send(message)

    return wrapped


async def _send_error(
    send: Send,
    exc: FabricError,
    request_id: str,
    trace: TraceContext,
) -> None:
    body = json.dumps(
        {
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "request_id": request_id,
                **_error_extras(exc, trace),
            }
        }
    ).encode("utf-8")

    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("latin-1")),
        (b"x-fabric-request-id", request_id.encode("latin-1")),
        (b"traceparent", trace.to_traceparent().encode("latin-1")),
    ]
    if isinstance(exc, AuthenticationError):
        # RFC 6750: a 401 must say how to authenticate.
        headers.append((b"www-authenticate", b'Bearer realm="myvista-llm-fabric"'))
    retry_after = getattr(exc, "retry_after_s", None)
    if isinstance(retry_after, int):
        headers.append((b"retry-after", str(retry_after).encode("latin-1")))
    headers.extend(
        [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
        ]
    )

    await send({"type": "http.response.start", "status": exc.status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _error_extras(exc: FabricError, trace: TraceContext) -> dict[str, object]:
    extras: dict[str, object] = {}
    if getattr(exc, "retryable", None) is True:
        extras["retryable"] = True
        extras["trace_id"] = trace.trace_id
    return extras
