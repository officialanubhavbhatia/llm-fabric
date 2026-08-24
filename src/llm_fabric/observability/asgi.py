"""ASGI wrapper that times every HTTP request and opens the `request` span.

Lives outside Starlette middleware so streaming responses are not buffered.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from llm_fabric.observability.telemetry import Telemetry, bind_telemetry, reset_telemetry
from llm_fabric.observability.trace import bind_trace, current_trace, reset_trace

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


class RequestTelemetryMiddleware:
    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        telemetry: Telemetry,
    ) -> None:
        self._app = app
        self._telemetry = telemetry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "")
        started = time.perf_counter()
        status_holder = {"code": 500}
        self._telemetry.metrics.active_requests.inc()

        async def wrapped_send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = int(message.get("status", 500))
            await send(message)

        token = bind_telemetry(self._telemetry)
        rebound = None
        try:
            with self._telemetry.span(
                "request",
                http_method=method,
                http_route=self._telemetry.metrics.path_label(path),
            ):
                await self._app(scope, receive, wrapped_send)
                # Auth unbinds the request-local trace when it returns. Re-bind
                # the finished, tenant-stamped context so the request span is
                # journalled under the tenant that actually ran it.
                state = scope.get("state") or {}
                finished = state.get("trace")
                if finished is not None and current_trace() is None:
                    rebound = bind_trace(finished)
        finally:
            if rebound is not None:
                reset_trace(rebound)
            reset_telemetry(token)
            self._telemetry.metrics.active_requests.dec()
            self._telemetry.metrics.observe_http(
                method=method,
                path=path,
                status=status_holder["code"],
                duration_s=time.perf_counter() - started,
            )
