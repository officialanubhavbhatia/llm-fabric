"""Synchronous MyVista client.

Ordinary inference is one object and one call. Routing, retries and provenance
are handled; callers do not configure them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from myvista._http import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    auth_headers,
    backoff_s,
    headers_map,
    iter_sse,
    new_request_id,
    parse_json,
    raise_for_status,
    resolve_api_key,
    resolve_base_url,
    should_retry,
    sleep,
    wrap_transport_error,
)
from myvista.errors import MyVistaError, UnsupportedError
from myvista.types import ChatChunk, ChatCompletion, Response

_QUALITY_POLICY = {
    "high": "quality_first",
    "standard": "balanced",
    "low": "cost_first",
    "fast": "latency_first",
}


class MyVista:
    """Talk to a running fabric. Defaults to the local gateway."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = resolve_base_url(base_url)
        self.api_key = resolve_api_key(api_key)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )
        self.chat = _Chat(self)
        self.responses = _Responses(self)
        self.embeddings = _Embeddings()
        self.intents = _Intents(self)
        self.routes = _Routes(self)
        self.evals = _Evals(self)
        self.traces = _Traces(self)
        self.agents = _Agents()
        self.models = _Models(self)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MyVista:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> httpx.Response:
        rid = request_id or new_request_id()
        headers = auth_headers(self.api_key, rid)
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, path, json=json_body, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries or not should_retry(exc, None):
                    raise wrap_transport_error(exc) from exc
                sleep(backoff_s(attempt, None))
                continue
            if response.is_success:
                return response
            if attempt >= self.max_retries or not should_retry(None, response):
                raise_for_status(response)
            retry_after = None
            raw = response.headers.get("retry-after")
            if raw:
                try:
                    retry_after = int(raw)
                except ValueError:
                    retry_after = None
            sleep(backoff_s(attempt, retry_after))
        if last_error is not None:
            raise wrap_transport_error(last_error)
        raise MyVistaError("request failed")


class _Chat:
    def __init__(self, client: MyVista) -> None:
        self.completions = _Completions(client)


class _Completions:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str = "auto",
        messages: list[dict[str, str]],
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> ChatCompletion | Iterator[ChatChunk]:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(extra)
        response = self._client.request("POST", "/v1/chat/completions", json_body=body)
        raise_for_status(response)
        if stream:
            return (ChatChunk.from_payload(item) for item in iter_sse(response.text))
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty chat response")
        return ChatCompletion.from_http(payload, headers_map(response.headers))


class _Responses:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def create(
        self,
        *,
        input: str,
        model: str = "auto",
        quality: str | None = None,
        latency_slo_ms: float | None = None,
        **extra: Any,
    ) -> Response:
        """The constitution's simple path: a string in, text out."""
        messages = [{"role": "user", "content": input}]
        chosen = model
        policy = _QUALITY_POLICY.get((quality or "").lower()) if quality else None
        if policy is not None or latency_slo_ms is not None:
            preview = self._client.routes.preview(
                model=model,
                messages=messages,
                policy=policy,
                latency_slo_ms=latency_slo_ms,
            )
            selected = preview.get("selected") or {}
            chosen_id = selected.get("model_id") or selected.get("id")
            if chosen_id:
                chosen = str(chosen_id)
        completion = self._client.chat.completions.create(
            model=chosen,
            messages=messages,
            **extra,
        )
        if isinstance(completion, ChatCompletion):
            return Response.from_completion(completion)
        raise MyVistaError("streaming is not supported on responses.create")


class _Embeddings:
    def create(self, **_: Any) -> None:
        raise UnsupportedError(
            "embeddings are not served by this fabric yet. "
            "There is no /v1/embeddings route and no vectors are synthesized."
        )


class _Agents:
    def run(self, **_: Any) -> None:
        raise UnsupportedError(
            "agents are not served by this fabric yet. "
            "There is no agent runtime and no run is synthesized."
        )


class _Intents:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def classify(self, input: str, *, language: str = "en") -> dict[str, Any]:
        response = self._client.request(
            "POST",
            "/v1/intents/classify",
            json_body={"input": input, "language": language},
        )
        raise_for_status(response)
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty classify response")
        return payload


class _Routes:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def preview(
        self,
        *,
        model: str = "auto",
        messages: list[dict[str, str]] | None = None,
        policy: str | None = None,
        latency_slo_ms: float | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, **extra}
        if messages is not None:
            body["messages"] = messages
        if policy is not None:
            body["policy"] = policy
        if latency_slo_ms is not None:
            body["latency_slo_ms"] = latency_slo_ms
        response = self._client.request("POST", "/v1/routes/preview", json_body=body)
        raise_for_status(response)
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty route preview")
        return payload


class _Evals:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def run(self, suite: str = "ci") -> dict[str, Any]:
        response = self._client.request("POST", "/v1/evals/run", json_body={"suite": suite})
        raise_for_status(response)
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty eval response")
        return payload


class _Traces:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def list(self) -> dict[str, Any]:
        response = self._client.request("GET", "/v1/observability/traces")
        raise_for_status(response)
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty traces response")
        return payload

    def get(self, trace_id: str) -> dict[str, Any]:
        response = self._client.request("GET", f"/v1/observability/traces/{trace_id}")
        raise_for_status(response)
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty trace response")
        return payload


class _Models:
    def __init__(self, client: MyVista) -> None:
        self._client = client

    def list(self) -> dict[str, Any]:
        response = self._client.request("GET", "/v1/models")
        raise_for_status(response)
        payload = parse_json(response)
        if payload is None:
            raise MyVistaError("empty models response")
        return payload
