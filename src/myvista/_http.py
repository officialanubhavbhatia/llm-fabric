"""Shared HTTP plumbing: URL normalisation, retries, timeouts, error mapping."""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from myvista.errors import (
    APIConnectionError,
    APITimeoutError,
    MyVistaError,
    error_from_response,
)

DEFAULT_BASE_URL = "http://127.0.0.1:47317"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


def resolve_base_url(base_url: str | None) -> str:
    raw = (base_url or os.environ.get("MYVISTA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    return raw


def resolve_api_key(api_key: str | None) -> str | None:
    return api_key if api_key is not None else os.environ.get("MYVISTA_API_KEY")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def auth_headers(api_key: str | None, request_id: str) -> dict[str, str]:
    headers = {
        "accept": "application/json",
        "x-request-id": request_id,
    }
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def parse_json(response: httpx.Response) -> dict[str, Any] | None:
    if not response.content:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    payload = parse_json(response)
    raise error_from_response(
        status_code=response.status_code,
        payload=payload,
        request_id=response.headers.get("x-fabric-request-id"),
        retry_after=response.headers.get("retry-after"),
    )


def should_retry(exc: BaseException | None, response: httpx.Response | None) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if response is None:
        return False
    return response.status_code in _RETRY_STATUSES


def backoff_s(attempt: int, retry_after_s: int | None) -> float:
    if retry_after_s is not None:
        return float(retry_after_s)
    return float(min(8.0, (0.25 * (2**attempt)) + random.random() * 0.1))


def iter_sse(text: str) -> Iterator[dict[str, Any]]:
    for block in text.split("\n\n"):
        data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            return
        try:
            parsed = json.loads(data)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            if "error" in parsed and "choices" not in parsed:
                raise error_from_response(
                    status_code=200,
                    payload=parsed,
                    request_id=None,
                    retry_after=None,
                )
            yield parsed


def headers_map(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items()}


def wrap_transport_error(exc: BaseException) -> MyVistaError:
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(str(exc) or "request timed out")
    if isinstance(exc, httpx.HTTPError):
        return APIConnectionError(str(exc) or "connection failed")
    raise exc


def sleep(seconds: float) -> None:
    time.sleep(seconds)
