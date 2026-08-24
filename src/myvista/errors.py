"""Typed errors the SDK raises. Names match the gateway's `error.type`."""

from __future__ import annotations

from typing import Any


class MyVistaError(Exception):
    """Base class for every error the SDK raises deliberately."""

    error_type: str = "api_error"
    status_code: int | None = None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        retry_after_s: int | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code if status_code is not None else self.status_code
        self.request_id = request_id
        self.retry_after_s = retry_after_s
        self.details = details or []


class AuthenticationError(MyVistaError):
    error_type = "authentication_error"
    status_code = 401


class AuthorizationError(MyVistaError):
    error_type = "permission_error"
    status_code = 403


class InvalidRequestError(MyVistaError):
    error_type = "invalid_request_error"
    status_code = 400


class ModelNotFoundError(InvalidRequestError):
    error_type = "model_not_found"


class NotFoundError(MyVistaError):
    error_type = "not_found"
    status_code = 404


class QuotaExceededError(MyVistaError):
    error_type = "quota_exceeded"
    status_code = 429


class APIConnectionError(MyVistaError):
    error_type = "api_connection_error"


class APITimeoutError(APIConnectionError):
    error_type = "api_timeout_error"


class APIStatusError(MyVistaError):
    error_type = "api_error"


class UnsupportedError(MyVistaError):
    """The method exists so the surface is complete; the backend is not built."""

    error_type = "unsupported"
    status_code = None


_BY_TYPE: dict[str, type[MyVistaError]] = {
    "authentication_error": AuthenticationError,
    "permission_error": AuthorizationError,
    "invalid_request_error": InvalidRequestError,
    "model_not_found": ModelNotFoundError,
    "not_found": NotFoundError,
    "quota_exceeded": QuotaExceededError,
    "no_candidate": APIStatusError,
    "all_candidates_failed": APIStatusError,
    "provider_timeout": APIStatusError,
    "provider_unavailable": APIStatusError,
    "upstream_error": APIStatusError,
    "configuration_error": APIStatusError,
}


def error_from_response(
    *,
    status_code: int,
    payload: dict[str, Any] | None,
    request_id: str | None,
    retry_after: str | None,
) -> MyVistaError:
    error = (payload or {}).get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        error = {}
    message = str(error.get("message") or f"HTTP {status_code}")
    error_type = str(error.get("type") or "")
    rid = error.get("request_id") or request_id
    details = error.get("details") if isinstance(error.get("details"), list) else None
    retry_after_s = None
    if retry_after:
        try:
            retry_after_s = int(retry_after)
        except ValueError:
            retry_after_s = None
    cls = _BY_TYPE.get(error_type, APIStatusError)
    return cls(
        message,
        status_code=status_code,
        request_id=str(rid) if rid else None,
        retry_after_s=retry_after_s,
        details=details,
    )
