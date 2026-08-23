"""Error taxonomy for the fabric.

Errors are split by whether retrying or failing over to another backend could
plausibly succeed. The router relies on that distinction: a `RetryableError`
means try the next candidate, anything else means stop and report.
"""

from __future__ import annotations


class FabricError(Exception):
    """Base class for every error the fabric raises deliberately."""

    status_code: int = 500
    error_type: str = "fabric_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(FabricError):
    """The fabric is misconfigured and cannot serve the request."""

    status_code = 500
    error_type = "configuration_error"


class AuthenticationError(FabricError):
    status_code = 401
    error_type = "authentication_error"


class InvalidRequestError(FabricError):
    """The caller's request is malformed or asks for something unavailable."""

    status_code = 400
    error_type = "invalid_request_error"


class ModelNotFoundError(InvalidRequestError):
    error_type = "model_not_found"


class NoCandidateError(FabricError):
    """No model in the registry satisfies the request and its policy."""

    status_code = 503
    error_type = "no_candidate"


class RetryableError(FabricError):
    """A backend failed in a way where another attempt may succeed."""

    status_code = 502
    error_type = "upstream_error"


class ProviderTimeoutError(RetryableError):
    error_type = "provider_timeout"


class ProviderUnavailableError(RetryableError):
    error_type = "provider_unavailable"


class AllCandidatesFailedError(FabricError):
    """Every candidate in the fallback chain was attempted and failed."""

    status_code = 502
    error_type = "all_candidates_failed"
