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
        # Set by the router when a decision was reached so the gateway can meter
        # failures, not only successes.
        self.decision: object | None = None


class ConfigurationError(FabricError):
    """The fabric is misconfigured and cannot serve the request."""

    status_code = 500
    error_type = "configuration_error"


class AuthenticationError(FabricError):
    """The caller could not be identified. Never says why in detail."""

    status_code = 401
    error_type = "authentication_error"


class AuthorizationError(FabricError):
    """The caller is known but lacks the scope or role for this operation."""

    status_code = 403
    error_type = "permission_error"


class ResourceNotFoundError(FabricError):
    """The resource does not exist *for this tenant*.

    Cross-tenant reads are reported as absence rather than denial. A 403 would
    confirm that the resource exists in some other tenant, which is itself a
    leak; 404 reveals nothing.
    """

    status_code = 404
    error_type = "not_found"


class TenantIsolationError(FabricError):
    """A tenant boundary was crossed inside the process.

    This is raised by defence-in-depth checks that should be unreachable. It
    indicates a bug in the fabric, never a bad request, and it is deliberately
    not mapped onto a helpful client-facing message.
    """

    status_code = 500
    error_type = "internal_error"


class QuotaExceededError(FabricError):
    """A tenant or user quota was exhausted."""

    status_code = 429
    error_type = "quota_exceeded"

    def __init__(self, message: str, *, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ForbiddenRemediationError(AuthorizationError):
    """A self-healing action asked to mutate authorization or safety policy."""

    error_type = "forbidden_remediation"


class InvalidRequestError(FabricError):
    """The caller's request is malformed or asks for something unavailable."""

    status_code = 400
    error_type = "invalid_request_error"


class ModelNotFoundError(InvalidRequestError):
    error_type = "model_not_found"


class GuardrailBlockedError(InvalidRequestError):
    """A guardrail stage refused the request."""

    error_type = "guardrail_blocked"


class ContextTooLargeError(InvalidRequestError):
    """The prompt exceeds what the selected deployment can accept.

    A caller error by status, but a routable one: a larger deployment may serve
    the same request unchanged, which is why the router gives it its own
    fallback reason. The context compiler raises it when the blocks it may not
    drop do not fit on their own.
    """

    error_type = "context_too_large"


class DependencyUnavailableError(FabricError):
    """A mandatory serving dependency is known unavailable.

    This is an expected infrastructure-degraded condition, not an internal
    bug. Clients may retry after the dependency recovers. The status is 503
    rather than 500 so load balancers and callers can distinguish the two.
    """

    status_code = 503
    error_type = "dependency_unavailable"
    retryable = True


class NoCandidateError(FabricError):
    """No model in the registry satisfies the request and its policy."""

    status_code = 503
    error_type = "no_candidate"


class RetryableError(FabricError):
    """A backend failed in a way where another attempt may succeed."""

    status_code = 502
    error_type = "upstream_error"


class ProviderTimeoutError(RetryableError):
    error_type = "runtime_timeout"


class ProviderUnavailableError(RetryableError):
    error_type = "provider_unavailable"


class LiteLLMUnavailableError(ProviderUnavailableError):
    error_type = "litellm_unavailable"


class OllamaUnavailableError(ProviderUnavailableError):
    error_type = "ollama_unavailable"


class VllmUnavailableError(ProviderUnavailableError):
    error_type = "vllm_unavailable"


class RateLimitedError(RetryableError):
    status_code = 429
    error_type = "rate_limited"
    retryable = True

    def __init__(self, message: str, *, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ModelUnavailableError(RetryableError):
    error_type = "model_unavailable"


class AllCandidatesFailedError(FabricError):
    """Every candidate in the fallback chain was attempted and failed."""

    status_code = 502
    error_type = "route_exhausted"
