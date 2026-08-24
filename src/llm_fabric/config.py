"""Typed configuration, loaded from the environment.

Every setting is prefixed `LLM_FABRIC_`. Provider credentials also accept the
unprefixed names the providers themselves document (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`) so an existing environment works without renaming keys.
The prefixed form wins when both are set.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from llm_fabric.errors import ConfigurationError
from llm_fabric.identity.apikey import MIN_API_KEY_LENGTH, ApiCredential, fingerprint_key
from llm_fabric.identity.claims import ClaimsMapping
from llm_fabric.tenancy.quota import QuotaPolicy

AuthMode = Literal["disabled", "api_key", "dev", "oidc"]
EnvironmentName = Literal["development", "test", "production"]
VALID_ENVIRONMENTS: tuple[str, ...] = ("development", "test", "production")
ENVIRONMENT_REQUIRED = (
    "LLM_FABRIC_ENVIRONMENT is required; set it to development, test, or production"
)

#: Tenant assigned to credentials configured through the legacy flat
#: `LLM_FABRIC_API_KEYS` list, which carries no tenant of its own.
LEGACY_TENANT = "default"

# Production safety ceilings. Operators may raise them with explicit env vars.
# These bound runaway clients on a single-VPC internal deployment; they are
# not billing-grade and are not unlimited.
PRODUCTION_TENANT_RPM = 3_000  # 50 requests/s
PRODUCTION_TENANT_RPD = 500_000
PRODUCTION_TENANT_RPMONTH = 10_000_000
PRODUCTION_TENANT_TOKENS_PER_DAY = 50_000_000
PRODUCTION_TENANT_CONCURRENCY = 64
PRODUCTION_PROJECT_RPM = 3_000
PRODUCTION_PROVIDER_RPM = 6_000
PRODUCTION_MODEL_RPM = 3_000
PRODUCTION_USER_RPM = 1_200  # 20 requests/s
PRODUCTION_USER_RPD = 100_000
PRODUCTION_USER_TOKENS_PER_DAY = 10_000_000
PRODUCTION_USER_CONCURRENCY = 16
PRODUCTION_MAX_INPUT_TOKENS = 32_000
PRODUCTION_MAX_OUTPUT_TOKENS = 8_192
PRODUCTION_BREAKER_MAX_CONCURRENCY = 256


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_FABRIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 47317

    #: Deployment mode. There is no implicit default. Unset, empty, or an
    #: unknown value is a configuration error: the process must not quietly
    #: become development. Production is fail-closed.
    environment: EnvironmentName

    registry_path: Path = Path("config/models.yaml")

    # -- serving -------------------------------------------------------------

    #: OS processes serving the app. `None` means one worker in the foreground,
    #: which is the only shape that keeps the in-process quota ledger, health
    #: tracker and meter correct — see `docs/BENCHMARKS.md`. Raising this
    #: multiplies throughput and *also* multiplies every per-process limit, so
    #: it is deliberately not defaulted to the CPU count.
    workers: int | None = None

    #: Sockets accepted but not yet served. Bounds how much work can queue in
    #: the kernel before connections are refused outright, which is preferable
    #: to accepting work the process cannot reach before the client times out.
    backlog: int = 2048

    #: Seconds uvicorn waits for in-flight requests (including SSE) after
    #: SIGTERM. Keep below Kubernetes `terminationGracePeriodSeconds` so the
    #: process exits before SIGKILL. `None` would wait forever.
    graceful_shutdown_timeout_s: int = 25

    #: Requests one worker serves before it is replaced. Guards against slow
    #: leaks in long-lived processes. `None` disables recycling.
    max_requests_per_worker: int | None = None

    #: Required to start with `workers > 1`.
    #:
    #: Every piece of shared state in this build — the quota ledger, the health
    #: tracker and its circuit breakers, the usage meter, and all seven caches —
    #: lives in one process. A second worker does not share any of it, so each
    #: worker enforces the full quota independently and a tenant limited to 100
    #: requests per minute gets `100 * workers`. That is a correctness failure,
    #: not a tuning question, so it cannot be reached by setting a worker count
    #: alone. Measured throughput per worker is in `docs/BENCHMARKS.md`.
    allow_unsafe_multiworker: bool = False

    # -- authentication ------------------------------------------------------

    #: Left unset, the mode is inferred from whichever credentials are present.
    #: Inference is a development convenience. Production refuses to start
    #: unless a complete, usable identity source is configured — see
    #: `validate_startup`.
    auth_mode: AuthMode | None = None

    #: Explicit anonymous bypass. Default is on so local work and pytest stay
    #: usable. Production rejects this flag and refuses to start without
    #: authentication, even if it is set in a copied `.env`.
    allow_anonymous: bool = True

    #: Scopes every accepted token must carry, regardless of identity source.
    #: Empty means the gateway does not impose a global scope floor; routes
    #: still enforce their own `require_scopes` checks.
    required_scopes: Annotated[list[str], NoDecode] = Field(default_factory=list)

    #: Legacy flat key list. Every key lands in the `default` tenant, which is
    #: why it is only appropriate for single-tenant or local use. Use
    #: `api_credentials` to bind keys to real tenants.
    #:
    #: `NoDecode` on both list fields suppresses pydantic-settings' automatic
    #: JSON decoding so the validators below own parsing. Without it an empty
    #: value in a `.env` file raises a JSON decode error rather than meaning
    #: "unset", which makes `cp .env.example .env` fail on a fresh checkout.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    #: JSON list of `{"key", "tenant_id", "user_id", "roles", "scopes"}`.
    api_credentials: Annotated[list[ApiCredential], NoDecode] = Field(default_factory=list)

    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_uri: str | None = None
    oidc_jwks_cache_seconds: float = 300.0

    #: Enables the local development issuer. Must be at least 32 characters.
    dev_auth_secret: str | None = None

    claim_tenant: str = "tenant_id"
    claim_user: str = "user_id"
    claim_project: str = "project_id"
    claim_roles: str = "roles"
    claim_scopes: str = "scope"

    #: Tenant used when authentication is disabled entirely.
    anonymous_tenant: str = "public"

    # -- distributed state ---------------------------------------------------

    #: PostgreSQL URL. Unset keeps tenant records in process memory.
    #: Production refuses to start without it.
    database_url: str | None = None

    #: Redis/Valkey URL. Unset keeps quotas, breakers and revocation in process
    #: memory. Production refuses to start without it.
    redis_url: str | None = None

    #: Background dependency probes. `/readyz` and admission read cached
    #: state; these bound how stale that state may be. Worst-case probe-only
    #: detection is `fail_threshold * (interval + timeout)` seconds. A
    #: serving-path failure marks the dependency unhealthy immediately.
    health_probe_interval_s: float = 2.0
    health_probe_timeout_s: float = 1.0
    health_fail_threshold: int = 2
    health_recovery_threshold: int = 2

    #: Analytical store DSN (ClickHouse or compatible). Never used on the
    #: synchronous request path. Unset means analytics events are discarded.
    analytics_url: str | None = None

    # -- quotas --------------------------------------------------------------

    quota_tenant_requests_per_minute: int | None = None
    quota_tenant_requests_per_day: int | None = None
    quota_tenant_tokens_per_day: int | None = None
    quota_tenant_cost_per_day_usd: float | None = None
    quota_tenant_requests_per_month: int | None = None
    quota_tenant_max_concurrency: int | None = None
    quota_tenant_project_requests_per_minute: int | None = None
    quota_tenant_provider_requests_per_minute: int | None = None
    quota_tenant_model_requests_per_minute: int | None = None

    quota_user_requests_per_minute: int | None = None
    quota_user_requests_per_day: int | None = None
    quota_user_tokens_per_day: int | None = None
    quota_user_cost_per_day_usd: float | None = None
    quota_user_max_concurrency: int | None = None

    # -- inference -----------------------------------------------------------

    # Per-attempt ceiling, not a total request budget.
    request_timeout_s: float = 60.0

    # Total attempts across the whole fallback chain, including the first.
    max_attempts: int = 3

    #: Canonical policy names are the constitution's (`cost_first`,
    #: `quality_first`, ...). The older `cheapest` still parses.
    default_policy: str = "cost_first"
    log_level: str = "INFO"

    # -- routing reliability -------------------------------------------------

    breaker_consecutive_failures: int = 5
    breaker_error_rate: float = 0.5
    breaker_minimum_samples: int = 10
    breaker_open_duration_s: float = 30.0
    breaker_half_open_successes: int = 2

    #: Concurrent in-flight requests one deployment may hold. Unset means
    #: unlimited in development/test. Production fills a finite default.
    breaker_max_concurrency: int | None = None

    #: Prompt and completion token ceilings. Production fills finite defaults.
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    #: Test-only: slow the mock provider so streaming/SIGTERM can be observed.
    mock_delay_s: float = 0.0

    #: Ceilings on what falling back may consume, beyond the attempt limit.
    #: Unset means the attempt limit is the only bound.
    fallback_max_cost_usd: float | None = None
    fallback_max_latency_ms: float | None = None

    # -- intent --------------------------------------------------------------

    #: Classify every chat request and route on the result.
    #:
    #: Production refuses to start when this is false. Development and test may
    #: disable the cascade for focused testing; they still attach a SAFE_FALLBACK
    #: IntentResult so provider invocations are never intent-less.
    intent_classification_enabled: bool = False
    #: Classify on the serving path but do not change the route. Recorded on
    #: `x-fabric-intent-shadow-*` headers. Ignored when classification is on.
    intent_shadow: bool = False
    #: hashing (deterministic tests) | local/bge-small | minilm
    intent_embedder: str = "hashing"
    #: Production refuses HashingEmbedder unless this is explicitly true.
    intent_allow_hashing_embedder: bool = False
    #: Attach the local description reranker as L4. Off by default. L5 stays off.
    intent_l4_rerank: bool = False

    #: Compare the live route against a quality_first ranking without changing
    #: the served deployment. Off by default.
    routing_quality_shadow: bool = False

    # -- observability -------------------------------------------------------

    #: OTLP HTTP traces endpoint. Unset means spans stay in-process for the
    #: Command Center and are never exported.
    otel_exporter_otlp_endpoint: str | None = None
    #: Optional `k=v,k2=v2` header string. Keep this in a Secret, not a ConfigMap.
    otel_exporter_otlp_headers: str | None = None
    #: Optional CA file for TLS to the collector.
    otel_exporter_otlp_certificate: str | None = None

    #: Langfuse public ingestion. All three must be set or the adapter is a no-op.
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    #: Maximum JSON/body size the gateway will accept. Oversized requests are
    #: rejected before routing. Guardrails apply a matching bound to text.
    max_request_bytes: int = 1_048_576

    #: Origins allowed for browser CORS. Empty means CORS is not enabled.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    #: Peer addresses allowed to set `X-Forwarded-For` / `X-Forwarded-Proto`.
    #: Empty means those headers are ignored.
    trusted_proxies: Annotated[list[str], NoDecode] = Field(default_factory=list)

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_FABRIC_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_FABRIC_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    #: Local Ollama OpenAI-compatible endpoint. No OpenAI key is required.
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_api_key: str | None = None

    #: Production vLLM OpenAI-compatible endpoint. No OpenAI key is required.
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_api_key: str | None = None

    #: LiteLLM proxy. Transport only; MyVista still selects the model name.
    litellm_base_url: str = "http://127.0.0.1:4000/v1"
    litellm_api_key: str | None = None
    #: Expected LiteLLM-side num_retries. Used to refuse retry amplification.
    #: The LiteLLM config itself must set num_retries to this value.
    litellm_num_retries: int = 0

    #: Intent → capability → preferred-tier policy. Missing file loads empty.
    routing_config_path: Path = Path("config/routing.yaml")

    #: Promotion policy. Missing file loads empty (no extra gates).
    promotion_config_path: Path = Path("config/promotion.yaml")
    #: Overlay of evidence-bound lifecycle. Does not rewrite models.yaml comments.
    promotion_state_path: Path = Path("datasets/eval/models/promotion-state.json")

    #: Per-provider OpenAI-compatible base URLs, JSON object.
    #: Example: `{"vllm-coding":"http://vllm-coding:8000/v1"}`.
    provider_base_urls: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)

    @field_validator(
        "api_keys", "required_scopes", "cors_origins", "trusted_proxies", mode="before"
    )
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return [key.strip() for key in value.replace(",", " ").split() if key.strip()]
        return value

    @field_validator("environment", mode="before")
    @classmethod
    def _require_explicit_environment(cls, value: object) -> object:
        """Refuse missing, blank, or unknown deployment modes.

        An empty `LLM_FABRIC_ENVIRONMENT=` is not development. It is an error.
        """
        if value is None:
            raise ValueError(ENVIRONMENT_REQUIRED)
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                raise ValueError(ENVIRONMENT_REQUIRED)
            if text not in VALID_ENVIRONMENTS:
                raise ValueError(
                    f"LLM_FABRIC_ENVIRONMENT={value!r} is not valid; "
                    "set it to development, test, or production"
                )
            return text
        return value

    @field_validator("api_credentials", mode="before")
    @classmethod
    def _parse_credentials(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("LLM_FABRIC_API_CREDENTIALS must be valid JSON") from exc
        return value

    @field_validator("provider_base_urls", mode="before")
    @classmethod
    def _parse_provider_base_urls(cls, value: object) -> object:
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("LLM_FABRIC_PROVIDER_BASE_URLS must be a JSON object") from exc
            if not isinstance(parsed, dict):
                raise ValueError("LLM_FABRIC_PROVIDER_BASE_URLS must be a JSON object")
            return {str(key): str(item) for key, item in parsed.items()}
        return value

    @field_validator(
        "auth_mode",
        "oidc_issuer",
        "oidc_audience",
        "oidc_jwks_uri",
        "dev_auth_secret",
        "database_url",
        "redis_url",
        "analytics_url",
        "quota_tenant_requests_per_minute",
        "quota_tenant_requests_per_day",
        "quota_tenant_tokens_per_day",
        "quota_tenant_cost_per_day_usd",
        "quota_tenant_requests_per_month",
        "quota_tenant_max_concurrency",
        "quota_tenant_project_requests_per_minute",
        "quota_tenant_provider_requests_per_minute",
        "quota_tenant_model_requests_per_minute",
        "quota_user_requests_per_minute",
        "quota_user_requests_per_day",
        "quota_user_tokens_per_day",
        "quota_user_cost_per_day_usd",
        "quota_user_max_concurrency",
        "breaker_max_concurrency",
        "max_input_tokens",
        "max_output_tokens",
        "fallback_max_cost_usd",
        "fallback_max_latency_ms",
        "workers",
        "max_requests_per_worker",
        "openai_api_key",
        "anthropic_api_key",
        "ollama_api_key",
        "vllm_api_key",
        "otel_exporter_otlp_endpoint",
        "otel_exporter_otlp_headers",
        "otel_exporter_otlp_certificate",
        "langfuse_host",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat an empty environment variable as absent.

        `NAME=` in a `.env` file is how people write "I have not set this". For
        an optional field it must mean `None`, not an empty string that fails to
        parse as an integer.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def __init__(self, **values: Any) -> None:
        try:
            super().__init__(**values)
        except ValidationError as exc:
            raise ConfigurationError(_settings_error_message(exc)) from None

    # -- derived -------------------------------------------------------------

    @property
    def effective_auth_mode(self) -> AuthMode:
        if self.auth_mode is not None:
            return self.auth_mode
        if self.oidc_issuer:
            return "oidc"
        if self.dev_auth_secret:
            return "dev"
        if self.api_credentials or self.api_keys:
            return "api_key"
        return "disabled"

    @property
    def auth_required(self) -> bool:
        """Whether callers must present a credential.

        Production is always authenticated. Development and test may run
        anonymously only when the mode is `disabled` *and* `allow_anonymous`
        is set — the environment name itself is the explicit choice of a
        non-production mode.
        """
        if self.environment == "production":
            return True
        if self.allow_anonymous and self.effective_auth_mode == "disabled":
            return False
        return self.effective_auth_mode != "disabled"

    @property
    def claims_mapping(self) -> ClaimsMapping:
        return ClaimsMapping(
            tenant=self.claim_tenant,
            user=self.claim_user,
            project=self.claim_project,
            roles=self.claim_roles,
            scopes=self.claim_scopes,
        )

    def resolved_credentials(self) -> list[ApiCredential]:
        """Explicit credentials plus any legacy flat keys."""
        credentials = list(self.api_credentials)
        for key in self.api_keys:
            try:
                credentials.append(
                    ApiCredential(
                        key=key,
                        tenant_id=LEGACY_TENANT,
                        # Distinct users inside one tenant, so per-user quotas
                        # still separate two legacy keys sharing a tenant.
                        user_id=f"legacy-{fingerprint_key(key)}",
                    )
                )
            except ValidationError as exc:
                raise ConfigurationError(
                    f"LLM_FABRIC_API_KEYS contains a key shorter than {MIN_API_KEY_LENGTH} "
                    "characters; short keys are guessable and are refused"
                ) from exc
        return credentials

    @property
    def tenant_quota_policy(self) -> QuotaPolicy:
        if self.environment == "production":
            return QuotaPolicy(
                requests_per_minute=self.quota_tenant_requests_per_minute
                if self.quota_tenant_requests_per_minute is not None
                else PRODUCTION_TENANT_RPM,
                requests_per_day=self.quota_tenant_requests_per_day
                if self.quota_tenant_requests_per_day is not None
                else PRODUCTION_TENANT_RPD,
                requests_per_month=self.quota_tenant_requests_per_month
                if self.quota_tenant_requests_per_month is not None
                else PRODUCTION_TENANT_RPMONTH,
                tokens_per_day=self.quota_tenant_tokens_per_day
                if self.quota_tenant_tokens_per_day is not None
                else PRODUCTION_TENANT_TOKENS_PER_DAY,
                cost_per_day_usd=self.quota_tenant_cost_per_day_usd,
                max_concurrency=self.quota_tenant_max_concurrency
                if self.quota_tenant_max_concurrency is not None
                else PRODUCTION_TENANT_CONCURRENCY,
                project_requests_per_minute=self.quota_tenant_project_requests_per_minute
                if self.quota_tenant_project_requests_per_minute is not None
                else PRODUCTION_PROJECT_RPM,
                provider_requests_per_minute=self.quota_tenant_provider_requests_per_minute
                if self.quota_tenant_provider_requests_per_minute is not None
                else PRODUCTION_PROVIDER_RPM,
                model_requests_per_minute=self.quota_tenant_model_requests_per_minute
                if self.quota_tenant_model_requests_per_minute is not None
                else PRODUCTION_MODEL_RPM,
            )
        return QuotaPolicy(
            requests_per_minute=self.quota_tenant_requests_per_minute,
            requests_per_day=self.quota_tenant_requests_per_day,
            requests_per_month=self.quota_tenant_requests_per_month,
            tokens_per_day=self.quota_tenant_tokens_per_day,
            cost_per_day_usd=self.quota_tenant_cost_per_day_usd,
            max_concurrency=self.quota_tenant_max_concurrency,
            project_requests_per_minute=self.quota_tenant_project_requests_per_minute,
            provider_requests_per_minute=self.quota_tenant_provider_requests_per_minute,
            model_requests_per_minute=self.quota_tenant_model_requests_per_minute,
        )

    @property
    def user_quota_policy(self) -> QuotaPolicy:
        if self.environment == "production":
            return QuotaPolicy(
                requests_per_minute=self.quota_user_requests_per_minute
                if self.quota_user_requests_per_minute is not None
                else PRODUCTION_USER_RPM,
                requests_per_day=self.quota_user_requests_per_day
                if self.quota_user_requests_per_day is not None
                else PRODUCTION_USER_RPD,
                tokens_per_day=self.quota_user_tokens_per_day
                if self.quota_user_tokens_per_day is not None
                else PRODUCTION_USER_TOKENS_PER_DAY,
                cost_per_day_usd=self.quota_user_cost_per_day_usd,
                max_concurrency=self.quota_user_max_concurrency
                if self.quota_user_max_concurrency is not None
                else PRODUCTION_USER_CONCURRENCY,
            )
        return QuotaPolicy(
            requests_per_minute=self.quota_user_requests_per_minute,
            requests_per_day=self.quota_user_requests_per_day,
            tokens_per_day=self.quota_user_tokens_per_day,
            cost_per_day_usd=self.quota_user_cost_per_day_usd,
            max_concurrency=self.quota_user_max_concurrency,
        )

    @property
    def effective_max_input_tokens(self) -> int:
        if self.max_input_tokens is not None:
            return self.max_input_tokens
        if self.environment == "production":
            return PRODUCTION_MAX_INPUT_TOKENS
        return PRODUCTION_MAX_INPUT_TOKENS

    @property
    def effective_max_output_tokens(self) -> int:
        if self.max_output_tokens is not None:
            return self.max_output_tokens
        return PRODUCTION_MAX_OUTPUT_TOKENS

    @property
    def effective_breaker_max_concurrency(self) -> int | None:
        if self.breaker_max_concurrency is not None:
            return self.breaker_max_concurrency
        if self.environment == "production":
            return PRODUCTION_BREAKER_MAX_CONCURRENCY
        return None

    def describe_auth(self) -> dict[str, Any]:
        """Startup-log summary. Contains no secret material."""
        mode = self.effective_auth_mode
        return {
            "environment": self.environment,
            "auth_mode": mode,
            "auth_required": self.auth_required,
            "allow_anonymous": self.allow_anonymous,
            "configured_credentials": len(self.resolved_credentials()),
            "oidc_issuer": self.oidc_issuer,
            "legacy_keys": len(self.api_keys),
            "required_scopes": list(self.required_scopes),
        }

    @property
    def health_detection_bound_s(self) -> float:
        """Worst-case seconds until a probe-only outage is UNHEALTHY."""
        return self.health_fail_threshold * (
            self.health_probe_interval_s + self.health_probe_timeout_s
        )

    @property
    def health_recovery_bound_s(self) -> float:
        """Worst-case seconds until a restored dependency is HEALTHY again."""
        return self.health_recovery_threshold * (
            self.health_probe_interval_s + self.health_probe_timeout_s
        )


def validate_startup(settings: Settings) -> None:
    """Refuse to start when authentication cannot be enforced as configured.

    Called from `create_app` and `python -m llm_fabric` *before* the server
    binds a port. Production has no silent-open path: missing, incomplete,
    contradictory or development-only identity configuration is a startup
    failure, not a warning.
    """
    from llm_fabric.serving.topology import refuse_retry_amplification

    refuse_retry_amplification(
        fabric_attempts=settings.max_attempts,
        transport_retries=settings.litellm_num_retries,
    )
    mode = settings.effective_auth_mode

    if settings.environment == "production":
        _validate_production_auth(settings, mode)
        return

    if mode == "disabled" and not settings.allow_anonymous:
        raise ConfigurationError(
            "authentication is disabled but LLM_FABRIC_ALLOW_ANONYMOUS is false; "
            "configure an identity source or explicitly allow anonymous access"
        )
    _validate_identity_source_is_complete(settings, mode)


def _validate_production_auth(settings: Settings, mode: AuthMode) -> None:
    if settings.allow_anonymous:
        raise ConfigurationError(
            "production refused to start: LLM_FABRIC_ALLOW_ANONYMOUS is enabled; "
            "authentication bypass is a development affordance"
        )
    if settings.dev_auth_secret:
        raise ConfigurationError(
            "production refused to start: LLM_FABRIC_DEV_AUTH_SECRET must not be set"
        )
    if mode == "dev":
        raise ConfigurationError(
            "production refused to start: the development identity provider is forbidden"
        )
    if mode == "disabled":
        raise ConfigurationError(
            "production refused to start without authentication: configure "
            "OIDC (LLM_FABRIC_OIDC_ISSUER and LLM_FABRIC_OIDC_AUDIENCE) or "
            "API credentials"
        )
    _validate_identity_source_is_complete(settings, mode)
    if settings.allow_unsafe_multiworker:
        raise ConfigurationError(
            "production refused to start: LLM_FABRIC_ALLOW_UNSAFE_MULTIWORKER "
            "is a development escape hatch"
        )
    if any(origin.strip() == "*" for origin in settings.cors_origins):
        raise ConfigurationError(
            "production refused to start: LLM_FABRIC_CORS_ORIGINS must not include '*'; "
            "wildcard CORS is a development affordance"
        )
    if not settings.database_url:
        raise ConfigurationError("production refused to start: LLM_FABRIC_DATABASE_URL is required")
    if settings.database_url.startswith("sqlite"):
        raise ConfigurationError(
            "production refused to start: sqlite is not an acceptable durable store"
        )
    if not settings.redis_url:
        raise ConfigurationError("production refused to start: LLM_FABRIC_REDIS_URL is required")
    if not settings.intent_classification_enabled:
        raise ConfigurationError(
            "production refused to start: serving-path IntentOS is mandatory; "
            "set LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED=true"
        )
    embedder = (settings.intent_embedder or "hashing").strip().lower()
    if embedder in {"hashing", "hash", "lexical"} and not settings.intent_allow_hashing_embedder:
        raise ConfigurationError(
            "production refused to start: HashingEmbedder is not a semantic model; "
            "set LLM_FABRIC_INTENT_EMBEDDER to local or minilm, or set "
            "LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER=true to accept lexical hashing"
        )
    if settings.workers and settings.workers > 1 and not settings.redis_url:
        raise ConfigurationError(
            "production refused to start with multiple workers: quotas, "
            "circuit breakers and the revocation denylist are still per process"
        )
    _validate_identity_source_is_complete(settings, mode)
    if not settings.auth_required:
        raise ConfigurationError("production refused to start without authentication")


def _validate_identity_source_is_complete(settings: Settings, mode: AuthMode) -> None:
    if mode == "oidc" and (not settings.oidc_issuer or not settings.oidc_audience):
        raise ConfigurationError(
            "OIDC authentication is incomplete: LLM_FABRIC_OIDC_ISSUER and "
            "LLM_FABRIC_OIDC_AUDIENCE are both required"
        )
    if mode == "api_key" and not settings.resolved_credentials():
        raise ConfigurationError("auth_mode is 'api_key' but no API credentials are configured")
    if mode == "dev" and not settings.dev_auth_secret:
        raise ConfigurationError("auth_mode is 'dev' but LLM_FABRIC_DEV_AUTH_SECRET is unset")


def _settings_error_message(exc: ValidationError) -> str:
    """Turn a pydantic settings failure into an operator-facing sentence."""
    for error in exc.errors():
        loc = error.get("loc") or ()
        if loc and loc[0] == "environment":
            message = str(error.get("msg") or "")
            if message.startswith("Value error, "):
                return message.removeprefix("Value error, ")
            if error.get("type") == "missing":
                return ENVIRONMENT_REQUIRED
            return message or ENVIRONMENT_REQUIRED
    return str(exc)


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ConfigurationError:
        raise
    except ValidationError as exc:
        raise ConfigurationError(_settings_error_message(exc)) from exc
