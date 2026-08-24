from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from llm_fabric.config import Settings, validate_startup
from llm_fabric.errors import ConfigurationError
from llm_fabric.identity.apikey import ApiCredential

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_reads_prefixed_provider_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_FABRIC_OPENAI_API_KEY", "prefixed")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings(_env_file=None).openai_api_key == "prefixed"


def test_reads_unprefixed_provider_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_FABRIC_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unprefixed")
    assert Settings(_env_file=None).openai_api_key == "unprefixed"


def test_prefixed_provider_key_wins(monkeypatch) -> None:
    monkeypatch.setenv("LLM_FABRIC_OPENAI_API_KEY", "prefixed")
    monkeypatch.setenv("OPENAI_API_KEY", "unprefixed")
    assert Settings(_env_file=None).openai_api_key == "prefixed"


def test_reads_unprefixed_anthropic_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_FABRIC_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    assert Settings(_env_file=None).anthropic_api_key == "claude-key"


# -- the shipped example file ------------------------------------------------


def test_the_example_env_file_loads_as_written(tmp_path: Path, monkeypatch) -> None:
    """`cp .env.example .env` must work on a fresh checkout.

    The example sets optional values to an empty string, which is how people
    write "unset". Before this was handled, that produced a JSON decode error on
    the list fields and a type error on the numeric ones.
    """
    for name in [key for key in os.environ if key.startswith("LLM_FABRIC_")]:
        monkeypatch.delenv(name, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text((REPO_ROOT / ".env.example").read_text(), encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.effective_auth_mode == "disabled"
    assert settings.environment == "development"
    assert settings.allow_anonymous is True
    assert settings.api_keys == []
    assert settings.api_credentials == []
    assert settings.quota_tenant_requests_per_minute is None
    assert settings.oidc_issuer is None
    assert settings.tenant_quota_policy.is_unlimited
    validate_startup(settings)


def test_production_fills_finite_quota_ceilings() -> None:
    from llm_fabric.config import (
        PRODUCTION_TENANT_CONCURRENCY,
        PRODUCTION_TENANT_RPM,
        PRODUCTION_USER_RPM,
        Settings,
    )

    settings = Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        api_keys=["a-long-enough-api-key"],
        database_url="postgresql://fabric_app:fabric@127.0.0.1:5432/fabric",
        redis_url="redis://127.0.0.1:6379/0",
    )
    assert settings.tenant_quota_policy.requests_per_minute == PRODUCTION_TENANT_RPM
    assert settings.tenant_quota_policy.max_concurrency == PRODUCTION_TENANT_CONCURRENCY
    assert settings.user_quota_policy.requests_per_minute == PRODUCTION_USER_RPM
    assert not settings.tenant_quota_policy.is_unlimited
    assert settings.effective_max_input_tokens == 32_000
    assert settings.effective_max_output_tokens == 8_192
    assert settings.effective_breaker_max_concurrency == 256


def test_production_explicit_quota_wins_over_default() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        api_keys=["a-long-enough-api-key"],
        database_url="postgresql://fabric_app:fabric@127.0.0.1:5432/fabric",
        redis_url="redis://127.0.0.1:6379/0",
        quota_tenant_requests_per_minute=12,
        quota_tenant_max_concurrency=3,
    )
    assert settings.tenant_quota_policy.requests_per_minute == 12
    assert settings.tenant_quota_policy.max_concurrency == 3


# -- authentication mode inference -------------------------------------------


def test_mode_is_disabled_without_credentials() -> None:
    assert Settings(_env_file=None, api_keys=[]).effective_auth_mode == "disabled"


def test_mode_is_inferred_from_whichever_credential_is_present() -> None:
    assert Settings(_env_file=None, api_keys=["a-long-enough-key"]).effective_auth_mode == (
        "api_key"
    )
    assert Settings(_env_file=None, dev_auth_secret="x" * 40).effective_auth_mode == "dev"
    assert Settings(_env_file=None, oidc_issuer="https://i.example").effective_auth_mode == ("oidc")


def test_an_explicit_mode_overrides_inference() -> None:
    """Inference is a convenience; the explicit setting is the safety mechanism."""
    settings = Settings(_env_file=None, auth_mode="oidc", api_keys=["a-long-enough-key"])

    assert settings.effective_auth_mode == "oidc"
    # Development with an explicit mode still requires a credential; the
    # issuer/audience completeness check is `validate_startup`.
    assert settings.auth_required is True


def test_oidc_takes_precedence_over_a_stray_dev_secret() -> None:
    settings = Settings(_env_file=None, oidc_issuer="https://i.example", dev_auth_secret="x" * 40)

    assert settings.effective_auth_mode == "oidc"


# -- credentials -------------------------------------------------------------


def test_credentials_parse_from_json() -> None:
    payload = json.dumps(
        [{"key": "a-long-enough-api-key", "tenant_id": "acme", "scopes": ["chat:write"]}]
    )

    settings = Settings(_env_file=None, api_credentials=payload)

    assert settings.api_credentials == (
        [ApiCredential(key="a-long-enough-api-key", tenant_id="acme", scopes=("chat:write",))]
    )


def test_malformed_credential_json_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="valid JSON"):
        Settings(_env_file=None, api_credentials="{not json")


def test_legacy_keys_become_distinct_users_in_one_tenant() -> None:
    """Two legacy keys share a tenant but must not share a per-user quota."""
    settings = Settings(
        _env_file=None, api_keys=["first-key-long-enough", "second-key-long-enough"]
    )

    credentials = settings.resolved_credentials()

    assert {c.tenant_id for c in credentials} == {"default"}
    assert len({c.user_id for c in credentials}) == 2


def test_the_auth_summary_carries_no_secret_material() -> None:
    settings = Settings(_env_file=None, api_keys=["a-very-secret-api-key"])

    summary = json.dumps(settings.describe_auth())

    assert "a-very-secret-api-key" not in summary
    assert summary.count("configured_credentials") == 1
    assert '"environment"' in summary


def test_blank_environment_is_refused_not_coerced_to_development() -> None:
    with pytest.raises(ConfigurationError, match="LLM_FABRIC_ENVIRONMENT is required"):
        Settings(_env_file=None, environment="")  # type: ignore[arg-type]


def test_missing_environment_variable_is_refused(monkeypatch) -> None:
    monkeypatch.delenv("LLM_FABRIC_ENVIRONMENT", raising=False)
    with pytest.raises(ConfigurationError, match="LLM_FABRIC_ENVIRONMENT is required"):
        Settings(_env_file=None)


def test_required_scopes_parse_from_a_space_delimited_string() -> None:
    settings = Settings(_env_file=None, required_scopes="chat:write fabric:observe")
    assert settings.required_scopes == ["chat:write", "fabric:observe"]


def test_production_auth_required_cannot_be_switched_off() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        oidc_issuer="https://issuer.example",
        oidc_audience="aud",
        database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
        redis_url="redis://127.0.0.1:6379/0",
    )
    assert settings.auth_required is True


def test_incomplete_oidc_is_refused_in_every_environment() -> None:
    with pytest.raises(ConfigurationError, match="incomplete"):
        validate_startup(
            Settings(_env_file=None, auth_mode="oidc", oidc_issuer="https://issuer.example")
        )
