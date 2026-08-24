"""Production authentication is fail-closed at process start.

These tests assert the gateway refuses to assemble when production identity
is missing, incomplete, or a development escape hatch. A warning is not a
refusal: the process must not bind a port.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_fabric.config import Settings, get_settings, validate_startup
from llm_fabric.errors import ConfigurationError
from llm_fabric.gateway.app import create_app
from llm_fabric.identity.oidc import OIDCTokenVerifier
from llm_fabric.identity.revocation import InMemoryRevocationStore
from llm_fabric.router.health import HealthTracker
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.quota import QuotaLedger


def _development(**overrides: object) -> Settings:
    values: dict[str, object] = {"environment": "development", "api_keys": []}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def _production(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "allow_anonymous": False,
        "api_keys": [],
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture
def without_live_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config-only tests of a production app must not require Postgres/Redis."""
    monkeypatch.setattr("llm_fabric.runtime.probe_distributed_state", lambda _settings: None)


def test_unset_environment_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_FABRIC_ENVIRONMENT", raising=False)
    with pytest.raises(ConfigurationError, match="LLM_FABRIC_ENVIRONMENT is required"):
        Settings(_env_file=None)


def test_blank_environment_refuses_to_start() -> None:
    with pytest.raises(ConfigurationError, match="LLM_FABRIC_ENVIRONMENT is required"):
        Settings(_env_file=None, environment="")  # type: ignore[arg-type]


def test_invalid_environment_refuses_to_start() -> None:
    with pytest.raises(ConfigurationError, match="not valid"):
        Settings(_env_file=None, environment="staging")  # type: ignore[arg-type]


def test_create_app_cannot_bypass_a_missing_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_FABRIC_ENVIRONMENT", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError, match="LLM_FABRIC_ENVIRONMENT is required"):
        create_app()
    get_settings.cache_clear()


def test_development_is_an_explicit_choice() -> None:
    settings = _development()
    assert settings.environment == "development"
    validate_startup(settings)
    assert settings.auth_required is False


def test_test_environment_is_not_production_fail_closed() -> None:
    settings = Settings(_env_file=None, environment="test", api_keys=[], allow_anonymous=True)
    validate_startup(settings)
    assert settings.environment == "test"
    assert settings.auth_required is False


def test_development_may_start_anonymously_when_explicitly_allowed() -> None:
    settings = _development(allow_anonymous=True)
    validate_startup(settings)
    assert settings.auth_required is False


def test_development_refuses_anonymous_access_when_it_is_not_allowed() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_ANONYMOUS"):
        validate_startup(_development(allow_anonymous=False))


def test_production_refuses_to_start_without_authentication() -> None:
    with pytest.raises(ConfigurationError, match="without authentication"):
        validate_startup(_production())


def test_production_refuses_an_explicit_disabled_mode() -> None:
    with pytest.raises(ConfigurationError, match="without authentication"):
        validate_startup(_production(auth_mode="disabled"))


def test_production_refuses_anonymous_bypass_even_with_oidc() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_ANONYMOUS"):
        validate_startup(
            _production(
                allow_anonymous=True,
                oidc_issuer="https://issuer.example",
                oidc_audience="myvista-llm-fabric",
            )
        )


def test_production_refuses_the_development_issuer() -> None:
    with pytest.raises(ConfigurationError, match="DEV_AUTH_SECRET"):
        validate_startup(_production(dev_auth_secret="x" * 40, allow_anonymous=False))


def test_production_refuses_incomplete_oidc() -> None:
    with pytest.raises(ConfigurationError, match="incomplete"):
        validate_startup(_production(oidc_issuer="https://issuer.example"))


def test_production_refuses_api_key_mode_without_credentials() -> None:
    with pytest.raises(ConfigurationError, match="no API credentials"):
        validate_startup(_production(auth_mode="api_key"))


def test_production_refuses_the_unsafe_multiworker_escape() -> None:
    with pytest.raises(ConfigurationError, match="UNSAFE_MULTIWORKER"):
        validate_startup(
            _production(
                oidc_issuer="https://issuer.example",
                oidc_audience="myvista",
                allow_unsafe_multiworker=True,
            )
        )


def test_production_refuses_wildcard_cors() -> None:
    with pytest.raises(ConfigurationError, match="CORS"):
        validate_startup(
            _production(
                oidc_issuer="https://issuer.example",
                oidc_audience="myvista",
                database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
                redis_url="redis://127.0.0.1:6379/0",
                cors_origins=["*"],
            )
        )


def test_production_accepts_complete_oidc_configuration() -> None:
    validate_startup(
        _production(
            oidc_issuer="https://issuer.example",
            oidc_audience="myvista-llm-fabric",
            database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
            redis_url="redis://127.0.0.1:6379/0",
        )
    )


def test_production_accepts_configured_api_keys() -> None:
    validate_startup(
        _production(
            api_keys=["a-long-enough-api-key"],
            database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
            redis_url="redis://127.0.0.1:6379/0",
        )
    )


def test_production_does_not_mount_openapi_docs(without_live_dependencies: None) -> None:
    stores = TenantStores()
    app = create_app(
        settings=_production(
            api_keys=["a-long-enough-api-key"],
            database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
            redis_url="redis://127.0.0.1:6379/0",
        ),
        stores=stores,
        cache=TenantScopedCache(audit=stores.audit),
        quota=QuotaLedger(),
        health_tracker=HealthTracker(),
        revocation_store=InMemoryRevocationStore(),
    )
    assert app.docs_url is None
    assert app.openapi_url is None


def test_development_mounts_openapi_docs() -> None:
    app = create_app(settings=_development())
    assert app.docs_url == "/docs"
    assert app.openapi_url == "/openapi.json"


def test_production_refuses_missing_database_url() -> None:
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        validate_startup(
            _production(
                oidc_issuer="https://issuer.example",
                oidc_audience="myvista",
                redis_url="redis://127.0.0.1:6379/0",
            )
        )


def test_production_refuses_sqlite_as_durable_store() -> None:
    with pytest.raises(ConfigurationError, match="sqlite"):
        validate_startup(
            _production(
                oidc_issuer="https://issuer.example",
                oidc_audience="myvista",
                database_url="sqlite:///tmp/fabric.db",
                redis_url="redis://127.0.0.1:6379/0",
            )
        )


def test_create_app_refuses_production_without_authentication() -> None:
    with pytest.raises(ConfigurationError, match="without authentication|ALLOW_ANONYMOUS"):
        create_app(settings=_production())


def test_create_app_never_emits_the_open_gateway_warning_in_production(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    with pytest.raises(ConfigurationError):
        create_app(settings=_production(allow_anonymous=True))
    assert "gateway is running without authentication" not in caplog.text


def test_development_warns_when_running_without_authentication(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Depending on capture order the JSON line lands in caplog or stdout.
    with caplog.at_level(logging.WARNING, logger="llm_fabric"):
        app = create_app(settings=_development())
        with TestClient(app):
            pass
    recorded = caplog.text + capsys.readouterr().out
    assert "gateway is running without authentication" in recorded


def test_production_refuses_to_become_ready_when_oidc_jwks_is_unreachable(
    without_live_dependencies: None,
) -> None:
    def unreachable(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider is down")

    verifier = OIDCTokenVerifier(
        issuer="https://issuer.example",
        audience="myvista-llm-fabric",
        jwks_uri="https://issuer.example/jwks",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(unreachable)),
    )
    settings = _production(
        oidc_issuer="https://issuer.example",
        oidc_audience="myvista-llm-fabric",
        database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
        redis_url="redis://127.0.0.1:6379/0",
    )
    stores = TenantStores()
    app = create_app(
        settings=settings,
        verifier=verifier,
        stores=stores,
        cache=TenantScopedCache(audit=stores.audit),
        quota=QuotaLedger(),
        health_tracker=HealthTracker(),
        revocation_store=InMemoryRevocationStore(),
    )

    with pytest.raises(ConfigurationError, match="unreachable|no usable keys"), TestClient(app):
        pass
