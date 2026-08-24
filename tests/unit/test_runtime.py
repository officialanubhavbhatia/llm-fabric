"""initialize_runtime is the only production initialization path.

Narrow unit tests mock the network. Live subprocess tests in
`test_startup_paths.py` talk to real PostgreSQL and Redis.
"""

from __future__ import annotations

import time

import pytest

from llm_fabric.config import Settings, validate_startup
from llm_fabric.errors import ConfigurationError
from llm_fabric.gateway.app import create_app
from llm_fabric.identity.revocation import InMemoryRevocationStore
from llm_fabric.router.health import HealthTracker
from llm_fabric.runtime import initialize_runtime
from llm_fabric.storage.postgres import probe_database
from llm_fabric.storage.redis import probe_redis
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.quota import QuotaLedger


def _production(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "allow_anonymous": False,
        "api_keys": ["a-long-enough-api-key"],
        "database_url": "postgresql://fabric:supersecret@127.0.0.1:1/fabric",
        "redis_url": "redis://127.0.0.1:1/0",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_development_does_not_probe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_settings: Settings) -> None:
        raise AssertionError("development must not probe production dependencies")

    monkeypatch.setattr("llm_fabric.runtime.probe_distributed_state", fail)
    settings = Settings(_env_file=None, environment="development", api_keys=[])
    initialize_runtime(settings)


def test_test_environment_does_not_probe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_settings: Settings) -> None:
        raise AssertionError("test must not probe production dependencies")

    monkeypatch.setattr("llm_fabric.runtime.probe_distributed_state", fail)
    settings = Settings(_env_file=None, environment="test", api_keys=[], allow_anonymous=True)
    initialize_runtime(settings)


def test_production_initialize_runtime_probes_postgres_and_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def note_db(url: str, **_kwargs: object) -> None:
        seen.append(f"postgres:{url}")

    def note_redis(url: str, **_kwargs: object) -> None:
        seen.append(f"redis:{url}")

    monkeypatch.setattr("llm_fabric.storage.runtime.probe_database", note_db)
    monkeypatch.setattr("llm_fabric.storage.runtime.probe_redis", note_redis)
    monkeypatch.setattr("llm_fabric.storage.runtime.assert_schema_revision", lambda _url: None)
    settings = _production(
        database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
        redis_url="redis://127.0.0.1:6379/0",
    )
    runtime = initialize_runtime(settings)
    assert runtime.settings is settings
    assert seen == [
        "postgres:postgresql://fabric:fabric@127.0.0.1:5432/fabric",
        "redis:redis://127.0.0.1:6379/0",
    ]


def test_create_app_uses_initialize_runtime_not_a_cli_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def mark(settings: Settings) -> None:
        called.append(settings.environment)

    monkeypatch.setattr("llm_fabric.runtime.probe_distributed_state", mark)
    stores = TenantStores()
    create_app(
        settings=_production(
            database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
            redis_url="redis://127.0.0.1:6379/0",
        ),
        stores=stores,
        cache=TenantScopedCache(audit=stores.audit),
        quota=QuotaLedger(),
        health_tracker=HealthTracker(),
        revocation_store=InMemoryRevocationStore(),
    )
    assert called == ["production"]


def test_create_app_refuses_production_when_postgres_is_unreachable() -> None:
    with pytest.raises(ConfigurationError, match="PostgreSQL is unreachable"):
        create_app(settings=_production())


def test_create_app_refuses_production_when_redis_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llm_fabric.storage.runtime.probe_database", lambda _url, **_k: None)
    with pytest.raises(ConfigurationError, match="Redis is unreachable"):
        create_app(
            settings=_production(
                database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
                redis_url="redis://127.0.0.1:1/0",
            )
        )


def test_probe_database_does_not_leak_credentials() -> None:
    secret = "supersecret-password-value"
    with pytest.raises(ConfigurationError) as raised:
        probe_database(f"postgresql://fabric:{secret}@127.0.0.1:1/fabric")
    text = raised.value.message + str(raised.value)
    assert secret not in text
    assert "postgresql://" not in text
    assert raised.value.__cause__ is None


def test_probe_redis_does_not_leak_credentials() -> None:
    secret = "supersecret-redis-password"
    with pytest.raises(ConfigurationError) as raised:
        probe_redis(f"redis://:{secret}@127.0.0.1:1/0")
    text = raised.value.message + str(raised.value)
    assert secret not in text
    assert raised.value.__cause__ is None


def test_postgres_probe_is_bounded() -> None:
    started = time.monotonic()
    with pytest.raises(ConfigurationError, match="PostgreSQL is unreachable"):
        probe_database(
            "postgresql://fabric:fabric@192.0.2.1:5432/fabric",
            timeout_s=2,
        )
    elapsed = time.monotonic() - started
    assert elapsed < 8, f"probe hung for {elapsed:.1f}s"


def test_redis_probe_is_bounded() -> None:
    started = time.monotonic()
    with pytest.raises(ConfigurationError, match="Redis is unreachable"):
        probe_redis("redis://192.0.2.1:6379/0", timeout_s=2)
    elapsed = time.monotonic() - started
    assert elapsed < 8, f"probe hung for {elapsed:.1f}s"


def test_production_config_still_refuses_missing_auth() -> None:
    with pytest.raises(ConfigurationError, match="ALLOW_ANONYMOUS|without authentication"):
        initialize_runtime(
            Settings(
                _env_file=None,
                environment="production",
                allow_anonymous=True,
                api_keys=[],
                database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
                redis_url="redis://127.0.0.1:6379/0",
            )
        )


def test_validate_startup_does_not_require_a_live_database() -> None:
    validate_startup(
        Settings(
            _env_file=None,
            environment="production",
            allow_anonymous=False,
            api_keys=["a-long-enough-api-key"],
            database_url="postgresql://fabric:fabric@127.0.0.1:5432/fabric",
            redis_url="redis://127.0.0.1:6379/0",
        )
    )
