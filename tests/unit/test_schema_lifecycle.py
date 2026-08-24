"""Production workers never DDL. Alembic is the schema source of truth."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import inspect

from llm_fabric.config import Settings
from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import (
    create_database_engine,
    init_schema,
    provision_application_role,
)
from llm_fabric.storage.runtime import build_engine
from llm_fabric.storage.schema import EXPECTED_HEAD, assert_schema_revision, expected_heads


def _settings(environment: str, database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment=environment,
        allow_anonymous=environment != "production",
        api_keys=["a-long-enough-api-key"] if environment == "production" else [],
        database_url=database_url,
        redis_url="redis://127.0.0.1:6379/0",
    )


def test_expected_head_matches_alembic_scripts() -> None:
    assert expected_heads() == {EXPECTED_HEAD}


def test_provision_application_role_never_grants_schema_create() -> None:
    import inspect as stdlib_inspect

    source = stdlib_inspect.getsource(provision_application_role)
    assert "GRANT USAGE, CREATE ON SCHEMA" not in source
    assert "REVOKE CREATE ON SCHEMA public FROM" in source


def test_postgres_init_never_grants_schema_create() -> None:
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[2]
        / "deployments"
        / "docker"
        / "postgres-init"
        / "01-app-role.sql"
    ).read_text(encoding="utf-8")
    assert "GRANT USAGE ON SCHEMA public TO fabric_app" in sql
    assert "REVOKE CREATE ON SCHEMA public FROM fabric_app" in sql
    assert "GRANT USAGE, CREATE ON SCHEMA" not in sql


def test_development_build_engine_runs_create_all(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'dev.db'}"
    with patch("llm_fabric.storage.runtime.init_schema", wraps=init_schema) as wrapped:
        engine = build_engine(_settings("development", url))
        assert engine is not None
        wrapped.assert_called_once()
        tables = set(inspect(engine).get_table_names())
        engine.dispose()
    assert "tenant_records" in tables
    assert "usage_events" in tables


def test_production_build_engine_does_not_run_create_all(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'prod.db'}"
    with patch("llm_fabric.storage.runtime.init_schema") as init:
        engine = build_engine(_settings("production", url))
        assert engine is not None
        init.assert_not_called()
        tables = set(inspect(engine).get_table_names())
        engine.dispose()
    assert "tenant_records" not in tables


def test_test_environment_build_engine_still_bootstraps(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = build_engine(_settings("test", url))
    assert engine is not None
    names = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "tenant_records" in names
    assert "usage_events" in names


def test_init_schema_still_creates_sqlite_tables(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'init.db'}"
    engine = create_database_engine(url)
    init_schema(engine)
    names = set(inspect(engine).get_table_names())
    engine.dispose()
    assert "audit_events" in names


def test_assert_schema_revision_refuses_empty_sqlite(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    engine = create_database_engine(url)
    engine.dispose()
    with pytest.raises(ConfigurationError, match="no Alembic"):
        assert_schema_revision(url)
