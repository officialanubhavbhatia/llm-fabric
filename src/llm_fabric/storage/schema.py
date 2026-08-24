"""Schema lifecycle: Alembic is the production source of truth.

Production gateway workers never create or mutate tables. They confirm the
database is already at the migration head and then only run DML. Development
and test may still call `init_schema` (`create_all`) as an explicit convenience.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from llm_fabric.errors import ConfigurationError
from llm_fabric.storage.postgres import create_database_engine

#: Linear Alembic head for this release. Update when adding a revision.
#: `expected_heads()` prefers the on-disk Alembic scripts when they are present.
EXPECTED_HEAD = "0003_revoke_app_ddl"

REQUIRED_TABLES = frozenset(
    {
        "alembic_version",
        "audit_events",
        "tenant_records",
        "tenants",
        "usage_events",
        "users",
    }
)


def _alembic_paths() -> tuple[Path, Path] | None:
    """Locate alembic.ini independent of process cwd."""
    here = Path(__file__).resolve()
    searched = [Path.cwd(), *here.parents]
    for root in searched:
        ini = root / "alembic.ini"
        scripts = root / "alembic"
        if ini.is_file() and (scripts / "versions").is_dir():
            return ini, scripts
    return None


def expected_heads() -> frozenset[str]:
    """The Alembic head revision(s) this build will serve."""
    paths = _alembic_paths()
    if paths is None:
        return frozenset({EXPECTED_HEAD})
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    ini, scripts = paths
    config = Config(str(ini))
    config.set_main_option("script_location", str(scripts))
    return frozenset(ScriptDirectory.from_config(config).get_heads())


def current_revision(engine: Engine) -> str | None:
    """Return the database's Alembic revision, or None if unmigrated."""
    inspector = inspect(engine)
    if "alembic_version" not in inspector.get_table_names():
        return None
    with engine.connect() as connection:
        value = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    if value is None:
        return None
    return str(value)


def assert_schema_revision(url: str) -> None:
    """Refuse to serve when the database is not at this build's migration head.

    Missing, empty, or behind `alembic_version` is a startup failure. Workers
    never run `alembic upgrade` themselves.
    """
    engine = create_database_engine(url, connect_timeout_s=3)
    try:
        _assert_engine_schema(engine)
    finally:
        engine.dispose()


def _assert_engine_schema(engine: Engine) -> None:
    heads = expected_heads()
    if len(heads) != 1:
        raise ConfigurationError(
            "production startup validation failed: this build has multiple "
            "Alembic heads; linearise migrations before serving"
        )
    head = next(iter(heads))
    present = set(inspect(engine).get_table_names())
    missing = sorted(REQUIRED_TABLES - present)
    revision = current_revision(engine)
    if revision is None:
        raise ConfigurationError(
            "production startup validation failed: database has no Alembic "
            "revision; run `alembic upgrade head` before starting workers"
        )
    if revision not in heads:
        raise ConfigurationError(
            "production startup validation failed: database revision "
            f"'{revision}' is not head '{head}'; run `alembic upgrade head` "
            "before starting workers"
        )
    if missing:
        raise ConfigurationError(
            "production startup validation failed: migrated database is missing "
            "tables: " + ", ".join(missing)
        )
