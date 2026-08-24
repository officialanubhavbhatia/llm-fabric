"""Alembic environment. Schema metadata comes from the SQLAlchemy models."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from llm_fabric.storage.postgres import Base, create_database_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    url = (
        os.environ.get("LLM_FABRIC_MIGRATION_DATABASE_URL")
        or os.environ.get("LLM_FABRIC_DATABASE_URL")
        or config.get_main_option("sqlalchemy.url")
    )
    if not url:
        raise RuntimeError(
            "LLM_FABRIC_MIGRATION_DATABASE_URL or LLM_FABRIC_DATABASE_URL is "
            "required to run migrations"
        )
    return url


def run_migrations_offline() -> None:
    url = _database_url()
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _database_url()
    connectable = create_database_engine(url)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
