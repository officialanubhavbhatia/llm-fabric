"""Durable tenant store over SQLite (PostgreSQL dialect of the same schema)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from llm_fabric.errors import TenantIsolationError
from llm_fabric.storage.postgres import PostgresTenantStore, create_database_engine, init_schema
from llm_fabric.storage.records import Conversation, ConversationMessage
from llm_fabric.storage.repositories import ConversationRepository, TenantStores
from llm_fabric.tenancy.scope import TenantScope


@dataclass(frozen=True, slots=True)
class Note:
    tenant_id: str
    body: str


@pytest.fixture
def engine(tmp_path: Path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'fabric.db'}")
    init_schema(engine)
    return engine


def test_postgres_store_round_trips_a_record(engine) -> None:
    store = PostgresTenantStore("note", Note, engine, max_records_per_tenant=10)
    scope = TenantScope(tenant_id="acme", user_id="alice")
    store.put(scope, "n1", Note(tenant_id="acme", body="hello"))

    assert store.get(scope, "n1") == Note(tenant_id="acme", body="hello")
    assert store.count(scope) == 1


def test_postgres_store_hides_another_tenants_record(engine) -> None:
    store = PostgresTenantStore("note", Note, engine)
    acme = TenantScope(tenant_id="acme")
    mallory = TenantScope(tenant_id="mallory")
    store.put(acme, "shared-key", Note(tenant_id="acme", body="secret"))

    assert store.get(mallory, "shared-key") is None
    assert store.list(mallory) == []
    assert store.delete(mallory, "shared-key") is False
    assert store.get(acme, "shared-key") is not None


def test_postgres_store_refuses_to_write_a_foreign_record(engine) -> None:
    store = PostgresTenantStore("note", Note, engine)
    acme = TenantScope(tenant_id="acme")
    with pytest.raises(TenantIsolationError):
        store.put(acme, "x", Note(tenant_id="mallory", body="nope"))


def test_postgres_store_bounds_per_tenant(engine) -> None:
    store = PostgresTenantStore("note", Note, engine, max_records_per_tenant=3)
    scope = TenantScope(tenant_id="acme")
    for index in range(10):
        store.put(scope, f"n{index}", Note(tenant_id="acme", body=str(index)))
    assert store.count(scope) == 3
    assert store.get(scope, "n0") is None
    assert store.get(scope, "n9") is not None


def test_tenant_stores_over_sqlite_hold_conversations(engine) -> None:
    stores = TenantStores(engine=engine)
    scope = TenantScope(tenant_id="acme", user_id="alice")
    created = stores.conversations.create(
        scope,
        title="hello",
        messages=(ConversationMessage(role="user", content="hi"),),
    )
    loaded = stores.conversations.require(scope, created.conversation_id)
    assert loaded.title == "hello"
    assert loaded.messages[0].content == "hi"
    assert isinstance(stores.conversations, ConversationRepository)


def test_conversation_codec_survives_nested_messages(engine) -> None:
    store = PostgresTenantStore("conversation", Conversation, engine)
    scope = TenantScope(tenant_id="acme")
    record = Conversation(
        tenant_id="acme",
        user_id="alice",
        title="t",
        messages=(ConversationMessage(role="user", content="ping"),),
    )
    store.put(scope, record.conversation_id, record)
    loaded = store.require(scope, record.conversation_id)
    assert loaded.messages[0].content == "ping"
