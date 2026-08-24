"""Backup dump and restore against a real sqlite engine."""

from __future__ import annotations

from pathlib import Path

from llm_fabric.storage.backup import clone_sqlite
from llm_fabric.storage.postgres import create_database_engine, init_schema
from llm_fabric.storage.records import ConversationMessage
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.scope import TenantScope


def test_sqlite_backup_restores_tenant_records(tmp_path: Path) -> None:
    source_url = f"sqlite:///{tmp_path / 'source.db'}"
    dest_url = f"sqlite:///{tmp_path / 'restored.db'}"
    engine = create_database_engine(source_url)
    init_schema(engine)
    stores = TenantStores(engine=engine)
    alice = TenantScope(tenant_id="tenant-a", user_id="alice")
    bob = TenantScope(tenant_id="tenant-b", user_id="bob")
    stores.conversations.create(
        alice, title="a-secret", messages=(ConversationMessage(role="user", content="private"),)
    )
    stores.conversations.create(bob, title="b-secret")
    stores.audit_events.record(
        alice, actor="alice", action="policy.changed", target="routing", reason="test"
    )
    engine.dispose()

    dump = tmp_path / "backup.json"
    counts = clone_sqlite(source_url, dest_url, dump)
    assert counts["tenant_records"] >= 2

    restored = create_database_engine(dest_url)
    restored_stores = TenantStores(engine=restored)
    assert {row.title for row in restored_stores.conversations.list(alice)} == {"a-secret"}
    assert {row.title for row in restored_stores.conversations.list(bob)} == {"b-secret"}
    alice_id = restored_stores.conversations.list(alice)[0].conversation_id
    assert restored_stores.conversations.get(bob, alice_id) is None
    events = restored_stores.audit_events.list(alice)
    assert events[0].action == "policy.changed"
    restored.dispose()
