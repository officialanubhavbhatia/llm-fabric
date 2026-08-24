"""Cross-tenant isolation against the durable record store.

The in-memory suite remains. These tests exercise the SQLAlchemy backend
(SQLite always; PostgreSQL when `LLM_FABRIC_TEST_DATABASE_URL` is set).

PostgreSQL RLS is only meaningful for a NOSUPERUSER / NOBYPASSRLS role. The
Docker superuser `fabric` is used to create that role; the tests then connect
as `fabric_app`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError

from llm_fabric.errors import ResourceNotFoundError, TenantIsolationError
from llm_fabric.storage.postgres import (
    APPLICATION_ROLE,
    create_database_engine,
    current_role_bypasses_rls,
    init_schema,
    provision_application_role,
)
from llm_fabric.storage.records import ConversationMessage, IntentExample, PromptDefinition
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

pytestmark = [pytest.mark.isolation, pytest.mark.tenant_isolation]


def _postgres_url() -> str | None:
    url = os.environ.get("LLM_FABRIC_TEST_DATABASE_URL")
    if url and url.startswith("postgresql"):
        return url
    return None


def _app_database_url(admin_url: str) -> str:
    override = os.environ.get("LLM_FABRIC_TEST_DATABASE_APP_URL")
    if override:
        return override
    if "://fabric:" in admin_url:
        return admin_url.replace("://fabric:", f"://{APPLICATION_ROLE}:", 1)
    return admin_url


def _stores(tmp_path: Path) -> TenantStores:
    url = os.environ.get("LLM_FABRIC_TEST_DATABASE_URL")
    if url:
        engine = create_database_engine(url)
        init_schema(engine)
        if url.startswith("postgresql"):
            provision_application_role(engine)
            app_url = _app_database_url(url)
            if app_url != url:
                engine.dispose()
                engine = create_database_engine(app_url)
        return TenantStores(engine=engine)
    engine = create_database_engine(f"sqlite:///{tmp_path / 'isolation.db'}")
    init_schema(engine)
    return TenantStores(engine=engine)


@pytest.fixture
def stores(tmp_path: Path) -> TenantStores:
    return _stores(tmp_path)


@pytest.fixture
def tenants() -> dict[str, TenantScope]:
    suffix = uuid.uuid4().hex[:8]
    return {
        "a": TenantScope(tenant_id=f"tenant-a-{suffix}", user_id="alice"),
        "b": TenantScope(tenant_id=f"tenant-b-{suffix}", user_id="bob"),
        "c": TenantScope(tenant_id=f"tenant-c-{suffix}", user_id="cara"),
    }


def test_conversation_read_write_list_delete_are_tenant_bound(
    stores: TenantStores, tenants: dict[str, TenantScope]
) -> None:
    a, b, c = tenants["a"], tenants["b"], tenants["c"]
    conv_a = stores.conversations.create(
        a, title="a-secret", messages=(ConversationMessage(role="user", content="private-a"),)
    )
    conv_b = stores.conversations.create(b, title="b-secret")
    stores.conversations.create(c, title="c-secret")

    assert stores.conversations.get(b, conv_a.conversation_id) is None
    assert stores.conversations.get(c, conv_a.conversation_id) is None
    assert stores.conversations.require(a, conv_a.conversation_id).title == "a-secret"
    assert {row.title for row in stores.conversations.list(a)} == {"a-secret"}
    assert stores.conversations.delete(b, conv_a.conversation_id) is False
    assert stores.conversations.get(a, conv_a.conversation_id) is not None
    assert stores.conversations.delete(a, conv_b.conversation_id) is False


def test_conversation_update_is_tenant_bound(
    stores: TenantStores, tenants: dict[str, TenantScope]
) -> None:
    a, b = tenants["a"], tenants["b"]
    conv = stores.conversations.create(a, title="a-secret")
    updated = stores.conversations.append_message(
        a, conv.conversation_id, ConversationMessage(role="assistant", content="reply-a")
    )
    assert len(updated.messages) == 1
    with pytest.raises(ResourceNotFoundError):
        stores.conversations.append_message(
            b, conv.conversation_id, ConversationMessage(role="user", content="stolen")
        )
    assert stores.conversations.require(a, conv.conversation_id).messages[0].content == "reply-a"


def test_wrong_tenant_put_is_refused_before_sql(
    stores: TenantStores, tenants: dict[str, TenantScope]
) -> None:
    a, b = tenants["a"], tenants["b"]
    conv = stores.conversations.create(a, title="owned-by-a")
    with pytest.raises(TenantIsolationError):
        stores.conversations.store.put(b, conv.conversation_id, conv)


def test_traces_prompts_intents_and_evals_do_not_leak(
    stores: TenantStores, tenants: dict[str, TenantScope]
) -> None:
    from llm_fabric.storage.records import TraceRecord

    a, b = tenants["a"], tenants["b"]
    trace = TraceRecord(tenant_id=a.tenant_id, trace_id="tr_a", request_id="r1")
    stores.traces.record(a, trace)
    assert stores.traces.get(b, "tr_a") is None
    assert {row.trace_id for row in stores.traces.list(a)} == {"tr_a"}
    assert stores.traces.list(b) == []

    prompt = PromptDefinition(
        tenant_id=a.tenant_id,
        prompt_id="sys",
        version=1,
        owner="alice",
        purpose="chat",
        template="hello",
    )
    stores.prompts.publish(a, prompt)
    assert stores.prompts.get(b, "sys", 1) is None

    example = IntentExample(
        tenant_id=a.tenant_id,
        text="reset password",
        intent_id="account.reset",
        taxonomy_version="v1",
    )
    stores.intent_examples.add(a, example)
    assert stores.intent_examples.get(b, example.example_id) is None
    assert stores.intent_examples.get(a, example.example_id) is not None

    dataset = stores.eval_datasets.create(a, name="gold")
    assert stores.eval_datasets.get(b, dataset.dataset_id) is None
    assert stores.eval_datasets.get(a, dataset.dataset_id) is not None


def test_cache_hit_and_invalidation_are_tenant_bound(tenants: dict[str, TenantScope]) -> None:
    cache = TenantScopedCache()
    a, b = tenants["a"], tenants["b"]
    parts = {"prompt": "same-bytes"}
    cache.put(a, CacheNamespace.EXACT_RESPONSE, parts, {"answer": "for-a"})
    cache.put(b, CacheNamespace.EXACT_RESPONSE, parts, {"answer": "for-b"})

    assert cache.get(a, CacheNamespace.EXACT_RESPONSE, parts) == {"answer": "for-a"}
    assert cache.get(b, CacheNamespace.EXACT_RESPONSE, parts) == {"answer": "for-b"}
    assert cache.invalidate(b, CacheNamespace.EXACT_RESPONSE, parts) is True
    assert cache.get(a, CacheNamespace.EXACT_RESPONSE, parts) == {"answer": "for-a"}
    assert cache.get(b, CacheNamespace.EXACT_RESPONSE, parts) is None


@pytest.mark.skipif(
    _postgres_url() is None,
    reason="PostgreSQL RLS is only verified against a real Postgres",
)
def test_postgres_rls_hides_rows_when_session_tenant_disagrees(tmp_path: Path) -> None:
    """An incorrectly written SELECT without a tenant_id filter still cannot see A."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from llm_fabric.storage.postgres import TenantRecordRow, _bind_tenant

    stores = _stores(tmp_path)
    bypasses, role = current_role_bypasses_rls(stores.conversations.store._engine)  # type: ignore[attr-defined]
    assert not bypasses, f"RLS test connected as {role}, which bypasses row-level security"

    suffix = uuid.uuid4().hex[:8]
    a = TenantScope(tenant_id=f"tenant-a-{suffix}", user_id="alice")
    b = TenantScope(tenant_id=f"tenant-b-{suffix}", user_id="bob")
    stores.conversations.create(a, title="hidden")
    engine = stores.conversations.store._engine  # type: ignore[attr-defined]
    with Session(engine) as session:
        _bind_tenant(session, b.tenant_id)
        rows = session.execute(select(TenantRecordRow)).scalars().all()
    assert rows == []


@pytest.mark.skipif(
    _postgres_url() is None,
    reason="PostgreSQL RLS is only verified against a real Postgres",
)
def test_postgres_rls_blocks_insert_update_delete_for_the_wrong_tenant(tmp_path: Path) -> None:
    from sqlalchemy import delete, select, text, update
    from sqlalchemy.orm import Session

    from llm_fabric.storage.postgres import TenantRecordRow, _bind_tenant

    stores = _stores(tmp_path)
    engine = stores.conversations.store._engine  # type: ignore[attr-defined]
    bypasses, role = current_role_bypasses_rls(engine)
    assert not bypasses, f"RLS test connected as {role}, which bypasses row-level security"

    suffix = uuid.uuid4().hex[:8]
    a = TenantScope(tenant_id=f"tenant-a-{suffix}", user_id="alice")
    b = TenantScope(tenant_id=f"tenant-b-{suffix}", user_id="bob")
    conv = stores.conversations.create(a, title="owned-by-a")

    with Session(engine) as session:
        _bind_tenant(session, b.tenant_id)
        session.add(
            TenantRecordRow(
                store="conversation",
                tenant_id=a.tenant_id,
                key=f"stolen-{suffix}",
                payload={"title": "injected"},
                updated_at=0.0,
            )
        )
        with pytest.raises((DBAPIError, IntegrityError, ProgrammingError)):
            session.commit()

    with Session(engine) as session:
        _bind_tenant(session, b.tenant_id)
        result = session.execute(
            update(TenantRecordRow)
            .where(TenantRecordRow.key == conv.conversation_id)
            .values(payload={"title": "rewritten-by-b"})
        )
        session.commit()
        assert result.rowcount == 0

    with Session(engine) as session:
        _bind_tenant(session, b.tenant_id)
        result = session.execute(
            delete(TenantRecordRow).where(TenantRecordRow.key == conv.conversation_id)
        )
        session.commit()
        assert result.rowcount == 0

    with Session(engine) as session:
        _bind_tenant(session, a.tenant_id)
        remaining = (
            session.execute(
                select(TenantRecordRow).where(TenantRecordRow.key == conv.conversation_id)
            )
            .scalars()
            .all()
        )
    assert len(remaining) == 1
    assert remaining[0].payload.get("title") == "owned-by-a"

    with Session(engine) as session:
        _bind_tenant(session, b.tenant_id)
        session.execute(text("SELECT 1"))
        rows = session.execute(
            text("SELECT tenant_id FROM tenant_records WHERE key = :key"),
            {"key": conv.conversation_id},
        ).all()
    assert rows == []


@pytest.mark.skipif(
    _postgres_url() is None,
    reason="PostgreSQL RLS is only verified against a real Postgres",
)
def test_postgres_rls_covers_tenants_users_and_audit_tables(tmp_path: Path) -> None:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from llm_fabric.storage.postgres import AuditEventRow, TenantRow, UserRow, _bind_tenant

    stores = _stores(tmp_path)
    engine = stores.conversations.store._engine  # type: ignore[attr-defined]
    suffix = uuid.uuid4().hex[:8]
    tenant_a = f"tenant-a-{suffix}"
    tenant_b = f"tenant-b-{suffix}"

    admin_url = _postgres_url()
    assert admin_url is not None
    admin = create_database_engine(admin_url)
    with Session(admin) as session:
        session.add(TenantRow(tenant_id=tenant_a, name="A", created_at=1.0))
        session.add(UserRow(tenant_id=tenant_a, user_id="alice", roles="owner", created_at=1.0))
        session.add(
            AuditEventRow(
                event_id=f"aud_{suffix}",
                tenant_id=tenant_a,
                actor="alice",
                action="create",
                target="tenant",
                created_at=1.0,
            )
        )
        session.commit()
    admin.dispose()

    with Session(engine) as session:
        _bind_tenant(session, tenant_b)
        assert session.execute(select(TenantRow)).scalars().all() == []
        assert session.execute(select(UserRow)).scalars().all() == []
        assert session.execute(select(AuditEventRow)).scalars().all() == []
