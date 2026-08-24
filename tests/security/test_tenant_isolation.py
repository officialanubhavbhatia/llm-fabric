"""Adversarial cross-tenant isolation tests.

A failure in this module means one customer can read another customer's data.
It is marked `isolation` so CI can run it as its own gate.

The six resource families named in the phase brief are exercised through one
shared battery of attacks, driven by the table below. Doing it that way means a
new repository added later cannot quietly ship with weaker guarantees than the
existing ones: it either joins the table or its absence is visible.

Every attack is paired with a control assertion showing the victim can still
reach their own record. Without the control, a fixture that silently stored
nothing would make the whole file pass while proving nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from llm_fabric.errors import ResourceNotFoundError, TenantIsolationError
from llm_fabric.eval.schema import EvalProvenance, EvalResult, EvalRun, EvaluatorKind
from llm_fabric.heal.schema import Incident, IncidentSeverity, LearningJob
from llm_fabric.storage.records import (
    ConversationMessage,
    EvalExample,
    IntentExample,
    PromptDefinition,
    TraceRecord,
    TraceSpan,
)
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.cache import CacheNamespace, TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope
from llm_fabric.tenancy.store import TenantScopedStore

pytestmark = pytest.mark.isolation


# -- the resource table ------------------------------------------------------


@dataclass(frozen=True)
class Resource:
    """One resource family and how to seed, read and enumerate it."""

    name: str
    seed: Callable[[TenantStores, TenantScope], str]
    get: Callable[[TenantStores, TenantScope, str], Any]
    require: Callable[[TenantStores, TenantScope, str], Any]
    listing: Callable[[TenantStores, TenantScope], list[Any]]
    store: Callable[[TenantStores], TenantScopedStore[Any]]
    secret: str


def _seed_conversation(stores: TenantStores, scope: TenantScope) -> str:
    conversation = stores.conversations.create(scope, title="merger terms")
    stores.conversations.append_message(
        scope,
        conversation.conversation_id,
        ConversationMessage(role="user", content="ACME_CONFIDENTIAL_MERGER"),
    )
    return conversation.conversation_id


def _seed_trace(stores: TenantStores, scope: TenantScope) -> str:
    trace = TraceRecord(
        tenant_id=scope.tenant_id,
        trace_id="trace-acme-0001",
        request_id="req-acme-0001",
        user_id=scope.user_id,
        spans=(
            TraceSpan(
                name="router.select",
                started_at=0.0,
                duration_ms=1.5,
                attributes={"prompt_preview": "ACME_CONFIDENTIAL_MERGER"},
            ),
        ),
    )
    stores.traces.record(scope, trace)
    return trace.trace_id


def _seed_intent(stores: TenantStores, scope: TenantScope) -> str:
    example = IntentExample(
        tenant_id=scope.tenant_id,
        text="ACME_CONFIDENTIAL_MERGER",
        intent_id="billing.dispute",
        taxonomy_version="v3",
        example_id="intex-acme-0001",
    )
    stores.intent_examples.add(scope, example)
    return example.example_id


def _seed_prompt(stores: TenantStores, scope: TenantScope) -> str:
    prompt = PromptDefinition(
        tenant_id=scope.tenant_id,
        prompt_id="support.triage",
        version=7,
        owner="alice",
        purpose="triage inbound support",
        template="You are ACME_CONFIDENTIAL_MERGER assistant",
    )
    stores.prompts.publish(scope, prompt)
    return prompt.key


def _seed_dataset(stores: TenantStores, scope: TenantScope) -> str:
    dataset = stores.eval_datasets.create(
        scope,
        name="golden-set",
        examples=(EvalExample(input="ACME_CONFIDENTIAL_MERGER", expected="yes"),),
    )
    return dataset.dataset_id


def _seed_eval_run(stores: TenantStores, scope: TenantScope) -> str:
    run = EvalRun(
        tenant_id=scope.tenant_id,
        suite_name="golden",
        provenance=EvalProvenance(
            dataset_version="seed",
            configuration={"secret": "ACME_CONFIDENTIAL_MERGER"},
        ),
        results=(
            EvalResult(
                task="leak",
                evaluator=EvaluatorKind.DETERMINISTIC,
                metrics={"exact_match": 1.0},
                note="ACME_CONFIDENTIAL_MERGER",
            ),
        ),
    )
    stores.eval_runs.put(scope, run)
    return run.run_id


def _seed_incident(stores: TenantStores, scope: TenantScope) -> str:
    incident = Incident(
        tenant_id=scope.tenant_id,
        title="merger leak",
        severity=IncidentSeverity.WARNING,
        summary="ACME_CONFIDENTIAL_MERGER",
    )
    stores.incidents.put(scope, incident)
    return incident.incident_id


def _seed_learning_job(stores: TenantStores, scope: TenantScope) -> str:
    job = LearningJob(
        tenant_id=scope.tenant_id,
        reason="ACME_CONFIDENTIAL_MERGER",
    )
    stores.learning_jobs.put(scope, job)
    return job.job_id


RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="conversations",
        seed=_seed_conversation,
        get=lambda s, scope, key: s.conversations.get(scope, key),
        require=lambda s, scope, key: s.conversations.require(scope, key),
        listing=lambda s, scope: s.conversations.list(scope),
        store=lambda s: s.conversations.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="traces",
        seed=_seed_trace,
        get=lambda s, scope, key: s.traces.get(scope, key),
        require=lambda s, scope, key: s.traces.require(scope, key),
        listing=lambda s, scope: s.traces.list(scope),
        store=lambda s: s.traces.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="intents",
        seed=_seed_intent,
        get=lambda s, scope, key: s.intent_examples.get(scope, key),
        require=lambda s, scope, key: s.intent_examples.require(scope, key),
        listing=lambda s, scope: s.intent_examples.list(scope),
        store=lambda s: s.intent_examples.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="prompt_definitions",
        seed=_seed_prompt,
        get=lambda s, scope, key: s.prompts.get(scope, *_split_prompt_key(key)),
        require=lambda s, scope, key: s.prompts.require(scope, *_split_prompt_key(key)),
        listing=lambda s, scope: s.prompts.list(scope),
        store=lambda s: s.prompts.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="evaluation_datasets",
        seed=_seed_dataset,
        get=lambda s, scope, key: s.eval_datasets.get(scope, key),
        require=lambda s, scope, key: s.eval_datasets.require(scope, key),
        listing=lambda s, scope: s.eval_datasets.list(scope),
        store=lambda s: s.eval_datasets.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="evaluation_runs",
        seed=_seed_eval_run,
        get=lambda s, scope, key: s.eval_runs.get(scope, key),
        require=lambda s, scope, key: s.eval_runs.require(scope, key),
        listing=lambda s, scope: s.eval_runs.list(scope),
        store=lambda s: s.eval_runs.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="incidents",
        seed=_seed_incident,
        get=lambda s, scope, key: s.incidents.get(scope, key),
        require=lambda s, scope, key: s.incidents.require(scope, key),
        listing=lambda s, scope: s.incidents.list(scope),
        store=lambda s: s.incidents.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
    Resource(
        name="learning_jobs",
        seed=_seed_learning_job,
        get=lambda s, scope, key: s.learning_jobs.get(scope, key),
        require=lambda s, scope, key: s.learning_jobs.require(scope, key),
        listing=lambda s, scope: s.learning_jobs.list(scope),
        store=lambda s: s.learning_jobs.store,
        secret="ACME_CONFIDENTIAL_MERGER",
    ),
)


def _split_prompt_key(key: str) -> tuple[str, int]:
    prompt_id, _, version = key.partition("@")
    return prompt_id, int(version)


def _ids(resource: Resource) -> str:
    return resource.name


# -- the battery -------------------------------------------------------------


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_victim_can_read_their_own_record(
    resource: Resource, stores: TenantStores, victim: TenantScope
) -> None:
    """Control. If this fails the other assertions in this file prove nothing."""
    key = resource.seed(stores, victim)

    assert resource.get(stores, victim, key) is not None
    assert resource.require(stores, victim, key) is not None
    assert len(resource.listing(stores, victim)) == 1


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_attacker_cannot_read_victim_record_by_id(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    key = resource.seed(stores, victim)

    assert resource.get(stores, attacker, key) is None


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_attacker_sees_absence_rather_than_denial(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    """404, not 403.

    Refusing with "forbidden" would confirm the id exists somewhere, which turns
    the endpoint into an oracle for enumerating another tenant's record ids.
    """
    key = resource.seed(stores, victim)

    with pytest.raises(ResourceNotFoundError) as caught:
        resource.require(stores, attacker, key)

    assert caught.value.status_code == 404


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_attacker_listing_never_includes_victim_records(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    resource.seed(stores, victim)

    assert resource.listing(stores, attacker) == []


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_attacker_cannot_delete_victim_record(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    key = resource.seed(stores, victim)
    store = resource.store(stores)

    assert store.delete(attacker, key) is False
    assert resource.get(stores, victim, key) is not None, "victim's record was destroyed"


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_attacker_cannot_clear_victim_records(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    key = resource.seed(stores, victim)
    store = resource.store(stores)

    assert store.clear_tenant(attacker) == 0
    assert resource.get(stores, victim, key) is not None


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_attacker_key_enumeration_reveals_nothing(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    resource.seed(stores, victim)
    store = resource.store(stores)

    assert store.keys(attacker) == []
    assert store.count(attacker) == 0


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_sharing_a_user_id_does_not_share_data(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker_same_user_id: TenantScope,
) -> None:
    """Colliding user ids across tenants must not collide in storage."""
    key = resource.seed(stores, victim)

    assert resource.get(stores, attacker_same_user_id, key) is None
    assert resource.listing(stores, attacker_same_user_id) == []


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_victim_secret_never_appears_in_attacker_view(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    """Belt and braces: no reachable surface renders the victim's payload."""
    key = resource.seed(stores, victim)

    visible = repr(
        (
            resource.get(stores, attacker, key),
            resource.listing(stores, attacker),
            resource.store(stores).keys(attacker),
        )
    )

    assert resource.secret not in visible


@pytest.mark.parametrize("resource", RESOURCES, ids=_ids)
def test_writing_a_foreign_record_is_refused(
    resource: Resource,
    stores: TenantStores,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    """A record whose own tenant disagrees with the scope must not be stored.

    This is the poisoning direction: planting a record inside another tenant's
    namespace, or smuggling one out by mislabelling it.
    """
    key = resource.seed(stores, victim)
    store = resource.store(stores)
    victim_record = resource.get(stores, victim, key)

    with pytest.raises(TenantIsolationError):
        store.put(attacker, key, victim_record)


# -- caches ------------------------------------------------------------------

CACHE_NAMESPACES = tuple(CacheNamespace)


@pytest.mark.parametrize("namespace", CACHE_NAMESPACES, ids=lambda ns: ns.value)
def test_attacker_cannot_read_victim_cache_entry(
    namespace: CacheNamespace,
    cache: TenantScopedCache,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    """The classic leak: same prompt, two tenants, one answer.

    The discriminators are byte-identical for both callers. Only the tenant
    differs, and that must be enough.
    """
    parts = {"prompt": "what is our Q4 revenue?", "model": "cheap"}

    cache.put(victim, namespace, parts, "ACME_CONFIDENTIAL_REVENUE")

    assert cache.get(victim, namespace, parts) == "ACME_CONFIDENTIAL_REVENUE"
    assert cache.get(attacker, namespace, parts) is None


@pytest.mark.parametrize("namespace", CACHE_NAMESPACES, ids=lambda ns: ns.value)
def test_cache_keys_differ_across_tenants(
    namespace: CacheNamespace,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    from llm_fabric.tenancy.cache import CacheKey

    parts = {"prompt": "identical", "model": "cheap"}
    victim_key = CacheKey.build(namespace, victim, parts)
    attacker_key = CacheKey.build(namespace, attacker, parts)

    assert victim_key.fingerprint != attacker_key.fingerprint
    assert victim_key.as_storage_key() != attacker_key.as_storage_key()


@pytest.mark.parametrize("namespace", CACHE_NAMESPACES, ids=lambda ns: ns.value)
def test_attacker_cannot_invalidate_victim_cache(
    namespace: CacheNamespace,
    cache: TenantScopedCache,
    victim: TenantScope,
    attacker: TenantScope,
) -> None:
    """Cache poisoning by eviction is still an attack."""
    parts = {"prompt": "shared question", "model": "cheap"}
    cache.put(victim, namespace, parts, "victim value")

    assert cache.invalidate(attacker, namespace, parts) is False
    assert cache.invalidate_namespace(attacker, namespace) == 0
    assert cache.get(victim, namespace, parts) == "victim value"


def test_attacker_cannot_flush_every_victim_cache(
    cache: TenantScopedCache, victim: TenantScope, attacker: TenantScope
) -> None:
    parts = {"prompt": "shared question"}
    for namespace in CACHE_NAMESPACES:
        cache.put(victim, namespace, parts, f"victim-{namespace.value}")

    assert cache.invalidate_tenant(attacker) == 0

    for namespace in CACHE_NAMESPACES:
        assert cache.get(victim, namespace, parts) == f"victim-{namespace.value}"


def test_a_tenant_can_flush_only_their_own_caches(
    cache: TenantScopedCache, victim: TenantScope, attacker: TenantScope
) -> None:
    parts = {"prompt": "shared question"}
    cache.put(victim, CacheNamespace.EXACT_RESPONSE, parts, "victim value")
    cache.put(attacker, CacheNamespace.EXACT_RESPONSE, parts, "attacker value")

    cache.invalidate_tenant(attacker)

    assert cache.get(victim, CacheNamespace.EXACT_RESPONSE, parts) == "victim value"
    assert cache.get(attacker, CacheNamespace.EXACT_RESPONSE, parts) is None


def test_cache_namespaces_do_not_bleed_into_each_other(
    cache: TenantScopedCache, victim: TenantScope
) -> None:
    """Distinct caches stay distinct even for one tenant with one key."""
    parts = {"prompt": "same key material"}
    cache.put(victim, CacheNamespace.EXACT_RESPONSE, parts, "exact")

    assert cache.get(victim, CacheNamespace.SEMANTIC_RESPONSE, parts) is None
    assert cache.get(victim, CacheNamespace.INTENT, parts) is None
    assert cache.get(victim, CacheNamespace.EXACT_RESPONSE, parts) == "exact"


# -- the audit ---------------------------------------------------------------


def test_no_isolation_violations_are_recorded_during_normal_reads(
    stores: TenantStores, victim: TenantScope, attacker: TenantScope
) -> None:
    """Cross-tenant reads must be *absent*, not *caught*.

    Namespacing means an attacker's read never even reaches the ownership check.
    A violation appearing here would mean the second line of defence is doing
    work the first was supposed to make unnecessary.
    """
    for resource in RESOURCES:
        key = resource.seed(stores, victim)
        resource.get(stores, attacker, key)
        resource.listing(stores, attacker)

    assert stores.audit.count == 0


def test_the_ownership_check_actually_fires_when_namespacing_is_bypassed(
    stores: TenantStores, victim: TenantScope, attacker: TenantScope
) -> None:
    """Prove the second line of defence is live, not decorative.

    Reaching into the store's internals is exactly what a future refactor might
    do by accident. Simulating it here is the only way to show the ownership
    re-check would catch it.
    """
    store = stores.conversations.store
    key = _seed_conversation(stores, victim)
    smuggled = store.get(victim, key)

    # Plant the victim's record directly inside the attacker's bucket, skipping
    # the write-side guard the way a broken index or a bad migration would.
    store._buckets.setdefault(attacker.tenant_id, {})[key] = store._buckets[victim.tenant_id][key]

    with pytest.raises(TenantIsolationError):
        store.get(attacker, key)

    assert stores.audit.count == 1
    violation = stores.audit.violations[0]
    assert violation.requesting_tenant == attacker.tenant_id
    assert violation.owning_tenant == victim.tenant_id
    assert smuggled is not None
