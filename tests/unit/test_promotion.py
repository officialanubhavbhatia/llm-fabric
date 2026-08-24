"""Evidence-bound model promotion and production routing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.models import promotion
from llm_fabric.models.promotion import (
    PromotionConfig,
    PromotionStore,
    apply_overlay,
    apply_transition,
    artifact_ref,
    spec_identity,
)
from llm_fabric.router.plan import ExclusionRule, RoutePlanner, RouteRequest
from llm_fabric.router.registry import ModelRegistry, ModelSpec, PromotionState
from llm_fabric.router.tiers import ServiceTier


def _spec(**overrides: object) -> ModelSpec:
    values: dict[str, object] = {
        "id": "candidate",
        "provider": "vllm",
        "provider_model": "org/model",
        "revision": "rev-a",
        "digest": "sha256:a",
        "tiers": (ServiceTier.L10, ServiceTier.L12),
        "capabilities": ["chat", "streaming"],
    }
    values.update(overrides)
    return ModelSpec(**values)


def _probe(path: Path, spec: ModelSpec) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "deployment": spec.id,
                "provider": spec.provider,
                "model": spec.provider_model,
                "model_revision": spec.revision,
                "model_digest": spec.digest,
                "probe_version": "model-probe-v1",
                "environment": {"commit": "abc"},
                "capabilities": {
                    "chat": {"supported": True},
                    "streaming": {"supported": True},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _evaluation(path: Path, spec: ModelSpec) -> Path:
    path.write_text(
        json.dumps(
            {
                "eval_version": "model-eval-v1",
                "commit": "abc",
                "results": [
                    {
                        "deployment": spec.id,
                        "provider": spec.provider,
                        "status": "ok",
                        "identity": spec_identity(spec),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _shadow(path: Path, spec: ModelSpec) -> Path:
    path.write_text(
        json.dumps(
            {
                "eval_version": "routing-shadow-v1",
                "commit": "abc",
                "cases": [
                    {
                        "id": "coding-1",
                        "quality_shadow": {"shadow_selected": spec.id},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def artifact_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(promotion, "ARTIFACT_ROOTS", (tmp_path.resolve(),))
    return tmp_path


def test_promotion_without_required_probe_fails(tmp_path: Path) -> None:
    spec = _spec()
    store = PromotionStore(path=tmp_path / "state.json")
    result = apply_transition(
        spec,
        PromotionState.PROBED,
        store=store,
        config=PromotionConfig(require_probe=True),
        dry_run=True,
    )
    assert result["allowed"] is False
    assert "promotion without required probe" in result["blockers"]


def test_promotion_without_required_evaluation_fails(artifact_root: Path) -> None:
    spec = replace(_spec(), lifecycle=PromotionState.PROBED)
    store = PromotionStore(path=artifact_root / "state.json")
    row = store.record(spec.id)
    row["probe"] = artifact_ref(
        _probe(artifact_root / "probe.json", spec),
        passed=True,
        spec=spec,
        kind="probe",
    )
    row["identity"] = spec_identity(spec)
    result = apply_transition(
        spec,
        PromotionState.EVALUATED,
        store=store,
        config=PromotionConfig(require_probe=True, require_evaluation=True),
        dry_run=True,
    )
    assert result["allowed"] is False
    assert "promotion without required evaluation" in result["blockers"]


def test_artifact_path_cannot_escape_allowed_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(promotion, "ARTIFACT_ROOTS", (allowed.resolve(),))
    with pytest.raises(ConfigurationError, match="must live under"):
        artifact_ref(outside)


def test_probe_revision_mismatch_is_rejected(artifact_root: Path) -> None:
    spec = _spec()
    payload = json.loads(_probe(artifact_root / "probe.json", spec).read_text())
    payload["model_revision"] = "rev-b"
    (artifact_root / "probe.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="revision"):
        artifact_ref(
            artifact_root / "probe.json",
            passed=True,
            spec=spec,
            kind="probe",
        )


def test_approved_overlay_fails_closed_after_identity_change(tmp_path: Path) -> None:
    original = _spec(lifecycle=PromotionState.REGISTERED)
    store = PromotionStore(path=tmp_path / "state.json")
    row = store.record(original.id)
    row.update(
        {
            "state": "approved",
            "identity": spec_identity(original),
            "approved_tiers": ["L10"],
        }
    )
    changed = replace(original, revision="rev-b")
    registry = apply_overlay(ModelRegistry([changed]), store)
    loaded = registry.get(changed.id)
    assert loaded.lifecycle is PromotionState.REGISTERED
    assert loaded.promotion_identity_match is False
    assert loaded.promotion_evidence_bound is False
    assert loaded.approved_tiers == ()


def test_production_pinned_unapproved_model_is_rejected() -> None:
    registry = ModelRegistry([_spec(lifecycle=PromotionState.EVALUATED)])
    plan = RoutePlanner(
        registry,
        require_approved=True,
        pin_requires_approved=True,
    ).plan(RouteRequest("candidate"))
    assert plan.selected is None
    assert plan.excluded[-1].rule is ExclusionRule.NOT_APPROVED
    assert "production requires approved" in plan.excluded[-1].detail


def test_registry_approved_without_evidence_is_not_production_trust() -> None:
    registry = ModelRegistry([_spec(lifecycle=PromotionState.APPROVED)])
    plan = RoutePlanner(
        registry,
        require_approved=True,
        pin_requires_approved=True,
    ).plan(RouteRequest("candidate"))
    assert plan.selected is None
    assert plan.excluded[-1].rule is ExclusionRule.NOT_APPROVED
    assert "declaration is not evidence-bound" in plan.excluded[-1].detail


def test_approved_tiers_narrow_declared_tier_eligibility() -> None:
    spec = _spec(
        lifecycle=PromotionState.APPROVED,
        promotion_evidence_bound=True,
        approved_tiers=(ServiceTier.L10,),
    )
    registry = ModelRegistry([spec])
    planner = RoutePlanner(
        registry,
        require_approved=True,
        pin_requires_approved=True,
    )
    assert planner.plan(RouteRequest("L10")).selected_model == "candidate"
    plan = planner.plan(RouteRequest("L12"))
    assert plan.selected is None
    assert plan.excluded[-1].rule is ExclusionRule.NOT_APPROVED_FOR_TIER


def test_disable_remains_auditable_and_reversible(tmp_path: Path) -> None:
    spec = _spec(
        lifecycle=PromotionState.APPROVED,
        promotion_evidence_bound=True,
    )
    store = PromotionStore(path=tmp_path / "state.json")
    row = store.record(spec.id)
    row["state"] = "approved"
    row["identity"] = spec_identity(spec)
    result = apply_transition(
        spec,
        PromotionState.DISABLED,
        store=store,
        config=PromotionConfig(version="p1", content_hash="hash"),
        reason="rollback",
    )
    assert result["allowed"] is True
    assert store.deployments[spec.id]["state"] == "disabled"
    history = store.deployments[spec.id]["history"]
    assert history[-1]["reason"] == "rollback"
    assert history[-1]["from_state"] == "approved"
    assert history[-1]["to_state"] == "disabled"


def test_promotion_policy_hash_is_stable() -> None:
    first = PromotionConfig.from_yaml(Path("config/promotion.yaml"))
    second = PromotionConfig.from_yaml(Path("config/promotion.yaml"))
    assert first.version == "2026.08.24"
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 16


def test_evaluation_artifact_binds_deployment(artifact_root: Path) -> None:
    spec = _spec()
    other = replace(spec, id="other")
    path = _evaluation(artifact_root / "eval.json", other)
    with pytest.raises(ConfigurationError, match="no result"):
        artifact_ref(path, passed=True, spec=spec, kind="evaluation")


def test_full_lifecycle_binds_evidence_and_approved_tiers(
    artifact_root: Path,
) -> None:
    registry = ModelRegistry([_spec()])
    store = PromotionStore(path=artifact_root / "state.json")
    config = PromotionConfig(
        version="p1",
        content_hash="policy-hash",
        require_probe=True,
        require_evaluation=True,
        require_shadow_artifact=True,
    )
    probe = _probe(artifact_root / "probe.json", registry.get("candidate"))
    first = apply_transition(
        registry.get("candidate"),
        PromotionState.PROBED,
        store=store,
        config=config,
        probe=probe,
    )
    assert first["allowed"] is True
    registry = apply_overlay(ModelRegistry([_spec()]), store, config=config)

    evaluation = _evaluation(artifact_root / "eval.json", registry.get("candidate"))
    second = apply_transition(
        registry.get("candidate"),
        PromotionState.EVALUATED,
        store=store,
        config=config,
        evaluation=evaluation,
    )
    assert second["allowed"] is True
    registry = apply_overlay(ModelRegistry([_spec()]), store, config=config)

    shadow = _shadow(artifact_root / "shadow.json", registry.get("candidate"))
    third = apply_transition(
        registry.get("candidate"),
        PromotionState.SHADOW,
        store=store,
        config=config,
        shadow=shadow,
    )
    assert third["allowed"] is True
    registry = apply_overlay(ModelRegistry([_spec()]), store, config=config)

    fourth = apply_transition(
        registry.get("candidate"),
        PromotionState.APPROVED,
        store=store,
        config=config,
        approved_tiers=(ServiceTier.L10,),
        approved_workloads={"coding": (ServiceTier.L10,)},
    )
    assert fourth["allowed"] is True
    registry = apply_overlay(ModelRegistry([_spec()]), store, config=config)
    approved = registry.get("candidate")
    assert approved.lifecycle is PromotionState.APPROVED
    assert approved.promotion_evidence_bound is True
    assert approved.approved_tiers == (ServiceTier.L10,)
    assert approved.approved_workloads["coding"] == (ServiceTier.L10,)

    probe.write_text(probe.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = apply_overlay(ModelRegistry([_spec()]), store, config=config).get("candidate")
    assert tampered.lifecycle is PromotionState.APPROVED
    assert tampered.promotion_evidence_bound is False


def test_approved_state_without_bound_artifacts_is_not_evidence(
    tmp_path: Path,
) -> None:
    spec = _spec()
    store = PromotionStore(path=tmp_path / "state.json")
    store.record(spec.id).update(
        {
            "state": "approved",
            "identity": spec_identity(spec),
            "approval": {
                "approved": True,
                "policy_version": "p1",
                "policy_hash": "hash",
            },
        }
    )
    loaded = apply_overlay(
        ModelRegistry([spec]),
        store,
        config=PromotionConfig(version="p1", content_hash="hash"),
    ).get(spec.id)
    assert loaded.lifecycle is PromotionState.APPROVED
    assert loaded.promotion_evidence_bound is False
