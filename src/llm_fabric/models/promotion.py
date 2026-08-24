"""Model promotion: evidence-bound lifecycle, not declared-tier trust.

States: registered → probed → evaluated → shadow → approved, plus disabled.
Public chat cannot mutate this store. Hugging Face ids are not downloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.errors import ConfigurationError
from llm_fabric.models.artifacts import DEFAULT_DIR
from llm_fabric.router.registry import ModelRegistry, ModelSpec, PromotionState
from llm_fabric.router.tiers import ServiceTier

PROMOTION_VERSION = "promotion-v1"
DEFAULT_POLICY_PATH = Path("config/promotion.yaml")
DEFAULT_STATE_PATH = DEFAULT_DIR / "promotion-state.json"
ARTIFACT_ROOTS = (DEFAULT_DIR.resolve(), Path("artifacts").resolve())

TRANSITIONS: dict[PromotionState, frozenset[PromotionState]] = {
    PromotionState.REGISTERED: frozenset({PromotionState.PROBED, PromotionState.DISABLED}),
    PromotionState.PROBED: frozenset(
        {PromotionState.EVALUATED, PromotionState.REGISTERED, PromotionState.DISABLED}
    ),
    PromotionState.EVALUATED: frozenset(
        {PromotionState.SHADOW, PromotionState.PROBED, PromotionState.DISABLED}
    ),
    PromotionState.SHADOW: frozenset(
        {PromotionState.APPROVED, PromotionState.EVALUATED, PromotionState.DISABLED}
    ),
    PromotionState.APPROVED: frozenset({PromotionState.DISABLED, PromotionState.SHADOW}),
    PromotionState.DISABLED: frozenset({PromotionState.REGISTERED}),
}

ORDER = (
    PromotionState.REGISTERED,
    PromotionState.PROBED,
    PromotionState.EVALUATED,
    PromotionState.SHADOW,
    PromotionState.APPROVED,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def safe_artifact_path(path: Path) -> Path:
    """Refuse path traversal outside eval/artifact directories."""
    resolved = path.expanduser().resolve()
    for root in ARTIFACT_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ConfigurationError(
        f"artifact '{path}' must live under datasets/eval/models or artifacts/"
    )


def spec_identity(spec: ModelSpec) -> dict[str, str | None]:
    return {
        "provider": spec.provider,
        "provider_model": spec.provider_model,
        "huggingface_id": spec.huggingface_id,
        "revision": spec.revision,
        "digest": spec.digest,
        "pool": spec.pool,
    }


def identities_match(spec: ModelSpec, recorded: dict[str, Any] | None) -> bool:
    if not recorded:
        return True
    for key in ("provider", "provider_model", "revision", "digest"):
        left = getattr(spec, key) if key != "provider_model" else spec.provider_model
        right = recorded.get(key)
        if right in (None, "") or left in (None, ""):
            continue
        if str(left) != str(right):
            return False
    return True


@dataclass(frozen=True, slots=True)
class WorkloadPolicy:
    name: str
    required_probe_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    version: str = "unversioned"
    content_hash: str = ""
    auto_requires_approved: dict[str, bool] = field(default_factory=dict)
    pin_requires_approved: dict[str, bool] = field(default_factory=dict)
    commercial_use_required: bool = False
    bind_identity: bool = True
    require_probe: bool = True
    require_evaluation: bool = True
    require_shadow_artifact: bool = False
    policies: tuple[WorkloadPolicy, ...] = ()

    def auto_requires(self, environment: str) -> bool:
        return bool(self.auto_requires_approved.get(environment, False))

    def pin_requires(self, environment: str) -> bool:
        return bool(self.pin_requires_approved.get(environment, False))

    def policy_named(self, name: str) -> WorkloadPolicy:
        for item in self.policies:
            if item.name == name:
                return item
        return WorkloadPolicy(name=name)

    @classmethod
    def empty(cls) -> PromotionConfig:
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> PromotionConfig:
        if not path.exists():
            return cls.empty()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"promotion config at {path} must be a mapping")
        configured = data.get("promotion")
        source: dict[str, Any] = configured if isinstance(configured, dict) else data
        blob = json.dumps(source, sort_keys=True, default=str, separators=(",", ":"))
        policies = []
        raw_policies = source.get("policies") or {}
        if not isinstance(raw_policies, dict):
            raise ConfigurationError("promotion.policies must be a mapping")
        for name, raw in raw_policies.items():
            block = raw if isinstance(raw, dict) else {}
            caps = block.get("required_probe_capabilities") or []
            policies.append(
                WorkloadPolicy(
                    name=str(name),
                    required_probe_capabilities=tuple(str(item) for item in caps),
                )
            )
        auto = source.get("auto_requires_approved") or {}
        pin = source.get("pin_requires_approved") or {}
        return cls(
            version=str(source.get("version") or "unversioned"),
            content_hash=sha256(blob.encode("utf-8")).hexdigest()[:16],
            auto_requires_approved={str(k): bool(v) for k, v in auto.items()}
            if isinstance(auto, dict)
            else {},
            pin_requires_approved={str(k): bool(v) for k, v in pin.items()}
            if isinstance(pin, dict)
            else {},
            commercial_use_required=bool(source.get("commercial_use_required", False)),
            bind_identity=bool(source.get("bind_identity", True)),
            require_probe=bool(source.get("require_probe", True)),
            require_evaluation=bool(source.get("require_evaluation", True)),
            require_shadow_artifact=bool(source.get("require_shadow_artifact", False)),
            policies=tuple(policies),
        )


def artifact_ref(
    path: Path | None,
    *,
    passed: bool | None = None,
    spec: ModelSpec | None = None,
    kind: str | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    safe = safe_artifact_path(path)
    if not safe.is_file():
        raise ConfigurationError(f"artifact not found: {safe}")
    try:
        document = json.loads(safe.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"artifact is not valid JSON: {safe}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(f"artifact root must be an object: {safe}")
    if spec is not None and kind is not None:
        _validate_artifact_identity(document, spec=spec, kind=kind)
    try:
        display_path = safe.relative_to(Path.cwd().resolve())
    except ValueError:
        display_path = safe
    payload: dict[str, Any] = {
        "path": str(display_path),
        "sha256": sha256_file(safe),
    }
    if passed is not None:
        payload["passed"] = passed
    payload["artifact_version"] = document.get("probe_version", document.get("eval_version"))
    payload["fabric_commit"] = document.get("commit") or (document.get("environment") or {}).get(
        "commit"
    )
    if spec is not None:
        payload["identity"] = spec_identity(spec)
    if kind == "probe":
        payload["capabilities"] = document.get("capabilities")
    return payload


def _validate_artifact_identity(document: dict[str, Any], *, spec: ModelSpec, kind: str) -> None:
    if kind == "probe":
        if document.get("deployment") != spec.id:
            raise ConfigurationError(
                f"probe artifact deployment {document.get('deployment')!r} "
                f"does not match '{spec.id}'"
            )
        if document.get("provider") != spec.provider:
            raise ConfigurationError("probe artifact provider does not match registry")
        model = document.get("model") or document.get("model_id")
        if model and str(model) != spec.provider_model:
            raise ConfigurationError("probe artifact model identity does not match registry")
        if spec.revision and document.get("model_revision") != spec.revision:
            raise ConfigurationError("probe artifact revision does not match registry")
        if spec.digest and document.get("model_digest") != spec.digest:
            raise ConfigurationError("probe artifact digest does not match registry")
        return
    if kind == "evaluation":
        result = next(
            (
                row
                for row in document.get("results") or []
                if isinstance(row, dict) and row.get("deployment") == spec.id
            ),
            None,
        )
        if result is None:
            raise ConfigurationError(
                f"evaluation artifact has no result for deployment '{spec.id}'"
            )
        if result.get("provider") != spec.provider:
            raise ConfigurationError("evaluation artifact provider does not match registry")
        identity = result.get("identity") or {}
        if identity.get("provider_model") not in (None, spec.provider_model):
            raise ConfigurationError("evaluation artifact model does not match registry")
        if spec.revision and identity.get("revision") != spec.revision:
            raise ConfigurationError("evaluation artifact revision does not match registry")
        if spec.digest and identity.get("digest") != spec.digest:
            raise ConfigurationError("evaluation artifact digest does not match registry")
        return
    if kind == "shadow":
        referenced = {
            str(value)
            for row in document.get("cases") or []
            if isinstance(row, dict)
            for value in (
                row.get("deployment"),
                row.get("selected"),
                (row.get("quality_shadow") or {}).get("shadow_selected"),
            )
            if value
        }
        top_shadow = document.get("quality_shadow") or {}
        top_selected = document.get("selected") or {}
        referenced.update(
            str(value)
            for value in (
                top_shadow.get("shadow_selected"),
                top_shadow.get("live_selected"),
                top_selected.get("model_id"),
                (document.get("candidate") or {}).get("model"),
                (document.get("actual") or {}).get("model"),
            )
            if value
        )
        if spec.id not in referenced:
            raise ConfigurationError(f"shadow artifact does not reference deployment '{spec.id}'")


@dataclass
class PromotionStore:
    version: str = PROMOTION_VERSION
    policy_version: str = "unversioned"
    policy_hash: str = ""
    deployments: dict[str, dict[str, Any]] = field(default_factory=dict)
    path: Path = DEFAULT_STATE_PATH

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> PromotionStore:
        if not path.is_file():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError(f"promotion state at {path} must be a mapping")
        deployments = raw.get("deployments") or {}
        if not isinstance(deployments, dict):
            raise ConfigurationError("promotion state deployments must be a mapping")
        return cls(
            version=str(raw.get("version") or PROMOTION_VERSION),
            policy_version=str(raw.get("policy_version") or "unversioned"),
            policy_hash=str(raw.get("policy_hash") or ""),
            deployments={str(k): dict(v) for k, v in deployments.items() if isinstance(v, dict)},
            path=path,
        )

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash or None,
            "deployments": self.deployments,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return self.path

    def record(self, deployment_id: str) -> dict[str, Any]:
        return self.deployments.setdefault(
            deployment_id,
            {
                "deployment_id": deployment_id,
                "state": PromotionState.REGISTERED.value,
                "probe": None,
                "evaluation": None,
                "shadow": None,
                "approval": {"approved": False, "approved_at": None, "policy_version": None},
                "identity": None,
                "approved_tiers": [],
                "approved_workloads": {},
                "history": [],
            },
        )


def load_configured_registry(
    registry_path: Path,
    *,
    promotion_state_path: Path | None = None,
    promotion_config_path: Path | None = None,
) -> ModelRegistry:
    registry = ModelRegistry.from_yaml(registry_path)
    if promotion_state_path and promotion_state_path.is_file():
        config = PromotionConfig.from_yaml(promotion_config_path or DEFAULT_POLICY_PATH)
        return apply_overlay(
            registry,
            PromotionStore.load(promotion_state_path),
            config=config,
        )
    return registry


def _stored_artifact_valid(reference: object) -> bool:
    if not isinstance(reference, dict):
        return False
    path = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(path, str) or not isinstance(expected_hash, str):
        return False
    try:
        resolved = safe_artifact_path(Path(path))
    except ConfigurationError:
        return False
    return resolved.is_file() and sha256_file(resolved) == expected_hash


def _approval_evidence_valid(row: dict[str, Any], config: PromotionConfig) -> bool:
    if not _probe_passed(row) or not _stored_artifact_valid(row.get("probe")):
        return False
    approval = row.get("approval") or {}
    workload = config.policy_named(str(approval.get("workload_policy") or "default"))
    capabilities = (row.get("probe") or {}).get("capabilities") or {}
    if any(
        (capabilities.get(name) or {}).get("supported") is not True
        for name in workload.required_probe_capabilities
    ):
        return False
    if config.require_evaluation and (
        not _eval_present(row) or not _stored_artifact_valid(row.get("evaluation"))
    ):
        return False
    if config.require_shadow_artifact and (
        not _shadow_present(row) or not _stored_artifact_valid(row.get("shadow"))
    ):
        return False
    return bool(approval.get("approved")) and (
        approval.get("policy_version") == config.version
        and approval.get("policy_hash") == config.content_hash
    )


def apply_overlay(
    registry: ModelRegistry,
    store: PromotionStore,
    *,
    config: PromotionConfig | None = None,
) -> ModelRegistry:
    """Merge promotion state onto registry specs. Identity mismatch fail-closes."""
    for spec in list(registry.all_models()):
        row = store.deployments.get(spec.id)
        if not row:
            continue
        try:
            state = PromotionState(str(row.get("state") or spec.lifecycle.value))
        except ValueError:
            raise ConfigurationError(
                f"promotion state for '{spec.id}' has unknown lifecycle {row.get('state')!r}"
            ) from None
        recorded_identity = row.get("identity")
        match = isinstance(recorded_identity, dict) and identities_match(spec, recorded_identity)
        if state is PromotionState.REGISTERED and recorded_identity is None:
            match = True
        promotion_config = config or PromotionConfig.from_yaml(DEFAULT_POLICY_PATH)
        evidence_bound = (
            match
            and state is PromotionState.APPROVED
            and _approval_evidence_valid(row, promotion_config)
        )
        tiers_raw = row.get("approved_tiers") or []
        approved_tiers = tuple(ServiceTier.parse(str(item)) for item in tiers_raw)
        workloads_raw = row.get("approved_workloads") or {}
        approved_workloads = {
            str(name): tuple(ServiceTier.parse(str(item)) for item in tiers)
            for name, tiers in workloads_raw.items()
            if isinstance(tiers, list)
        }
        updated = replace(
            spec,
            lifecycle=state if match else PromotionState.REGISTERED,
            approved_tiers=approved_tiers if match else (),
            approved_workloads=approved_workloads if match else {},
            promotion_identity_match=match,
            promotion_evidence_bound=evidence_bound,
        )
        registry.replace(updated)
    return registry


def _probe_passed(row: dict[str, Any]) -> bool:
    probe = row.get("probe") or {}
    return bool(probe.get("passed")) and bool(probe.get("path") or probe.get("artifact"))


def _eval_present(row: dict[str, Any]) -> bool:
    evaluation = row.get("evaluation") or {}
    return bool(evaluation.get("passed")) and bool(
        evaluation.get("path") or evaluation.get("artifact")
    )


def _shadow_present(row: dict[str, Any]) -> bool:
    shadow = row.get("shadow") or {}
    return bool(shadow.get("path") or shadow.get("artifact"))


def validate_transition(
    spec: ModelSpec,
    target: PromotionState,
    *,
    store: PromotionStore,
    config: PromotionConfig,
    policy_name: str = "default",
    force: bool = False,
    reason: str | None = None,
    probe: Path | None = None,
    evaluation: Path | None = None,
    shadow: Path | None = None,
    approved_tiers: tuple[ServiceTier, ...] = (),
    approved_workloads: dict[str, tuple[ServiceTier, ...]] | None = None,
) -> list[str]:
    """Return blocking reasons. Empty means the transition may proceed."""
    current = spec.lifecycle
    blockers: list[str] = []
    if target is current:
        blockers.append(f"already {target.value}")
        return blockers
    allowed = TRANSITIONS.get(current, frozenset())
    if target not in allowed and not force:
        blockers.append(
            f"cannot move {current.value} → {target.value}; "
            f"allowed: {sorted(item.value for item in allowed)}"
        )
    if force and not (reason or "").strip():
        blockers.append("override requires --reason")
    row = dict(store.deployments.get(spec.id) or {})
    if probe is not None:
        row["probe"] = artifact_ref(probe, passed=True, spec=spec, kind="probe")
    if evaluation is not None:
        row["evaluation"] = artifact_ref(evaluation, passed=True, spec=spec, kind="evaluation")
    if shadow is not None:
        row["shadow"] = artifact_ref(shadow, spec=spec, kind="shadow")
    if config.bind_identity and not identities_match(spec, row.get("identity")):
        blockers.append("artifact_revision_mismatch")
    if not spec.promotion_identity_match:
        blockers.append("artifact_revision_mismatch")
    if (
        target
        in {
            PromotionState.PROBED,
            PromotionState.EVALUATED,
            PromotionState.SHADOW,
            PromotionState.APPROVED,
        }
        and config.require_probe
        and not _probe_passed(row)
    ):
        blockers.append("promotion without required probe")
    workload = config.policy_named(policy_name)
    probe_capabilities = (row.get("probe") or {}).get("capabilities") or {}
    missing_probe_capabilities = [
        capability
        for capability in workload.required_probe_capabilities
        if (probe_capabilities.get(capability) or {}).get("supported") is not True
    ]
    if (
        target
        in {
            PromotionState.PROBED,
            PromotionState.EVALUATED,
            PromotionState.SHADOW,
            PromotionState.APPROVED,
        }
        and missing_probe_capabilities
    ):
        blockers.append(
            "required probe capabilities not measured as supported: "
            + ", ".join(missing_probe_capabilities)
        )
    if (
        target in {PromotionState.EVALUATED, PromotionState.SHADOW, PromotionState.APPROVED}
        and config.require_evaluation
        and not _eval_present(row)
    ):
        blockers.append("promotion without required evaluation")
    if (
        target in {PromotionState.SHADOW, PromotionState.APPROVED}
        and config.require_shadow_artifact
        and not _shadow_present(row)
    ):
        blockers.append("promotion without required shadow artifact")
    if target is PromotionState.APPROVED:
        if not approved_tiers:
            blockers.append("approval requires explicit approved tiers")
        undeclared = [tier.value for tier in approved_tiers if tier not in spec.tiers]
        if undeclared:
            blockers.append("approved tiers must be declared eligible: " + ", ".join(undeclared))
        outside_approval = [
            f"{name}:{tier.value}"
            for name, tiers in (approved_workloads or {}).items()
            for tier in tiers
            if tier not in approved_tiers
        ]
        if outside_approval:
            blockers.append(
                "workload tiers must be included in approved tiers: " + ", ".join(outside_approval)
            )
        if config.commercial_use_required and spec.commercial_use is False:
            blockers.append("commercial_use is false under a commercial-use policy")
        if config.commercial_use_required and spec.commercial_use is None:
            blockers.append("commercial_use is unknown; unknown is not inferred")
    return blockers


def apply_transition(
    spec: ModelSpec,
    target: PromotionState,
    *,
    store: PromotionStore,
    config: PromotionConfig,
    actor: str | None = None,
    reason: str | None = None,
    probe: Path | None = None,
    evaluation: Path | None = None,
    shadow: Path | None = None,
    approved_tiers: tuple[ServiceTier, ...] = (),
    approved_workloads: dict[str, tuple[ServiceTier, ...]] | None = None,
    policy_name: str = "default",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    blockers = validate_transition(
        spec,
        target,
        store=store,
        config=config,
        force=force,
        reason=reason,
        probe=probe,
        evaluation=evaluation,
        shadow=shadow,
        approved_tiers=approved_tiers,
        approved_workloads=approved_workloads,
        policy_name=policy_name,
    )
    payload: dict[str, Any] = {
        "deployment": spec.id,
        "from_state": spec.lifecycle.value,
        "to_state": target.value,
        "dry_run": dry_run,
        "blockers": blockers,
        "allowed": not blockers,
        "force": force,
        "reason": reason,
        "policy_version": config.version,
        "policy_hash": config.content_hash or None,
    }
    if blockers or dry_run:
        return payload
    row = store.record(spec.id)
    if probe is not None:
        row["probe"] = artifact_ref(probe, passed=True, spec=spec, kind="probe")
    if evaluation is not None:
        row["evaluation"] = artifact_ref(evaluation, passed=True, spec=spec, kind="evaluation")
    if shadow is not None:
        row["shadow"] = artifact_ref(shadow, spec=spec, kind="shadow")
    row["state"] = target.value
    row["identity"] = spec_identity(spec)
    if approved_tiers:
        row["approved_tiers"] = [tier.value for tier in approved_tiers]
    if approved_workloads:
        row["approved_workloads"] = {
            name: [tier.value for tier in tiers] for name, tiers in approved_workloads.items()
        }
    row["approval"] = {
        "approved": target is PromotionState.APPROVED,
        "approved_at": _utc_now() if target is PromotionState.APPROVED else None,
        "policy_version": config.version if target is PromotionState.APPROVED else None,
        "policy_hash": config.content_hash if target is PromotionState.APPROVED else None,
        "workload_policy": policy_name if target is PromotionState.APPROVED else None,
    }
    history = list(row.get("history") or [])
    history.append(
        {
            "timestamp": _utc_now(),
            "deployment": spec.id,
            "from_state": spec.lifecycle.value,
            "to_state": target.value,
            "reason": reason,
            "policy_version": config.version,
            "policy_hash": config.content_hash or None,
            "actor": actor,
            "probe": (row.get("probe") or {}).get("path"),
            "evaluation": (row.get("evaluation") or {}).get("path"),
        }
    )
    row["history"] = history
    store.policy_version = config.version
    store.policy_hash = config.content_hash
    store.save()
    payload["record"] = row
    return payload


def status_payload(spec: ModelSpec, store: PromotionStore) -> dict[str, Any]:
    row = store.deployments.get(spec.id) or {}
    production_eligible = (
        spec.enabled
        and spec.lifecycle is PromotionState.APPROVED
        and spec.promotion_identity_match
        and spec.promotion_evidence_bound
    )
    return {
        "deployment_id": spec.id,
        "state": spec.lifecycle.value,
        "production_eligible": production_eligible,
        "provider": spec.provider,
        "provider_model": spec.provider_model,
        "pool": spec.pool,
        "identity": spec_identity(spec),
        "declared_tiers": [tier.value for tier in spec.tiers],
        "approved_tiers": [tier.value for tier in spec.approved_tiers] or None,
        "approved_workloads": {
            name: [tier.value for tier in tiers] for name, tiers in spec.approved_workloads.items()
        }
        or None,
        "promotion_identity_match": spec.promotion_identity_match,
        "promotion_evidence_bound": spec.promotion_evidence_bound,
        "probe": row.get("probe"),
        "evaluation": row.get("evaluation"),
        "shadow": row.get("shadow"),
        "approval": row.get("approval"),
        "history": row.get("history") or [],
        "license": spec.license,
        "commercial_use": spec.commercial_use,
        "note": None if production_eligible else "NOT PRODUCTION ELIGIBLE",
    }


def write_index(store: PromotionStore, path: Path | None = None) -> Path:
    target = path or (DEFAULT_DIR / "index.json")
    index = {
        deployment: {
            "state": row.get("state"),
            "probe": (row.get("probe") or {}).get("path"),
            "evaluation": (row.get("evaluation") or {}).get("path"),
            "shadow": (row.get("shadow") or {}).get("path"),
            "approval": row.get("approval"),
            "identity": row.get("identity"),
        }
        for deployment, row in store.deployments.items()
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return target
