"""Apply a remediation, or refuse it.

Learning-related changes require a passing evaluation before promotion.
Authorization and safety policy cannot be mutated from this path at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from llm_fabric.errors import ForbiddenRemediationError, InvalidRequestError
from llm_fabric.eval.gates import apply_gates, critical_failures
from llm_fabric.eval.schema import EvalGate, EvalRun
from llm_fabric.heal.controls import OperationalControls
from llm_fabric.heal.policies import classify, is_forbidden
from llm_fabric.heal.schema import (
    Incident,
    IncidentSeverity,
    LearningJob,
    LearningJobStatus,
    RemediationClass,
    RemediationKind,
    RemediationProposal,
    RemediationRecord,
)
from llm_fabric.heal.store import (
    IncidentRepository,
    LearningJobRepository,
    RemediationRepository,
)
from llm_fabric.intent.cascade import CandidateBuffer, IntentCascade
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.storage.records import PromptStatus
from llm_fabric.storage.repositories import PromptRepository
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope


class RemediationApplicator:
    def __init__(
        self,
        *,
        controls: OperationalControls,
        health: HealthTracker,
        cache: TenantScopedCache | None = None,
        prompts: PromptRepository | None = None,
        registry: ModelRegistry | None = None,
        incidents: IncidentRepository | None = None,
        remediations: RemediationRepository | None = None,
        jobs: LearningJobRepository | None = None,
        candidates: CandidateBuffer | None = None,
        install_cascade: Callable[[IntentCascade], None] | None = None,
    ) -> None:
        self._controls = controls
        self._health = health
        self._cache = cache
        self._prompts = prompts
        self._registry = registry
        self._incidents = incidents
        self._remediations = remediations
        self._jobs = jobs
        self._candidates = candidates
        self._install_cascade = install_cascade

    def apply(
        self,
        proposal: RemediationProposal,
        scope: TenantScope,
        *,
        eval_run: EvalRun | None = None,
        gates: list[EvalGate] | None = None,
    ) -> RemediationRecord:
        if is_forbidden(proposal.kind.value, proposal.target):
            raise ForbiddenRemediationError(
                "unsupervised mutation of authorization or safety policy is not permitted"
            )
        classification = classify(proposal.kind)
        learning_promote = (
            classification is RemediationClass.LEARNING
            and proposal.kind is not RemediationKind.LEARNING_JOB
        )
        if learning_promote:
            if eval_run is None:
                record = self._record(
                    scope,
                    proposal,
                    applied=False,
                    classification=classification,
                    note="learning-related change refused: no evaluation run was supplied",
                )
                return record
            if gates:
                verdicts = apply_gates(eval_run, gates)
                if critical_failures(verdicts):
                    return self._record(
                        scope,
                        proposal,
                        applied=False,
                        classification=classification,
                        note="evaluation gates failed; promotion refused",
                    )

        applied, note = self._dispatch(proposal, scope, eval_run=eval_run)
        return self._record(
            scope,
            proposal,
            applied=applied,
            classification=classification,
            note=note,
        )

    def promote_job(
        self,
        scope: TenantScope,
        job: LearningJob,
        *,
        eval_run: EvalRun,
        gates: list[EvalGate],
    ) -> LearningJob:
        """The only path that may mark a learning job promoted."""
        verdicts = apply_gates(eval_run, gates)
        if critical_failures(verdicts):
            updated = replace(
                job,
                status=LearningJobStatus.REJECTED,
                eval_run_id=eval_run.run_id,
                note="critical evaluation gate failed",
            )
        else:
            updated = replace(
                job,
                status=LearningJobStatus.PROMOTED,
                eval_run_id=eval_run.run_id,
                note="evaluation gates passed",
            )
        if self._jobs is not None:
            self._jobs.put(scope, updated)
        return updated

    def _dispatch(
        self,
        proposal: RemediationProposal,
        scope: TenantScope,
        *,
        eval_run: EvalRun | None,
    ) -> tuple[bool, str]:
        del eval_run
        kind = proposal.kind
        if kind is RemediationKind.RAISE_INCIDENT:
            return self._raise_incident(proposal, scope)
        if kind is RemediationKind.OPEN_CIRCUIT:
            self._health.force_open(proposal.target, reason=proposal.reason, hold=True)
            return True, f"circuit held open on '{proposal.target}'"
        if kind is RemediationKind.SHIFT_TRAFFIC:
            self._controls.traffic.exclude(proposal.target)
            return True, f"'{proposal.target}' excluded from the candidate set"
        if kind is RemediationKind.INVALIDATE_CACHE:
            return self._invalidate(scope)
        if kind is RemediationKind.REDUCE_CONTEXT:
            ceiling = int(proposal.parameters.get("tokens") or 2048)
            self._controls.set_context_ceiling(ceiling)
            return True, f"context ceiling set to {ceiling} tokens"
        if kind is RemediationKind.ROLLBACK_MODEL:
            return self._rollback_model(proposal.target)
        if kind is RemediationKind.ROLLBACK_PROMPT:
            return self._rollback_prompt(scope, proposal.target)
        if kind is RemediationKind.ROLLBACK_CLASSIFIER:
            return self._rollback_classifier()
        if kind is RemediationKind.LEARNING_JOB:
            return self._queue_job(proposal, scope)
        raise InvalidRequestError(f"unknown remediation '{kind}'")

    def _raise_incident(
        self, proposal: RemediationProposal, scope: TenantScope
    ) -> tuple[bool, str]:
        raw = str(proposal.parameters.get("severity") or "warning")
        try:
            severity = IncidentSeverity(raw)
        except ValueError:
            severity = IncidentSeverity.WARNING
        incident = Incident(
            tenant_id=scope.tenant_id,
            title=proposal.reason[:120],
            severity=severity,
            summary=proposal.reason,
        )
        if self._incidents is not None:
            self._incidents.put(scope, incident)
        return True, f"incident {incident.incident_id}"

    def _invalidate(self, scope: TenantScope) -> tuple[bool, str]:
        if self._cache is None:
            return False, "no cache is configured"
        removed = self._cache.invalidate_tenant(scope)
        cascade = self._controls.classifiers.current()
        extra = 0
        if cascade is not None:
            extra += cascade.exact_cache.invalidate_tenant(scope)
            if cascade.semantic_cache is not None:
                extra += cascade.semantic_cache.invalidate_tenant(scope)
        return True, f"invalidated {removed + extra} cache entries"

    def _rollback_model(self, model_id: str) -> tuple[bool, str]:
        if self._registry is None:
            return False, "no registry is configured"
        previous = self._controls.models.take_previous(model_id)
        if previous is None:
            return False, f"no prior spec for '{model_id}'"
        self._registry.replace(previous)
        return True, f"restored '{model_id}' to a prior spec"

    def _rollback_prompt(self, scope: TenantScope, prompt_id: str) -> tuple[bool, str]:
        if self._prompts is None:
            return False, "no prompt store is configured"
        versions = [
            item for item in self._prompts.list(scope, limit=500) if item.prompt_id == prompt_id
        ]
        versions.sort(key=lambda item: item.version)
        production = [item for item in versions if item.status is PromptStatus.PRODUCTION]
        if len(production) < 1:
            return False, f"no production version of '{prompt_id}'"
        current = production[-1]
        prior = None
        for item in reversed(versions):
            if item.version < current.version and item.status in {
                PromptStatus.PRODUCTION,
                PromptStatus.RETIRED,
                PromptStatus.EVALUATED,
                PromptStatus.CANARY,
            }:
                prior = item
                break
        if prior is None:
            return False, f"no earlier published version of '{prompt_id}'"
        self._prompts.promote(scope, current.prompt_id, current.version, PromptStatus.RETIRED)
        self._prompts.promote(scope, prior.prompt_id, prior.version, PromptStatus.PRODUCTION)
        return True, f"prompt '{prompt_id}' rolled back to version {prior.version}"

    def _rollback_classifier(self) -> tuple[bool, str]:
        restored = self._controls.classifiers.rollback()
        if restored is None:
            return False, "no prior classifier pin; refusing to invent one"
        if self._install_cascade is not None:
            self._install_cascade(restored)
        return True, f"classifier restored to {restored.version}"

    def _queue_job(self, proposal: RemediationProposal, scope: TenantScope) -> tuple[bool, str]:
        count = len(self._candidates.items) if self._candidates is not None else 0
        job = LearningJob(
            tenant_id=scope.tenant_id,
            reason=proposal.reason,
            candidate_count=count,
            note="proposed only; will not train or promote without an evaluation",
        )
        if self._jobs is not None:
            self._jobs.put(scope, job)
        return True, f"learning job {job.job_id} proposed ({count} candidates)"

    def _record(
        self,
        scope: TenantScope,
        proposal: RemediationProposal,
        *,
        applied: bool,
        classification: RemediationClass,
        note: str | None,
    ) -> RemediationRecord:
        record = RemediationRecord(
            tenant_id=scope.tenant_id,
            kind=proposal.kind,
            target=proposal.target,
            applied=applied,
            reason=proposal.reason,
            classification=classification,
            note=note,
            parameters=dict(proposal.parameters),
        )
        if self._remediations is not None:
            self._remediations.put(scope, record)
        return record


def proposal_from_dict(payload: dict[str, Any]) -> RemediationProposal:
    return RemediationProposal(
        kind=RemediationKind(payload["kind"]),
        target=str(payload.get("target") or ""),
        reason=str(payload.get("reason") or ""),
        classification=RemediationClass(payload.get("classification") or "operational"),
        auto=bool(payload.get("auto", False)),
        parameters=dict(payload.get("parameters") or {}),
    )
