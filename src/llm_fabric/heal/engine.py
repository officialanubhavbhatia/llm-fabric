"""Analyze drift, propose remediations, apply those the policy allows to auto-run."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from llm_fabric.eval.schema import EvalRun
from llm_fabric.heal.controls import OperationalControls
from llm_fabric.heal.drift import analyze
from llm_fabric.heal.policies import propose
from llm_fabric.heal.remediate import RemediationApplicator
from llm_fabric.heal.schema import DriftBaseline, DriftReport, RemediationRecord
from llm_fabric.heal.store import (
    DriftBaselineRepository,
    IncidentRepository,
    LearningJobRepository,
    RemediationRepository,
)
from llm_fabric.intent.cascade import CandidateBuffer, IntentCascade
from llm_fabric.observability.metering import UsageRecord
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.storage.repositories import PromptRepository
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope


class HealController:
    def __init__(
        self,
        *,
        controls: OperationalControls,
        health: HealthTracker,
        registry: ModelRegistry,
        cache: TenantScopedCache | None = None,
        prompts: PromptRepository | None = None,
        incidents: IncidentRepository | None = None,
        remediations: RemediationRepository | None = None,
        jobs: LearningJobRepository | None = None,
        baselines: DriftBaselineRepository | None = None,
        candidates: CandidateBuffer | None = None,
        install_cascade: Callable[[IntentCascade], None] | None = None,
    ) -> None:
        self.controls = controls
        self._health = health
        self._registry = registry
        self._baselines = baselines
        self.applicator = RemediationApplicator(
            controls=controls,
            health=health,
            cache=cache,
            prompts=prompts,
            registry=registry,
            incidents=incidents,
            remediations=remediations,
            jobs=jobs,
            candidates=candidates,
            install_cascade=install_cascade,
        )

    def analyze(
        self,
        records: Sequence[UsageRecord],
        scope: TenantScope,
        *,
        eval_runs: Sequence[EvalRun] = (),
        baseline: DriftBaseline | None = None,
        min_samples: int = 20,
    ) -> DriftReport:
        return analyze(
            records,
            tenant_id=scope.tenant_id,
            health=self._health,
            registry=self._registry,
            eval_runs=eval_runs,
            baseline=baseline,
            min_samples=min_samples,
        )

    def tick(
        self,
        records: Sequence[UsageRecord],
        scope: TenantScope,
        *,
        eval_runs: Sequence[EvalRun] = (),
        baseline: DriftBaseline | None = None,
        auto_apply: bool = True,
        min_samples: int = 20,
    ) -> tuple[DriftReport, list[RemediationRecord]]:
        """Analyze and apply only remediations the policy marked `auto`.

        Learning jobs are queued as proposals. They are not trained and not
        promoted from this method.
        """
        report = self.analyze(
            records, scope, eval_runs=eval_runs, baseline=baseline, min_samples=min_samples
        )
        applied: list[RemediationRecord] = []
        if not auto_apply:
            return report, applied
        for proposal in propose(report):
            if not proposal.auto:
                continue
            applied.append(self.applicator.apply(proposal, scope))
        return report, applied
