"""Which remediations a drift signal may trigger, and which it may never.

Authorization and safety policy are not on this list. The constitution forbids
disabling them to restore availability, and unsupervised mutation of either is
refused even when an operator names the action explicitly.
"""

from __future__ import annotations

from llm_fabric.heal.schema import (
    FORBIDDEN_KINDS,
    FORBIDDEN_TARGETS,
    LEARNING_KINDS,
    ComponentHealth,
    DriftKind,
    DriftReport,
    DriftSeverity,
    RemediationClass,
    RemediationKind,
    RemediationProposal,
)

#: Error-rate floor that may auto-open a breaker. A judgement, not a fitted value.
AUTO_OPEN_ERROR_RATE = 0.5
AUTO_OPEN_MIN_SAMPLES = 10


def classify(kind: RemediationKind) -> RemediationClass:
    if kind in LEARNING_KINDS:
        return RemediationClass.LEARNING
    if kind.value in FORBIDDEN_KINDS:
        return RemediationClass.FORBIDDEN
    return RemediationClass.OPERATIONAL


def is_forbidden(kind: str, target: str) -> bool:
    needle = f"{kind} {target}".lower()
    if kind.lower() in FORBIDDEN_KINDS:
        return True
    return any(token in needle for token in FORBIDDEN_TARGETS)


def propose(report: DriftReport) -> list[RemediationProposal]:
    """Map observed drift and health onto remediations.

    Significant drift always raises an incident. Learning-related actions are
    proposed as jobs, never as auto-promotions. Operational auto-apply is
    limited to opening a breaker on a measured-unhealthy deployment.
    """
    proposals: list[RemediationProposal] = []
    seen: set[tuple[RemediationKind, str]] = set()

    def add(proposal: RemediationProposal) -> None:
        key = (proposal.kind, proposal.target)
        if key in seen:
            return
        seen.add(key)
        proposals.append(proposal)

    significant = report.significant()
    if significant:
        add(
            RemediationProposal(
                kind=RemediationKind.RAISE_INCIDENT,
                target=report.tenant_id,
                reason="; ".join(f"{s.kind.value}:{s.metric}" for s in significant),
                classification=RemediationClass.OPERATIONAL,
                auto=True,
                parameters={"severity": "critical"},
            )
        )

    kinds = {signal.kind for signal in significant}
    if DriftKind.CLASSIFIER in kinds or DriftKind.INTENT in kinds:
        add(
            RemediationProposal(
                kind=RemediationKind.LEARNING_JOB,
                target="classifier",
                reason="classifier or intent distribution drifted; do not retrain from this alone",
                classification=RemediationClass.LEARNING,
                auto=False,
            )
        )
        add(
            RemediationProposal(
                kind=RemediationKind.INVALIDATE_CACHE,
                target="intent",
                reason="stale intent cache is a plausible cause of classifier/intent drift",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )
        add(
            RemediationProposal(
                kind=RemediationKind.ROLLBACK_CLASSIFIER,
                target="classifier",
                reason="rollback requires a previously pinned classifier revision",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )

    if DriftKind.ROUTING in kinds:
        add(
            RemediationProposal(
                kind=RemediationKind.SHIFT_TRAFFIC,
                target="routing",
                reason="route mix shifted; traffic can be moved off a named model",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )

    if DriftKind.LATENCY in kinds:
        add(
            RemediationProposal(
                kind=RemediationKind.REDUCE_CONTEXT,
                target="context",
                reason="latency drifted; a context ceiling is an operational cap, not a compiler",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )
        add(
            RemediationProposal(
                kind=RemediationKind.SHIFT_TRAFFIC,
                target="routing",
                reason="latency drifted; traffic can be moved off the slow deployment",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )

    if DriftKind.ERROR in kinds:
        add(
            RemediationProposal(
                kind=RemediationKind.OPEN_CIRCUIT,
                target="unhealthy",
                reason="error rate drifted; open the breaker on a measured-unhealthy deployment",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )

    if DriftKind.COST in kinds:
        add(
            RemediationProposal(
                kind=RemediationKind.SHIFT_TRAFFIC,
                target="routing",
                reason="measured cost drifted; traffic can be moved toward a cheaper model",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )

    if DriftKind.QUALITY in kinds:
        add(
            RemediationProposal(
                kind=RemediationKind.ROLLBACK_MODEL,
                target="model",
                reason="eval quality dropped; rollback needs a stored prior spec",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )
        add(
            RemediationProposal(
                kind=RemediationKind.ROLLBACK_PROMPT,
                target="prompt",
                reason="eval quality dropped; rollback needs a prior production version",
                classification=RemediationClass.OPERATIONAL,
                auto=False,
            )
        )

    for row in report.model_health:
        if _should_trip(row):
            add(
                RemediationProposal(
                    kind=RemediationKind.OPEN_CIRCUIT,
                    target=row.deployments[0] if row.deployments else row.id,
                    reason=(
                        f"model '{row.id}' health_score="
                        f"{row.health_score} error_rate={row.error_rate}"
                    ),
                    classification=RemediationClass.OPERATIONAL,
                    auto=True,
                )
            )
            add(
                RemediationProposal(
                    kind=RemediationKind.SHIFT_TRAFFIC,
                    target=row.id,
                    reason=f"remove '{row.id}' from the candidate set while the breaker is held",
                    classification=RemediationClass.OPERATIONAL,
                    auto=True,
                )
            )

    if not proposals and any(s.severity is DriftSeverity.MODERATE for s in report.signals):
        add(
            RemediationProposal(
                kind=RemediationKind.RAISE_INCIDENT,
                target=report.tenant_id,
                reason="moderate drift observed; no automatic mutation",
                classification=RemediationClass.OPERATIONAL,
                auto=True,
                parameters={"severity": "warning"},
            )
        )
    return proposals


def _should_trip(row: ComponentHealth) -> bool:
    if row.samples < AUTO_OPEN_MIN_SAMPLES:
        return False
    if row.circuit_open:
        return False
    if row.error_rate is not None and row.error_rate >= AUTO_OPEN_ERROR_RATE:
        return True
    return row.health_score is not None and row.health_score == 0.0
