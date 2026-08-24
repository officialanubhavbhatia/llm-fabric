"""Model and provider health scores from observations this process made.

Scores are sample-weighted aggregates of `HealthTracker` snapshots. A provider
nobody has called has no score. Declared registry quality is not mixed in:
that number was typed into YAML and is not a measurement of the running fleet.
"""

from __future__ import annotations

from collections import defaultdict

from llm_fabric.heal.schema import ComponentHealth
from llm_fabric.router.health import BreakerState, HealthSnapshot, HealthTracker
from llm_fabric.router.registry import ModelRegistry, ModelSpec


def model_health(health: HealthTracker, registry: ModelRegistry) -> tuple[ComponentHealth, ...]:
    rows: list[ComponentHealth] = []
    for spec in registry.all_models():
        snapshot = health.snapshot(spec.deployment_id)
        rows.append(_from_snapshot(spec.id, "model", snapshot, (spec.deployment_id,)))
    return tuple(rows)


def provider_health(health: HealthTracker, registry: ModelRegistry) -> tuple[ComponentHealth, ...]:
    grouped: dict[str, list[tuple[ModelSpec, HealthSnapshot]]] = defaultdict(list)
    for spec in registry.all_models():
        grouped[spec.provider].append((spec, health.snapshot(spec.deployment_id)))
    rows: list[ComponentHealth] = []
    for provider, members in sorted(grouped.items()):
        snapshots = [snap for _, snap in members]
        deployments = tuple(spec.deployment_id for spec, _ in members)
        rows.append(_aggregate(provider, "provider", snapshots, deployments))
    return tuple(rows)


def _from_snapshot(
    ident: str,
    kind: str,
    snapshot: HealthSnapshot,
    deployments: tuple[str, ...],
) -> ComponentHealth:
    return ComponentHealth(
        id=ident,
        kind=kind,
        samples=snapshot.samples,
        health_score=snapshot.health_score,
        error_rate=snapshot.error_rate,
        ewma_latency_ms=snapshot.ewma_latency_ms,
        circuit_open=snapshot.state is BreakerState.OPEN,
        deployments=deployments,
    )


def _aggregate(
    ident: str,
    kind: str,
    snapshots: list[HealthSnapshot],
    deployments: tuple[str, ...],
) -> ComponentHealth:
    total = sum(snap.samples for snap in snapshots)
    if total == 0:
        return ComponentHealth(
            id=ident,
            kind=kind,
            samples=0,
            health_score=None,
            error_rate=None,
            ewma_latency_ms=None,
            circuit_open=any(snap.state is BreakerState.OPEN for snap in snapshots),
            deployments=deployments,
        )
    score_num = 0.0
    score_den = 0
    error_num = 0.0
    error_den = 0
    latency_num = 0.0
    latency_den = 0
    for snap in snapshots:
        if snap.health_score is not None:
            score_num += snap.health_score * snap.samples
            score_den += snap.samples
        if snap.error_rate is not None:
            error_num += snap.error_rate * snap.samples
            error_den += snap.samples
        if snap.ewma_latency_ms is not None:
            latency_num += snap.ewma_latency_ms * snap.samples
            latency_den += snap.samples
    return ComponentHealth(
        id=ident,
        kind=kind,
        samples=total,
        health_score=(score_num / score_den) if score_den else None,
        error_rate=(error_num / error_den) if error_den else None,
        ewma_latency_ms=(latency_num / latency_den) if latency_den else None,
        circuit_open=any(snap.state is BreakerState.OPEN for snap in snapshots),
        deployments=deployments,
    )
