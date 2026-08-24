"""Liveness and readiness.

Kept separate because they answer different questions:

- `/healthz` asks whether this process is alive. External dependency outages
  do not fail liveness; that would turn a database incident into a Kubernetes
  restart storm.
- `/readyz` asks whether this instance is safe to receive NEW production
  serving traffic. Mandatory serving dependencies (PostgreSQL, Redis when
  this process uses them) plus at least one servable model must be healthy.

Readiness is a signal to infrastructure. Admission control on the inference
path is the correctness boundary for requests that still arrive.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from llm_fabric import __version__
from llm_fabric.deps.health import DependencyHealth
from llm_fabric.gateway.dependencies import get_dependency_health, get_providers, get_registry
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.factory import ProviderFactory

router = APIRouter(tags=["operations"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/readyz", summary="Readiness probe")
async def readyz(
    response: Response,
    registry: ModelRegistry = Depends(get_registry),
    providers: ProviderFactory = Depends(get_providers),
    dependency_health: DependencyHealth = Depends(get_dependency_health),
) -> dict[str, object]:
    models = registry.enabled_models()
    servable = [spec for spec in models if providers.constructible(spec.provider)]
    deps_ready = dependency_health.serving_ready()
    ready = bool(servable) and deps_ready
    if not servable:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_text = "no_models_enabled" if not models else "no_servable_provider"
    elif not deps_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_text = "dependency_unhealthy"
    else:
        status_text = "ready"
    return {
        "ready": ready,
        "status": status_text,
        "enabled_models": len(models),
        "servable_models": len(servable),
        "providers": sorted({spec.provider for spec in servable}),
        "dependencies": dependency_health.public_dependencies(),
    }
