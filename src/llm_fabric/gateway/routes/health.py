"""Liveness and readiness.

Kept separate because they answer different questions: liveness asks whether the
process is running, readiness asks whether it can actually serve a request. A
fabric with an empty registry is alive but not ready, and conflating the two
would keep traffic flowing to a gateway that can serve nothing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from llm_fabric import __version__
from llm_fabric.gateway.dependencies import get_registry
from llm_fabric.router.registry import ModelRegistry

router = APIRouter(tags=["operations"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/readyz", summary="Readiness probe")
async def readyz(
    response: Response,
    registry: ModelRegistry = Depends(get_registry),
) -> dict[str, object]:
    models = registry.enabled_models()
    ready = bool(models)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if ready else "no_models_enabled",
        "enabled_models": len(models),
        "providers": sorted(registry.providers_in_use()),
    }
