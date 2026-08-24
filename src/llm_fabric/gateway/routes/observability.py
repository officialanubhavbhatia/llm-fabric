"""Command Center API and Prometheus scrape endpoint.

Dashboards are tenant-scoped unless the caller holds `fabric:observe` or an
operator role. Prometheus `/metrics` is public: its labels are a closed set and
carry no tenant, user or request identity.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse

from llm_fabric.config import Settings
from llm_fabric.errors import InvalidRequestError, ResourceNotFoundError
from llm_fabric.gateway.dependencies import (
    get_intent_cascade,
    get_meter,
    get_optional_principal,
    get_registry,
    get_router,
    get_settings,
    get_stores,
    get_telemetry,
    get_tenant_scope,
)
from llm_fabric.identity.claims import Principal
from llm_fabric.intent.cascade import IntentCascade
from llm_fabric.observability.dashboards import VIEWS, DashboardAssembler
from llm_fabric.observability.metering import UsageMeter
from llm_fabric.observability.telemetry import Telemetry
from llm_fabric.router.engine import Router
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.storage.repositories import TenantStores
from llm_fabric.tenancy.scope import TenantScope

router = APIRouter(tags=["observability"])


def _meter_scope(meter: UsageMeter, *, fleet: bool, tenant_id: str) -> str:
    return meter.scope_note(fleet=fleet, tenant_id=tenant_id)


def _trace_scope(*, fleet: bool, tenant_id: str) -> str:
    suffix = "local-pod diagnostic only, not authoritative fleet trace history"
    if fleet:
        return f"fleet-wide, this process only, lost on restart; {suffix}"
    return f"tenant '{tenant_id}', this process only, lost on restart; {suffix}"


@router.get("/metrics", summary="Prometheus scrape")
async def prometheus_metrics(telemetry: Telemetry = Depends(get_telemetry)) -> PlainTextResponse:
    return PlainTextResponse(
        telemetry.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/command-center", response_class=HTMLResponse, summary="MyVista Command Center")
async def command_center() -> HTMLResponse:
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "static" / "command_center.html"
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/v1/observability/dashboards", summary="List Command Center views")
async def list_dashboards() -> dict[str, Any]:
    return {"views": list(VIEWS)}


@router.get("/v1/observability/dashboards/{view}", summary="One Command Center view")
async def get_dashboard(
    view: str,
    meter: UsageMeter = Depends(get_meter),
    telemetry: Telemetry = Depends(get_telemetry),
    fabric: Router = Depends(get_router),
    registry: ModelRegistry = Depends(get_registry),
    scope: TenantScope = Depends(get_tenant_scope),
    principal: Principal | None = Depends(get_optional_principal),
    cascade: IntentCascade | None = Depends(get_intent_cascade),
    stores: TenantStores = Depends(get_stores),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if view not in VIEWS:
        raise InvalidRequestError(f"unknown dashboard '{view}'")

    fleet = principal.may_observe_fleet if principal is not None else False
    if cascade is not None:
        intent_snapshot = dict(cascade.metrics.snapshot())
        intent_snapshot["classifier_version"] = cascade.version
        intent_snapshot["taxonomy_version"] = cascade.taxonomy.version
    else:
        intent_snapshot = {
            "classifications": 0,
            "serving_requests": 0,
            "known": 0,
            "unknown": 0,
            "abstentions": 0,
            "safe_fallback": 0,
            "errors": 0,
            "missing": 0,
        }
    intent_snapshot["routing_enabled"] = settings.intent_routing_enabled
    intent_snapshot["cascade_present"] = cascade is not None
    assembler = DashboardAssembler(
        meter=meter,
        journal=telemetry.tracer.journal,
        health=fabric.health,
        registry=registry,
        engines=telemetry.engines,
        intent_snapshot=intent_snapshot,
        eval_runs=stores.eval_runs.list(scope, limit=50),
        incidents=stores.incidents.list(scope, limit=50),
        remediations=stores.remediations.list(scope, limit=50),
        promotion_state_path=settings.promotion_state_path,
        context_records=telemetry.recent_context(),
    )
    return assembler.render(
        view,
        tenant_id=scope.tenant_id,
        fleet=fleet,
        scope_note=_meter_scope(meter, fleet=fleet, tenant_id=scope.tenant_id),
    )


@router.get("/v1/observability/traces", summary="Recent request traces")
async def list_traces(
    telemetry: Telemetry = Depends(get_telemetry),
    scope: TenantScope = Depends(get_tenant_scope),
    principal: Principal | None = Depends(get_optional_principal),
) -> dict[str, Any]:
    fleet = principal.may_observe_fleet if principal is not None else False
    return {
        "scope": _trace_scope(fleet=fleet, tenant_id=scope.tenant_id),
        "traces": telemetry.tracer.journal.traces(
            limit=50, tenant_id=None if fleet else scope.tenant_id
        ),
    }


@router.get("/v1/observability/traces/{trace_id}", summary="One request trace")
async def get_trace(
    trace_id: str,
    telemetry: Telemetry = Depends(get_telemetry),
    scope: TenantScope = Depends(get_tenant_scope),
    principal: Principal | None = Depends(get_optional_principal),
) -> dict[str, Any]:
    fleet = principal.may_observe_fleet if principal is not None else False
    tenant_id = None if fleet else scope.tenant_id
    for tree in telemetry.tracer.journal.traces(limit=200, tenant_id=tenant_id):
        if tree["trace_id"] == trace_id:
            return {"scope": _trace_scope(fleet=fleet, tenant_id=scope.tenant_id), **tree}
    raise ResourceNotFoundError("trace not found")
