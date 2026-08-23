"""Usage and route provenance for requests this process has served.

Backed by the in-memory meter, so the numbers cover **this process since it
started** and nothing more. The response says so explicitly rather than implying
a durable, fleet-wide ledger.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from llm_fabric.gateway.dependencies import get_client_id, get_meter
from llm_fabric.observability.metering import InMemoryMeter

router = APIRouter(prefix="/v1", tags=["operations"])

_SCOPE_NOTE = "in-memory, this process only, lost on restart"


@router.get("/usage", summary="Usage totals and recent routing decisions")
async def get_usage(
    limit: int = Query(default=20, ge=1, le=200),
    meter: InMemoryMeter = Depends(get_meter),
    _client: str | None = Depends(get_client_id),
) -> dict[str, Any]:
    totals = meter.totals()
    return {
        "scope": _SCOPE_NOTE,
        "totals": {
            "requests": totals.requests,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "total_tokens": totals.total_tokens,
            "cost_usd": round(totals.cost_usd, 6),
            "requests_with_estimated_cost": totals.estimated_cost_requests,
            "failovers": totals.failovers,
        },
        "recent": [record.as_dict() for record in meter.recent(limit)],
    }
