"""Usage and route provenance for the calling tenant.

When PostgreSQL is configured, totals come from the durable invocation ledger
and are independent of which worker handled the request. The in-memory meter
is the fallback for tests and processes without a database; the response
`scope` field says which one is in use.

Request-level `totals` match the OpenAI-compatible `usage` object on the
visible response (final model). `invocations` is every provider attempt,
including fallbacks.

Records are filtered to the caller's tenant. Fleet aggregation requires
`fabric:observe`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from llm_fabric.gateway.dependencies import (
    get_meter,
    get_optional_principal,
    get_quota,
    get_tenant_scope,
)
from llm_fabric.identity.claims import Principal
from llm_fabric.observability.metering import UsageMeter
from llm_fabric.tenancy.quota import QuotaLedger, QuotaPolicy
from llm_fabric.tenancy.scope import TenantScope

router = APIRouter(prefix="/v1", tags=["operations"])


def _quota_view(policy: QuotaPolicy, used: dict[str, Any]) -> dict[str, Any]:
    return {
        "limits": {
            "requests_per_minute": policy.requests_per_minute,
            "requests_per_day": policy.requests_per_day,
            "tokens_per_day": policy.tokens_per_day,
            "cost_per_day_usd": policy.cost_per_day_usd,
        },
        "used": used,
    }


@router.get("/usage", summary="Usage totals and recent routing decisions")
async def get_usage(
    limit: int = Query(default=20, ge=1, le=200),
    meter: UsageMeter = Depends(get_meter),
    scope: TenantScope = Depends(get_tenant_scope),
    quota: QuotaLedger = Depends(get_quota),
    principal: Principal | None = Depends(get_optional_principal),
) -> dict[str, Any]:
    fleet = principal.may_observe_fleet if principal is not None else False
    tenant_id = None if fleet else scope.tenant_id
    totals = meter.totals(tenant_id=tenant_id, observe=fleet)
    invocations = meter.invocation_totals(tenant_id=tenant_id, observe=fleet)
    tenant_usage = quota.snapshot(scope, "tenant")
    user_usage = quota.snapshot(scope, "user")
    events = meter.recent_events(limit=limit, tenant_id=tenant_id, observe=fleet)

    return {
        "scope": meter.scope_note(fleet=fleet, tenant_id=scope.tenant_id),
        "tenant_id": scope.tenant_id,
        "totals": {
            "requests": totals.requests,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "total_tokens": totals.total_tokens,
            "cost_usd": round(totals.cost_usd, 6),
            "requests_with_estimated_cost": totals.estimated_cost_requests,
            "failovers": totals.failovers,
        },
        "invocations": {
            "count": invocations.invocations,
            "requests": invocations.requests,
            "prompt_tokens": invocations.prompt_tokens,
            "completion_tokens": invocations.completion_tokens,
            "total_tokens": invocations.total_tokens,
            "provider_cost_usd": round(invocations.provider_cost_usd, 6),
            "compute_cost_estimate_usd": (
                round(invocations.compute_cost_estimate_usd, 6)
                if invocations.compute_cost_estimate_usd is not None
                else None
            ),
            "by_token_source": invocations.by_token_source,
            "estimated_invocations": invocations.estimated_invocations,
            "unavailable_invocations": invocations.unavailable_invocations,
        },
        "quota": {
            "tenant": _quota_view(
                tenant_usage.policy,
                {
                    "requests_this_minute": tenant_usage.requests_this_minute,
                    "requests_today": tenant_usage.requests_today,
                    "tokens_today": tenant_usage.tokens_today,
                    "cost_today_usd": round(tenant_usage.cost_today_usd, 6),
                },
            ),
            "user": _quota_view(
                user_usage.policy,
                {
                    "requests_this_minute": user_usage.requests_this_minute,
                    "requests_today": user_usage.requests_today,
                    "tokens_today": user_usage.tokens_today,
                    "cost_today_usd": round(user_usage.cost_today_usd, 6),
                },
            ),
        },
        "recent": [
            record.as_dict() for record in meter.recent(limit, tenant_id=tenant_id, observe=fleet)
        ],
        "recent_invocations": [event.as_dict() for event in events],
    }
