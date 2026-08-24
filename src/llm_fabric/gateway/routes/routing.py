"""Route preview: ask where a request *would* go, without sending it.

The constitution requires that no routing decision be hidden and that each be
explainable from stored features. An explanation only reachable after paying for
inference is not much of an explanation, so the same planner that serves traffic
answers this endpoint, and returns the same decision object.

Two isolation rules hold here, and both are tested adversarially.

**The tenant comes from the token, never from the body.** A caller cannot preview
as another tenant, because there is no field in which to ask. The tenant policy
in the response is always the caller's own.

**The response is scoped to what the caller may already see.** Model ids,
capabilities and prices are fleet configuration the tenant routes across anyway.
No other tenant's policy, traffic or identity appears, and the fleet-health view
reports deployment state without attributing any request to anyone.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from llm_fabric.contract.openai import ChatMessage
from llm_fabric.errors import InvalidRequestError
from llm_fabric.gateway.dependencies import get_router, get_tenant_scope
from llm_fabric.router.capabilities import normalise
from llm_fabric.router.engine import Router
from llm_fabric.router.grades import Grade
from llm_fabric.router.plan import RouteRequest
from llm_fabric.router.policy import parse_policy
from llm_fabric.serving.tokens import approximate_prompt_tokens
from llm_fabric.tenancy.scope import TenantScope

router = APIRouter(prefix="/v1/routes", tags=["routing"])


class RoutePreviewRequest(BaseModel):
    """What to route, and the constraints to route it under.

    Either `messages` or `prompt_tokens` may be given. Messages are only measured,
    never sent anywhere: preview performs no inference.
    """

    model: str = Field(description="A model id or alias, exactly as for a completion.")
    messages: list[ChatMessage] = Field(default_factory=list)
    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Used instead of measuring `messages`."
    )
    max_tokens: int | None = Field(default=None, ge=0)
    policy: str | None = Field(default=None, description="Override the policy for this preview.")
    latency_slo_ms: float | None = Field(default=None, ge=0)
    budget_usd: float | None = Field(default=None, ge=0)
    required_capabilities: list[str] = Field(default_factory=list)
    minimum_grade: str | None = Field(default=None, description="Floor such as `Grade12` or `L12`.")
    maximum_grade: str | None = Field(
        default=None, description="Ceiling such as `Grade18` or `L18`. Cannot raise a tenant cap."
    )
    intent_id: str | None = Field(
        default=None,
        description=(
            "Explicit intent id for dry-run policy lookup. Does not run IntentOS "
            "and does not enable serving-path classification."
        ),
    )


@router.post("/preview", summary="Explain where a request would be routed")
async def preview_route(
    body: RoutePreviewRequest,
    fabric: Router = Depends(get_router),
    scope: TenantScope = Depends(get_tenant_scope),
) -> dict[str, Any]:
    prompt_tokens = (
        body.prompt_tokens
        if body.prompt_tokens is not None
        else approximate_prompt_tokens(list(body.messages))
    )

    try:
        request = RouteRequest(
            requested_model=body.model,
            tenant_id=scope.tenant_id,
            policy=parse_policy(body.policy) if body.policy else None,
            required_capabilities=normalise(body.required_capabilities),
            minimum_grade=Grade.parse(body.minimum_grade) if body.minimum_grade else None,
            maximum_grade=Grade.parse(body.maximum_grade) if body.maximum_grade else None,
            intent_id=body.intent_id,
            prompt_tokens=prompt_tokens,
            max_output_tokens=body.max_tokens,
            latency_slo_ms=body.latency_slo_ms,
            budget_usd=body.budget_usd,
        )
    except Exception as exc:  # configuration errors here are caller mistakes
        raise InvalidRequestError(str(exc)) from exc

    plan = fabric.planner.plan(request)
    payload = plan.describe()
    payload["tenant_id"] = scope.tenant_id
    payload["prompt_tokens"] = prompt_tokens
    payload["prompt_tokens_are_estimated"] = body.prompt_tokens is None
    payload["executed"] = False
    return payload


@router.get("/health", summary="Observed health of every deployment")
async def fleet_health(
    fabric: Router = Depends(get_router),
    scope: TenantScope = Depends(get_tenant_scope),
) -> dict[str, Any]:
    """Circuit state and observed rates per deployment.

    Deliberately contains no per-tenant figures. Health is a property of a
    backend, and reporting it per tenant would leak one tenant's traffic volume
    to another.
    """
    del scope  # Authentication is required; the answer is not tenant-specific.
    snapshots = fabric.health.all_snapshots()
    return {
        "deployments": [snapshot.as_dict() for snapshot in snapshots.values()],
        "observed": len(snapshots),
        "note": (
            "measured by this process from attempts it made; a deployment with no "
            "samples reports null rather than a default"
        ),
    }
