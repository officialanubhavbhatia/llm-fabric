"""`llm-fabric route explain` — plan a request without calling a provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_fabric.config import get_settings
from llm_fabric.contract.openai import ChatMessage
from llm_fabric.errors import ConfigurationError, FabricError
from llm_fabric.router.grades import Grade
from llm_fabric.router.intent_routing import RoutingConfig
from llm_fabric.router.plan import RoutePlanner, RouteRequest
from llm_fabric.router.policy import parse_policy
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.tokens import approximate_prompt_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric route",
        description="Explain or replay a routing decision without invoking inference.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    explain = sub.add_parser("explain", help="print the planner decision")
    _add_request_flags(explain)
    explain.add_argument("--json", action="store_true", help="Machine-readable JSON")

    replay = sub.add_parser("replay", help="re-plan from a saved request artifact; no inference")
    replay.add_argument("artifact", type=Path)
    replay.add_argument("--policy-file", type=Path, default=None)
    replay.add_argument("--json", action="store_true")

    simulate = sub.add_parser("simulate", help="compare live policy vs a candidate YAML")
    _add_request_flags(simulate)
    simulate.add_argument("--candidate-policy", type=Path, required=True)
    simulate.add_argument("--json", action="store_true")
    return parser


def _add_request_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default="", help="User prompt; measured, never sent.")
    parser.add_argument("--model", default="auto", help="Model id, alias, or tier such as L4.")
    parser.add_argument(
        "--intent",
        default=None,
        help="Explicit intent id for policy lookup. Does not run IntentOS.",
    )
    parser.add_argument("--max-tier", default=None, help="Ceiling such as L18.")
    parser.add_argument("--min-tier", default=None, help="Floor such as L4.")
    parser.add_argument("--policy", default=None, help="Override routing policy name.")
    parser.add_argument(
        "--registry",
        default=None,
        help="Registry YAML. Default is LLM_FABRIC_REGISTRY_PATH.",
    )
    parser.add_argument("--tenant-max-tier", default=None, help="Tenant ceiling such as L10.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "explain":
            payload = explain_route(
                prompt=args.prompt,
                model=args.model,
                intent_id=args.intent,
                max_tier=args.max_tier,
                min_tier=args.min_tier,
                policy=args.policy,
                registry_path=args.registry,
                tenant_max_tier=args.tenant_max_tier,
            )
            _emit(payload, json_mode=args.json)
            return 0
        if args.command == "replay":
            payload = replay_artifact(args.artifact, policy_file=args.policy_file)
            _emit(payload, json_mode=args.json)
            return 0
        if args.command == "simulate":
            payload = simulate_policies(
                prompt=args.prompt,
                model=args.model,
                intent_id=args.intent,
                max_tier=args.max_tier,
                min_tier=args.min_tier,
                policy=args.policy,
                registry_path=args.registry,
                tenant_max_tier=args.tenant_max_tier,
                candidate_policy=args.candidate_policy,
            )
            _emit(payload, json_mode=True)
            return 0
    except (ConfigurationError, FabricError) as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


def _planner(registry_path: str | None) -> tuple[RoutePlanner, ModelRegistry, RoutingConfig]:
    settings = get_settings()
    from llm_fabric.models.promotion import PromotionConfig, load_configured_registry

    registry = load_configured_registry(
        Path(registry_path) if registry_path else settings.registry_path,
        promotion_state_path=settings.promotion_state_path,
        promotion_config_path=settings.promotion_config_path,
    )
    routing = RoutingConfig.from_yaml(settings.routing_config_path, registry=registry)
    promotion = PromotionConfig.from_yaml(settings.promotion_config_path)
    planner = RoutePlanner(
        registry,
        routing=routing,
        quality_shadow=settings.routing_quality_shadow,
        require_approved=promotion.auto_requires(settings.environment),
        pin_requires_approved=promotion.pin_requires(settings.environment),
        commercial_use_required=promotion.commercial_use_required,
    )
    return planner, registry, routing


def _request(
    *,
    prompt: str,
    model: str,
    intent_id: str | None,
    max_tier: str | None,
    min_tier: str | None,
    policy: str | None,
) -> RouteRequest:
    tokens = approximate_prompt_tokens([ChatMessage(role="user", content=prompt)]) if prompt else 0
    return RouteRequest(
        requested_model=model,
        intent_id=intent_id,
        policy=parse_policy(policy) if policy else None,
        minimum_grade=Grade.parse(min_tier) if min_tier else None,
        maximum_grade=Grade.parse(max_tier) if max_tier else None,
        prompt_tokens=tokens,
    )


def explain_route(
    *,
    prompt: str,
    model: str,
    intent_id: str | None,
    max_tier: str | None,
    min_tier: str | None,
    policy: str | None,
    registry_path: str | None,
    tenant_max_tier: str | None = None,
) -> dict[str, Any]:
    planner, _, routing = _planner(registry_path)
    if tenant_max_tier:
        from llm_fabric.router.plan import TenantRoutingPolicies, TenantRoutingPolicy

        planner = RoutePlanner(
            planner.registry,
            health=planner.health,
            tenant_policies=TenantRoutingPolicies(
                [
                    TenantRoutingPolicy(
                        tenant_id="cli",
                        maximum_grade=Grade.parse(tenant_max_tier),
                    )
                ]
            ),
            routing=routing,
            quality_shadow=planner.quality_shadow_enabled,
        )
        request = _request(
            prompt=prompt,
            model=model,
            intent_id=intent_id,
            max_tier=max_tier,
            min_tier=min_tier,
            policy=policy,
        )
        request = RouteRequest(
            requested_model=request.requested_model,
            tenant_id="cli",
            intent_id=request.intent_id,
            policy=request.policy,
            minimum_grade=request.minimum_grade,
            maximum_grade=request.maximum_grade,
            prompt_tokens=request.prompt_tokens,
        )
    else:
        request = _request(
            prompt=prompt,
            model=model,
            intent_id=intent_id,
            max_tier=max_tier,
            min_tier=min_tier,
            policy=policy,
        )
    payload = planner.plan(request).describe()
    payload["executed"] = False
    payload["prompt"] = prompt or None
    return payload


def replay_artifact(path: Path, *, policy_file: Path | None) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    request_meta = raw.get("request") or raw
    planner, registry, routing = _planner(raw.get("registry"))
    if policy_file is not None:
        routing = RoutingConfig.from_yaml(policy_file, registry=registry)
        planner = RoutePlanner(registry, routing=routing)
    request = _request(
        prompt=str(request_meta.get("prompt") or ""),
        model=str(request_meta.get("model") or request_meta.get("requested_model") or "auto"),
        intent_id=request_meta.get("intent") or request_meta.get("intent_id"),
        max_tier=request_meta.get("max_tier") or request_meta.get("maximum_grade"),
        min_tier=request_meta.get("min_tier") or request_meta.get("minimum_grade"),
        policy=request_meta.get("policy"),
    )
    payload = planner.plan(request).describe()
    payload["executed"] = False
    payload["replayed_from"] = str(path)
    return payload


def simulate_policies(
    *,
    prompt: str,
    model: str,
    intent_id: str | None,
    max_tier: str | None,
    min_tier: str | None,
    policy: str | None,
    registry_path: str | None,
    tenant_max_tier: str | None,
    candidate_policy: Path,
) -> dict[str, Any]:
    del tenant_max_tier
    planner, registry, _routing = _planner(registry_path)
    request = _request(
        prompt=prompt,
        model=model,
        intent_id=intent_id,
        max_tier=max_tier,
        min_tier=min_tier,
        policy=policy,
    )
    candidate = RoutingConfig.from_yaml(candidate_policy, registry=registry)
    payload = planner.compare_policies(request, candidate)
    payload["prompt"] = prompt or None
    return payload


def format_explain(payload: dict[str, Any]) -> str:
    selected = payload.get("selected") or {}
    tenant = payload.get("tenant_policy") or {}
    intent_value = next(
        (
            item.get("value")
            for item in payload.get("inputs") or []
            if item.get("input") == "intent"
        ),
        "none",
    )
    lines = [
        "Request",
        f"  model: {payload.get('requested_model')}",
        f"  intent: {intent_value}",
        f"  tenant ceiling: {tenant.get('maximum_grade') or 'none'}",
        "",
        "Requirements",
    ]
    for item in payload.get("inputs") or []:
        if item.get("input") in {
            "capabilities",
            "context_tokens",
            "minimum_grade",
            "maximum_grade",
        }:
            lines.append(f"  {item.get('input')}: {item.get('value')}")
    ceiling = (
        (payload.get("tier_policy") or {}).get("maximum") or tenant.get("maximum_grade") or "none"
    )
    lines.extend(
        [
            "",
            "Tier policy",
            f"  preferred: {(payload.get('tier_policy') or {}).get('preferred') or []}",
            f"  maximum: {ceiling}",
            "",
            "Eligible",
        ]
    )
    for row in payload.get("eligible") or []:
        tiers = ",".join(row.get("tiers") or []) or "ungraded"
        lines.append(f"  {row.get('deployment'):<20} {tiers}")
    lines.extend(["", "Rejected"])
    rejected = payload.get("rejected") or payload.get("excluded") or []
    if not rejected:
        lines.append("  (none)")
    for row in rejected:
        lines.append(f"  {row.get('model_id')}")
        lines.append(f"    reason: {row.get('rule')}")
        if row.get("detail"):
            lines.append(f"    detail: {row.get('detail')}")
    lines.extend(
        [
            "",
            "Selected",
            f"  {selected.get('model_id')}",
            f"  tier: {payload.get('selected_tier')}",
            f"  provider: {selected.get('provider')}",
            f"  lifecycle: {selected.get('lifecycle')}",
            "",
            "Why",
        ]
    )
    for code in payload.get("reason_codes") or []:
        lines.append(f"  {code}")
    lines.append(f"  policy version: {payload.get('routing_policy_version')}")
    lines.append(f"  policy hash: {payload.get('routing_policy_hash')}")
    return "\n".join(lines) + "\n"


def _emit(payload: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    sys.stdout.write(format_explain(payload))


if __name__ == "__main__":
    raise SystemExit(main())
