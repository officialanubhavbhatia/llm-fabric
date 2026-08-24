"""`llm-fabric model` — validate, probe, and evaluate registry deployments."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from llm_fabric.config import get_settings
from llm_fabric.errors import ConfigurationError, FabricError
from llm_fabric.models.artifacts import DEFAULT_DIR, artifact_path, write_json
from llm_fabric.models.eval import evaluate_registry
from llm_fabric.models.probe import probe_deployment, probe_provider_model
from llm_fabric.router.registry import ModelRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric model",
        description="Probe and evaluate registered deployments. Does not call IntentOS.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="load a registry YAML and report errors")
    validate.add_argument("path", type=Path)

    probe = sub.add_parser("probe", help="measure what a deployment actually supports")
    probe.add_argument("deployment", nargs="?", help="Registry id, for example local-small")
    probe.add_argument("--provider", default=None)
    probe.add_argument("--model", default=None, help="Provider-native model id")
    probe.add_argument("--registry", default=None)
    probe.add_argument("--output", type=Path, default=None)

    for name in ("eval", "evaluate"):
        evaluate = sub.add_parser(name, help="score registered deployments on fabric workloads")
        evaluate.add_argument("--registry", default=None)
        evaluate.add_argument("--dataset", type=Path, default=None)
        evaluate.add_argument("--only", nargs="*", default=None)
        evaluate.add_argument("--output", type=Path, default=None)
        evaluate.add_argument(
            "--leaderboard",
            type=Path,
            default=DEFAULT_DIR / "leaderboard.json",
        )

    status = sub.add_parser("status", help="show promotion state")
    status.add_argument("deployment", nargs="?", default=None)
    status.add_argument("--registry", default=None)
    listing = sub.add_parser("list", help="list deployments and promotion state")
    listing.add_argument("--registry", default=None)

    promote = sub.add_parser("promote", help="advance lifecycle after evidence checks")
    promote.add_argument("deployment")
    promote.add_argument(
        "--to",
        required=True,
        help="probed | evaluated | shadow | approved | disabled | registered",
    )
    promote.add_argument("--registry", default=None)
    promote.add_argument("--probe-artifact", type=Path, default=None)
    promote.add_argument("--eval-artifact", type=Path, default=None)
    promote.add_argument("--shadow-artifact", type=Path, default=None)
    promote.add_argument("--approved-tiers", nargs="*", default=None)
    promote.add_argument(
        "--approved-workload",
        action="append",
        default=[],
        help="Repeatable workload=L10,L11 approval",
    )
    promote.add_argument("--policy", default="default")
    promote.add_argument("--dry-run", action="store_true")
    promote.add_argument("--force", action="store_true")
    promote.add_argument("--reason", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.path)
        if args.command == "probe":
            return asyncio.run(_probe(args))
        if args.command in {"eval", "evaluate"}:
            return asyncio.run(_eval(args))
        if args.command in {"status", "list"}:
            if args.command == "list":
                args.deployment = None
            return _status(args)
        if args.command == "promote":
            return _promote(args)
    except (ConfigurationError, FabricError) as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


def _validate(path: Path) -> int:
    registry = ModelRegistry.from_yaml(path)
    payload = {
        "path": str(path),
        "models": len(registry),
        "enabled": [spec.id for spec in registry.enabled_models()],
        "providers": sorted(registry.providers_in_use()),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _registry(path: str | None) -> ModelRegistry:
    from llm_fabric.models.promotion import load_configured_registry

    settings = get_settings()
    return load_configured_registry(
        Path(path) if path else settings.registry_path,
        promotion_state_path=settings.promotion_state_path,
        promotion_config_path=settings.promotion_config_path,
    )


async def _probe(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.provider and args.model:
        payload = await probe_provider_model(
            provider=args.provider, model=args.model, settings=settings
        )
    elif args.deployment:
        payload = await probe_deployment(
            args.deployment, registry=_registry(args.registry), settings=settings
        )
    else:
        sys.stderr.write("pass a deployment id or --provider and --model\n")
        return 2
    text = json.dumps(payload, indent=2)
    print(text)
    output = args.output
    if output is None and payload.get("status") != "unavailable":
        prefix = (
            f"{payload.get('provider') or 'provider'}-"
            f"{payload.get('deployment') or payload.get('model')}"
        )
        output = artifact_path(prefix)
    if output is not None:
        write_json(output, payload)
        sys.stderr.write(f"wrote {output}\n")
        if args.deployment and payload.get("status") != "unavailable":
            _attach_probe(args.deployment, output, args.registry)
    return 0 if payload.get("status") != "unavailable" else 1


async def _eval(args: argparse.Namespace) -> int:
    from llm_fabric.models.eval import DEFAULT_DATASET

    settings = get_settings()
    payload = await evaluate_registry(
        _registry(args.registry),
        dataset=args.dataset or DEFAULT_DATASET,
        settings=settings,
        only=args.only,
    )
    text = json.dumps(payload, indent=2)
    print(text)
    output = args.output or artifact_path("model-eval")
    write_json(output, payload)
    if args.leaderboard:
        write_json(
            args.leaderboard,
            {
                "eval_version": payload.get("eval_version"),
                "commit": payload.get("commit"),
                "timestamp": payload.get("timestamp"),
                "models": payload.get("leaderboard"),
            },
        )
    sys.stderr.write(f"wrote {output}\n")
    _attach_evaluation(payload, output, args.registry)
    return 0


def _attach_probe(deployment: str, path: Path, registry_path: str | None) -> None:
    from llm_fabric.models.promotion import (
        PromotionStore,
        artifact_ref,
        spec_identity,
    )

    settings = get_settings()
    store = PromotionStore.load(settings.promotion_state_path)
    row = store.record(deployment)
    spec = _registry(registry_path).get(deployment)
    row["probe"] = artifact_ref(path, passed=True, spec=spec, kind="probe")
    row["identity"] = spec_identity(spec)
    store.save()


def _attach_evaluation(payload: dict[str, Any], path: Path, registry_path: str | None) -> None:
    from llm_fabric.models.promotion import (
        PromotionStore,
        artifact_ref,
        spec_identity,
    )

    settings = get_settings()
    store = PromotionStore.load(settings.promotion_state_path)
    registry = _registry(registry_path)
    for result in payload.get("results") or []:
        deployment = result.get("deployment")
        if not deployment or not registry.is_model(str(deployment)):
            continue
        if result.get("status") != "ok":
            continue
        spec = registry.get(str(deployment))
        row = store.record(str(deployment))
        row["evaluation"] = artifact_ref(
            path,
            passed=True,
            spec=spec,
            kind="evaluation",
        )
        row["identity"] = spec_identity(spec)
    store.save()


def _status(args: argparse.Namespace) -> int:
    from llm_fabric.models.promotion import PromotionStore, status_payload

    settings = get_settings()
    registry = _registry(args.registry)
    store = PromotionStore.load(settings.promotion_state_path)
    if args.deployment:
        payload = status_payload(registry.get(args.deployment), store)
    else:
        payload = {"models": [status_payload(spec, store) for spec in registry.enabled_models()]}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _promote(args: argparse.Namespace) -> int:
    from llm_fabric.models.promotion import (
        PromotionConfig,
        PromotionStore,
        apply_overlay,
        apply_transition,
        write_index,
    )
    from llm_fabric.router.registry import PromotionState
    from llm_fabric.router.tiers import ServiceTier

    settings = get_settings()
    try:
        target = PromotionState(str(args.to).strip().lower())
    except ValueError:
        sys.stderr.write(f"unknown lifecycle {args.to!r}\n")
        return 2
    registry = _registry(args.registry)
    spec = registry.get(args.deployment)
    store = PromotionStore.load(settings.promotion_state_path)
    config = PromotionConfig.from_yaml(settings.promotion_config_path)
    tiers: tuple[ServiceTier, ...] = ()
    if args.approved_tiers:
        tiers = tuple(ServiceTier.parse(item) for item in args.approved_tiers)
    workloads: dict[str, tuple[ServiceTier, ...]] = {}
    for value in args.approved_workload:
        if "=" not in value:
            raise ConfigurationError("--approved-workload must use workload=L10,L11")
        name, raw_tiers = value.split("=", 1)
        workloads[name.strip()] = tuple(
            ServiceTier.parse(item.strip()) for item in raw_tiers.split(",") if item.strip()
        )
    payload = apply_transition(
        spec,
        target,
        store=store,
        config=config,
        reason=args.reason,
        probe=args.probe_artifact,
        evaluation=args.eval_artifact,
        shadow=args.shadow_artifact,
        approved_tiers=tiers,
        approved_workloads=workloads,
        policy_name=args.policy,
        force=args.force,
        dry_run=args.dry_run,
    )
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if payload.get("allowed") and not args.dry_run:
        apply_overlay(registry, store, config=config)
        write_index(store)
        return 0
    return 0 if args.dry_run else 1


def format_probe_summary(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)
