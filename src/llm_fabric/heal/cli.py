"""`llm-fabric-heal` — analyze a usage dump and print drift plus proposals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_fabric.heal.controls import OperationalControls
from llm_fabric.heal.drift import usage_from_dicts
from llm_fabric.heal.engine import HealController
from llm_fabric.heal.policies import propose
from llm_fabric.router.health import HealthTracker
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.tenancy.scope import TenantScope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-heal",
        description="Analyze drift and print remediations. Does not mutate a live process.",
    )
    parser.add_argument("--records", type=Path, required=True, help="JSON array of usage records")
    parser.add_argument("--registry", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--tenant", default="public")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = json.loads(args.records.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("records file must be a JSON array", file=sys.stderr)
        return 2
    records = usage_from_dicts(raw)
    registry = ModelRegistry.from_yaml(args.registry)
    controller = HealController(
        controls=OperationalControls(),
        health=HealthTracker(),
        registry=registry,
    )
    scope = TenantScope(tenant_id=args.tenant, user_id="heal")
    report = controller.analyze(records, scope, min_samples=args.min_samples)
    payload = {
        "report": report.as_dict(),
        "proposals": [item.as_dict() for item in propose(report)],
        "note": (
            "This CLI does not apply remediations to a running gateway. "
            "Learning jobs are proposals; they are not trained from this command."
        ),
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
