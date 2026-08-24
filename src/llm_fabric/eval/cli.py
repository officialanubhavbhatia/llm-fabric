"""`llm-fabric-eval` — run suites, compare experiments, enforce gates."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_fabric.eval.compare import compare_runs
from llm_fabric.eval.gates import apply_gates, critical_failures
from llm_fabric.eval.runner import load_suite, run_suite
from llm_fabric.eval.schema import EvalGate, EvalRun, GateDirection, GateKind
from llm_fabric.gateway.app import create_app
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.tenancy.cache import TenantScopedCache
from llm_fabric.tenancy.scope import TenantScope

DEFAULT_SUITE = Path("datasets/eval/ci-suite.yaml")
DEFAULT_BASELINE = Path("datasets/eval/baseline.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-eval",
        description="Run evaluation suites and enforce deployment gates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute a suite and print the run")
    run.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run.add_argument("--output", type=Path)
    run.add_argument("--tenant", default="public")

    compare = sub.add_parser("compare", help="baseline vs candidate JSON runs")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)

    gate = sub.add_parser("gate", help="fail if a critical metric regresses")
    gate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    gate.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    gate.add_argument("--gates", type=Path, help="override gates YAML; default is the suite file")
    gate.add_argument("--output", type=Path)
    gate.add_argument("--tenant", default="public")
    return parser


def load_gates(path: Path) -> list[EvalGate]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gates = raw.get("gates") or []
    parsed: list[EvalGate] = []
    for item in gates:
        parsed.append(
            EvalGate(
                metric=str(item["metric"]),
                direction=GateDirection(item.get("direction") or "higher"),
                kind=GateKind(item["kind"]) if item.get("kind") else None,
                minimum=item.get("minimum"),
                maximum=item.get("maximum"),
                max_degradation=item.get("max_degradation"),
                critical=bool(item.get("critical", True)),
                description=str(item.get("description") or ""),
            )
        )
    return parsed


def run_from_dict(payload: dict[str, Any]) -> EvalRun:
    """Rehydrate a stored run enough for comparison and gates.

    Example rows are dropped: gates only need aggregated metrics and provenance.
    """
    from llm_fabric.eval.schema import (
        EvalProvenance,
        EvalResult,
        EvaluatorKind,
    )

    provenance = payload["provenance"]
    results = tuple(
        EvalResult(
            task=str(row["task"]),
            evaluator=EvaluatorKind(row["evaluator"]),
            metrics=dict(row.get("metrics") or {}),
            unavailable=tuple(row.get("unavailable") or ()),
            note=row.get("note"),
        )
        for row in payload.get("results") or ()
    )
    return EvalRun(
        tenant_id=str(payload.get("tenant_id") or "public"),
        suite_name=str(payload.get("suite_name") or ""),
        provenance=EvalProvenance(
            dataset_version=str(provenance["dataset_version"]),
            metric_version=str(provenance.get("metric_version") or ""),
            configuration=dict(provenance.get("configuration") or {}),
            timestamp=float(provenance.get("timestamp") or 0),
            commit=provenance.get("commit"),
            model=provenance.get("model"),
            model_version=provenance.get("model_version"),
            prompt_version=provenance.get("prompt_version"),
            taxonomy_version=provenance.get("taxonomy_version"),
        ),
        results=results,
        run_id=str(payload.get("run_id") or "rehydrated"),
        created_at=float(payload.get("created_at") or 0),
    )


async def _execute(suite_path: Path, tenant: str) -> EvalRun:
    suite = load_suite(suite_path)
    app = create_app()
    cascade = app.state.intent or build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
    scope = TenantScope(tenant_id=tenant, user_id="eval")
    return await run_suite(
        suite,
        scope=scope,
        router=app.state.router,
        cascade=cascade,
        repo=Path.cwd(),
    )


def main(argv: list[str] | None = None) -> int:
    environment = (os.environ.get("LLM_FABRIC_ENVIRONMENT") or "").strip()
    if not environment:
        print(
            "LLM_FABRIC_ENVIRONMENT is required; set it to development, test, "
            "or production before running evaluations",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("SKIP_EVALS") or os.environ.get("LLM_FABRIC_SKIP_EVALS"):
        print(
            "SKIP_EVALS is not a valid way to pass a release gate. "
            "Override a failed gate with an audited administrative action, "
            "not an environment variable.",
            file=sys.stderr,
        )
        return 2
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run = asyncio.run(_execute(args.suite, args.tenant))
        payload = run.as_dict()
        text = json.dumps(payload, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.command == "compare":
        baseline = run_from_dict(json.loads(args.baseline.read_text(encoding="utf-8")))
        candidate = run_from_dict(json.loads(args.candidate.read_text(encoding="utf-8")))
        comparison = compare_runs(baseline, candidate)
        text = json.dumps(comparison.as_dict(), indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    suite_path: Path = args.suite
    gates_path: Path = args.gates or suite_path
    candidate = asyncio.run(_execute(suite_path, args.tenant))
    baseline = run_from_dict(json.loads(args.baseline.read_text(encoding="utf-8")))
    comparison = compare_runs(baseline, candidate)
    verdicts = apply_gates(candidate, load_gates(gates_path), comparison=comparison)
    failures = critical_failures(verdicts)
    report = {
        "environment": environment,
        "dataset": str(suite_path),
        "baseline": str(args.baseline),
        "candidate": candidate.as_dict(),
        "comparison": comparison.as_dict(),
        "verdicts": [verdict.as_dict() for verdict in verdicts],
        "failed": [verdict.as_dict() for verdict in failures],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if failures:
        print(
            f"{len(failures)} critical gate(s) failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
