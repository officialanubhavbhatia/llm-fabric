"""`llm-fabric-bench` — run the intent benchmark from a terminal.

Offline by default. With no provider configured it exercises L0 through L3, so
a run costs nothing, needs no network and produces the same numbers twice. The
model-backed layers are opt-in through `--provider`, because a benchmark that
silently spends money is a benchmark nobody runs.

The exit code is a gate, not a grade: non-zero when a `--min-*` threshold you
asked for was missed, or when the run could not be completed. Passing means
"this did not regress past the bar you set", never "this classifier is good".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from llm_fabric.config import Settings
from llm_fabric.errors import FabricError, ModelNotFoundError
from llm_fabric.intent.benchmark import (
    BenchmarkReport,
    load_dataset,
    render_text,
    run_benchmark,
    summarise_dataset,
)
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.cache import SemanticCachePolicy
from llm_fabric.intent.cascade import CascadeThresholds, IntentCascade
from llm_fabric.intent.classifiers.embedding import PrototypeKind
from llm_fabric.intent.classifiers.structured import ClassifierPricing
from llm_fabric.intent.embeddings import resolve_embedder
from llm_fabric.intent.factory import build_full_cascade, build_offline_cascade
from llm_fabric.router.registry import ModelRegistry
from llm_fabric.serving.factory import ProviderFactory
from llm_fabric.tenancy.cache import TenantScopedCache

DEFAULT_DATASET = Path("datasets/intent/bootstrap.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-bench",
        description="Measure the intent classifier against a labelled dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"JSONL dataset to score (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--mode",
        choices=("classifier", "cache"),
        default="classifier",
        help=(
            "classifier: one cold pass, measures the classifiers. "
            "cache: warm on the prompts then score the paraphrases, measures "
            "cache hit rate and false-hit rate."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the full JSON report here as well as printing a summary",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="list every case the classifier got wrong",
    )
    parser.add_argument(
        "--describe-dataset",
        action="store_true",
        help="print the dataset's shape and exit without running anything",
    )

    layers = parser.add_argument_group("layers")
    layers.add_argument(
        "--embedder",
        default="hashing",
        help="hashing (default) | local/bge-small | minilm. local requires the embed extra.",
    )
    layers.add_argument(
        "--prototype",
        choices=("examples", "description", "mixed", "nearest"),
        default="examples",
        help="L3 intent representation. Default examples matches v1.",
    )
    layers.add_argument(
        "--l4-rerank",
        action="store_true",
        help="attach the local description reranker as L4. L5 stays off.",
    )
    layers.add_argument(
        "--hn-lambda",
        type=float,
        default=None,
        help="L3 hard-negative repulsion weight (default 0.35).",
    )
    layers.add_argument(
        "--cx-lambda",
        type=float,
        default=None,
        help="L3 counterexample repulsion weight (default 0.0).",
    )
    layers.add_argument(
        "--provider",
        help=("enable the model-backed layers L4/L5 using this configured provider. Costs money."),
    )
    layers.add_argument("--structured-model", help="model for L4")
    layers.add_argument("--escalation-model", help="model for L5 (should be stronger than L4)")

    tuning = parser.add_argument_group("thresholds")
    tuning.add_argument("--rules-threshold", type=float, help="confidence L2 must reach")
    tuning.add_argument("--embedding-threshold", type=float, help="confidence L3 must reach")
    tuning.add_argument("--structured-threshold", type=float, help="confidence L4 must reach")
    tuning.add_argument("--escalation-threshold", type=float, help="confidence L5 must reach")
    tuning.add_argument(
        "--semantic-similarity",
        type=float,
        help="cosine similarity a semantic cache hit must reach",
    )

    gates = parser.add_argument_group("gates (exit non-zero when missed)")
    gates.add_argument("--min-accuracy", type=float, help="strict accuracy floor")
    gates.add_argument("--min-lenient-accuracy", type=float, help="lenient accuracy floor")
    gates.add_argument("--min-macro-f1", type=float, help="macro-F1 floor")
    gates.add_argument("--min-unknown-recall", type=float, help="unknown-intent recall floor")
    gates.add_argument(
        "--min-high-confidence-precision",
        type=float,
        help="precision floor among cases with confidence at --high-confidence-threshold",
    )
    gates.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.90,
        help="confidence cut used by --min-high-confidence-precision (default 0.90)",
    )
    gates.add_argument(
        "--max-semantic-false-hit-rate",
        type=float,
        help="ceiling on the measured semantic-cache false-hit rate",
    )
    return parser


def _thresholds(args: argparse.Namespace) -> CascadeThresholds:
    defaults = CascadeThresholds()
    return CascadeThresholds(
        semantic_cache=defaults.semantic_cache,
        rules=args.rules_threshold if args.rules_threshold is not None else defaults.rules,
        embedding=(
            args.embedding_threshold if args.embedding_threshold is not None else defaults.embedding
        ),
        structured=(
            args.structured_threshold
            if args.structured_threshold is not None
            else defaults.structured
        ),
        escalation=(
            args.escalation_threshold
            if args.escalation_threshold is not None
            else defaults.escalation
        ),
    )


def _semantic_policy(args: argparse.Namespace) -> SemanticCachePolicy:
    default = SemanticCachePolicy()
    if args.semantic_similarity is None:
        return default
    return SemanticCachePolicy(
        similarity_threshold=args.semantic_similarity,
        confidence_threshold=default.confidence_threshold,
        ttl_seconds=default.ttl_seconds,
        capacity_per_signature=default.capacity_per_signature,
    )


def _build_cascade(args: argparse.Namespace) -> tuple[IntentCascade, list[str]]:
    """Assemble the cascade the flags ask for, and report anything it lacks."""
    taxonomy = bootstrap_taxonomy()
    cache = TenantScopedCache()
    thresholds = _thresholds(args)
    policy = _semantic_policy(args)
    notes: list[str] = []
    try:
        embedder = resolve_embedder(args.embedder)
    except (ValueError, RuntimeError) as exc:
        raise FabricError(str(exc)) from exc
    prototype = PrototypeKind(args.prototype)
    notes.append(f"embedder={embedder.model_id} prototype={prototype.value}")

    if not args.provider:
        if args.l4_rerank:
            notes.append("L4 local rerank enabled; L5 disabled")
        else:
            notes.append(
                "L4/L5 disabled: no --provider given, so only the offline layers were measured"
            )
        return (
            build_offline_cascade(
                taxonomy,
                cache,
                embedder=embedder,
                prototype=prototype,
                l4_rerank=args.l4_rerank,
                hn_lambda=args.hn_lambda,
                cx_lambda=args.cx_lambda,
                thresholds=thresholds,
                semantic_policy=policy,
            ),
            notes,
        )

    if not args.structured_model:
        raise FabricError("--provider requires --structured-model")

    settings = Settings()
    registry = ModelRegistry.from_yaml(settings.registry_path)
    provider = ProviderFactory(settings).get(args.provider)

    pricing = _pricing_for(registry, args.structured_model)
    escalation_pricing = (
        _pricing_for(registry, args.escalation_model) if args.escalation_model else None
    )
    if pricing is None:
        notes.append(
            f"model '{args.structured_model}' is unpriced in the registry, "
            "so reported classification cost is zero rather than unknown"
        )

    return (
        build_full_cascade(
            taxonomy,
            cache,
            provider=provider,
            structured_model=args.structured_model,
            escalation_model=args.escalation_model,
            structured_pricing=pricing,
            escalation_pricing=escalation_pricing,
            embedder=embedder,
            thresholds=thresholds,
            semantic_policy=policy,
        ),
        notes,
    )


def _pricing_for(registry: ModelRegistry, model: str | None) -> ClassifierPricing | None:
    """Registry prices for a model, or `None` when it has none.

    An unpriced model yields `None` rather than a zero-rate pricing object, so
    the caller can say "cost unknown" instead of reporting a confident zero.
    """
    if model is None:
        return None
    try:
        spec = registry.get(model)
    except ModelNotFoundError:
        return None
    input_cost = spec.input_cost_per_mtok
    output_cost = spec.output_cost_per_mtok
    if input_cost is None or output_cost is None:
        return None
    if input_cost == 0.0 and output_cost == 0.0:
        return None
    return ClassifierPricing(
        input_cost_per_mtok=input_cost,
        output_cost_per_mtok=output_cost,
    )


def _check_gates(report: BenchmarkReport, args: argparse.Namespace) -> list[str]:
    """Compare the run against the floors the caller asked for.

    A gate whose metric is `None` fails. "Not measured" must not be allowed to
    pass as "met the bar".
    """
    failures: list[str] = []

    def gate(name: str, actual: float | None, floor: float | None) -> None:
        if floor is None:
            return
        if actual is None:
            failures.append(f"{name}: not measured, cannot satisfy floor {floor}")
        elif actual < floor:
            failures.append(f"{name}: {actual:.4f} below floor {floor}")

    gate("accuracy", report.accuracy, args.min_accuracy)
    gate("lenient accuracy", report.lenient_accuracy, args.min_lenient_accuracy)
    gate("macro F1", report.macro_f1, args.min_macro_f1)
    gate(
        "unknown-intent recall",
        report.abstention_scores["unknown_intent_recall"],
        args.min_unknown_recall,
    )
    if args.min_high_confidence_precision is not None:
        rows = report.routing_precision((args.high_confidence_threshold,))
        precision = rows[0]["precision"] if rows else None
        gate(
            f"high-confidence precision (≥{args.high_confidence_threshold:.2f})",
            precision,
            args.min_high_confidence_precision,
        )

    ceiling = args.max_semantic_false_hit_rate
    if ceiling is not None:
        rate = report.cache_scores["semantic_false_hit_rate"]
        if rate is None:
            failures.append(
                "semantic false-hit rate: no semantic cache hits were served, "
                "so the rate is unmeasured"
            )
        elif rate > ceiling:
            failures.append(f"semantic false-hit rate: {rate:.4f} above ceiling {ceiling}")

    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cases = load_dataset(args.dataset)
    except FabricError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.describe_dataset:
        print(json.dumps(summarise_dataset(cases), indent=2))
        return 0

    try:
        cascade, notes = _build_cascade(args)
    except FabricError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = asyncio.run(run_benchmark(cascade, cases, mode=args.mode))
    report.warnings.extend(notes)

    payload: dict[str, Any] = report.as_dict()
    if args.show_failures:
        payload["failures"] = report.failures()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(report))
        if args.show_failures:
            print()
            print(f"Failures ({len(payload['failures'])})")
            for failure in payload["failures"]:
                print(
                    f"  [{failure['id']}] expected {failure['expected']}, "
                    f"got {failure['predicted']} ({failure['confidence']:.3f} "
                    f"via {failure['layer']})"
                )
                print(f"      {failure['text']}")

    gate_failures = _check_gates(report, args)
    if gate_failures:
        print("\nGates missed:", file=sys.stderr)
        for failure in gate_failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
