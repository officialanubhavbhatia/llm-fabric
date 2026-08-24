"""Measuring the classifier instead of asserting that it is good.

Nothing in IntentOS is entitled to a quality claim until this module has been
run against a labelled dataset and the numbers published. That is the whole
reason it exists.

**Dataset format** — one JSON object per line:

```json
{"id": "c-001",
 "text": "Summarise this thread in two lines",
 "expected_intent_id": "summarization",
 "acceptable_intent_ids": ["summarization.thread"],
 "paraphrases": ["Give me a two-line summary of this thread"],
 "language": "en",
 "policy_version": "v1",
 "hard_negative": false,
 "multi_intent": false,
 "notes": "why this label"}
```

Only `id`, `text` and `expected_intent_id` are required. `expected_intent_id`
may be `"unknown"`, meaning the correct behaviour is to abstain — that is a
label like any other, and a classifier that never abstains will lose marks for
it.

`acceptable_intent_ids` exists for prompts where more than one label is
genuinely defensible. Strict accuracy ignores it; lenient accuracy honours it.
Both are reported, because collapsing them hides how much of a score rests on
judgement calls.

**Two run modes**, because they measure different things and the constitution
requires them to be separable:

- `classifier` — one pass over a cold cache. Measures the classifiers.
- `cache` — a warming pass over `text`, then a scored pass over `paraphrases`.
  Measures cache hit rate and, because ground truth is present, the *actual*
  semantic-cache false-hit rate rather than an estimate.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from llm_fabric.errors import ConfigurationError
from llm_fabric.intent.cascade import IntentCascade, IntentDecision
from llm_fabric.intent.metrics import percentile
from llm_fabric.intent.schema import (
    UNKNOWN_INTENT_ID,
    ClassificationRequest,
    ClassifierLayer,
)
from llm_fabric.tenancy.scope import TenantScope

BenchmarkMode = Literal["classifier", "cache"]

#: Bins for expected calibration error. Ten is the convention and keeps each bin
#: populated on datasets of a few hundred cases.
CALIBRATION_BINS = 10

#: Values of k reported for top-k accuracy. Bounded by how many alternatives a
#: classifier is allowed to emit.
TOP_K_VALUES: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One labelled prompt."""

    id: str
    text: str
    expected_intent_id: str
    acceptable_intent_ids: tuple[str, ...] = ()
    paraphrases: tuple[str, ...] = ()
    language: str = "en"
    policy_version: str = "v1"
    hard_negative: bool = False
    multi_intent: bool = False
    notes: str = ""
    split: str = "test"
    tags: tuple[str, ...] = ()
    difficulty: str = ""
    source: str = ""
    expected_domain: str | None = None
    expected_task: str | None = None
    conversation_context: str = ""
    required_capabilities: tuple[str, ...] = ()

    @property
    def expects_abstention(self) -> bool:
        return self.expected_intent_id == UNKNOWN_INTENT_ID

    @property
    def acceptable(self) -> frozenset[str]:
        return frozenset({self.expected_intent_id, *self.acceptable_intent_ids})

    def to_request(self, text: str | None = None) -> ClassificationRequest:
        return ClassificationRequest(
            text=text if text is not None else self.text,
            language=self.language,
            policy_version=self.policy_version,
            conversation_state_signature=self.conversation_context,
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], *, line: int) -> BenchmarkCase:
        for required in ("id", "text", "expected_intent_id"):
            if not payload.get(required):
                raise ConfigurationError(f"line {line}: benchmark case is missing '{required}'")
        return cls(
            id=str(payload["id"]),
            text=str(payload["text"]),
            expected_intent_id=str(payload["expected_intent_id"]),
            acceptable_intent_ids=tuple(
                str(item) for item in payload.get("acceptable_intent_ids", ())
            ),
            paraphrases=tuple(str(item) for item in payload.get("paraphrases", ())),
            language=str(payload.get("language", "en")),
            policy_version=str(payload.get("policy_version", "v1")),
            hard_negative=bool(payload.get("hard_negative", False)),
            multi_intent=bool(payload.get("multi_intent", False)),
            notes=str(payload.get("notes", "")),
            split=str(payload.get("split", "test")),
            tags=tuple(str(item) for item in payload.get("tags", ())),
            difficulty=str(payload.get("difficulty", "")),
            source=str(payload.get("source", "")),
            expected_domain=(
                str(payload["expected_domain"]) if payload.get("expected_domain") else None
            ),
            expected_task=str(payload["expected_task"]) if payload.get("expected_task") else None,
            conversation_context=str(payload.get("conversation_context", "")),
            required_capabilities=tuple(
                str(item) for item in payload.get("required_capabilities", ())
            ),
        )


def load_dataset(path: Path | str) -> list[BenchmarkCase]:
    """Read a JSONL dataset, refusing anything malformed.

    Silently skipping a bad line would quietly change the denominator of every
    metric computed from the file.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigurationError(f"no benchmark dataset at {resolved}")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()

    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ConfigurationError(f"line {line_number}: expected a JSON object")

            case = BenchmarkCase.from_json(payload, line=line_number)
            if case.id in seen:
                raise ConfigurationError(f"line {line_number}: duplicate case id '{case.id}'")
            seen.add(case.id)
            cases.append(case)

    if not cases:
        raise ConfigurationError(f"benchmark dataset {resolved} contains no cases")
    return cases


def validate_dataset(cases: Sequence[BenchmarkCase], cascade: IntentCascade) -> list[str]:
    """Report labels the taxonomy cannot produce.

    A dataset labelled against a different taxonomy version scores badly for
    reasons that have nothing to do with the classifier, so this is surfaced
    rather than left to be misread as a quality problem.
    """
    known = {node.intent_id for node in cascade.taxonomy} | {UNKNOWN_INTENT_ID}
    problems: list[str] = []
    for case in cases:
        for label in case.acceptable:
            if label not in known:
                problems.append(
                    f"case '{case.id}' expects intent '{label}', "
                    f"absent from taxonomy '{cascade.taxonomy.version}'"
                )
    return problems


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case: BenchmarkCase
    prompt: str
    decision: IntentDecision

    @property
    def predicted(self) -> str:
        return self.decision.classification.intent_id

    @property
    def confidence(self) -> float:
        return self.decision.classification.confidence

    @property
    def correct(self) -> bool:
        return self.predicted == self.case.expected_intent_id

    @property
    def lenient_correct(self) -> bool:
        return self.predicted in self.case.acceptable

    def top_k_correct(self, k: int) -> bool:
        ranked = [self.predicted] + [
            alt.intent_id for alt in self.decision.classification.alternatives
        ]
        return any(label in self.case.acceptable for label in ranked[:k])


@dataclass(slots=True)
class LabelScores:
    label: str
    support: int = 0
    predicted: int = 0
    true_positive: int = 0

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if denominator == 0.0:
            return 0.0
        return 2 * self.precision * self.recall / denominator

    def as_dict(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "predicted": self.predicted,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Everything measured in one run. No interpretation, no claims."""

    mode: BenchmarkMode
    taxonomy_version: str
    classifier_version: str
    dataset_size: int
    scored: int
    outcomes: list[CaseOutcome] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # -- headline ------------------------------------------------------------

    @property
    def accuracy(self) -> float | None:
        return _ratio(sum(1 for o in self.outcomes if o.correct), self.scored)

    @property
    def lenient_accuracy(self) -> float | None:
        return _ratio(sum(1 for o in self.outcomes if o.lenient_correct), self.scored)

    def top_k_accuracy(self, k: int) -> float | None:
        return _ratio(sum(1 for o in self.outcomes if o.top_k_correct(k)), self.scored)

    # -- per-label -----------------------------------------------------------

    def label_scores(self) -> dict[str, LabelScores]:
        scores: dict[str, LabelScores] = {}

        def slot(label: str) -> LabelScores:
            if label not in scores:
                scores[label] = LabelScores(label=label)
            return scores[label]

        for outcome in self.outcomes:
            expected = outcome.case.expected_intent_id
            predicted = outcome.predicted
            slot(expected).support += 1
            slot(predicted).predicted += 1
            if expected == predicted:
                slot(expected).true_positive += 1

        return dict(sorted(scores.items()))

    @property
    def macro_f1(self) -> float | None:
        """Unweighted mean F1 over labels that appear in the dataset.

        Labels with no support are excluded: averaging in a zero for an intent
        the dataset never exercises would penalise the classifier for the
        dataset's coverage rather than its own behaviour.
        """
        scores = [score for score in self.label_scores().values() if score.support]
        if not scores:
            return None
        return sum(score.f1 for score in scores) / len(scores)

    @property
    def micro_f1(self) -> float | None:
        """Globally pooled F1.

        For single-label classification where every case receives exactly one
        prediction, this equals accuracy. It is reported because the
        constitution asks for it, not because it adds information here.
        """
        scores = self.label_scores().values()
        true_positive = sum(score.true_positive for score in scores)
        predicted = sum(score.predicted for score in scores)
        support = sum(score.support for score in scores)
        if not predicted or not support:
            return None
        precision = true_positive / predicted
        recall = true_positive / support
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def routing_precision(self, thresholds: Sequence[float] | None = None) -> list[dict[str, Any]]:
        """Precision among cases whose confidence clears each threshold.

        This is the product metric: how often a *high-confidence* routing
        decision would have been wrong. Coverage is the share of the dataset
        that would have been routed at that threshold rather than abstaining.
        """
        cuts = thresholds or (0.80, 0.85, 0.90, 0.95, 0.98)
        rows: list[dict[str, Any]] = []
        for threshold in cuts:
            selected = [
                o for o in self.outcomes if o.confidence >= threshold and not o.decision.abstained
            ]
            routed = len(selected)
            rows.append(
                {
                    "threshold": threshold,
                    "coverage": _ratio(routed, self.scored),
                    "precision": _ratio(sum(1 for o in selected if o.correct), routed),
                    "lenient_precision": _ratio(
                        sum(1 for o in selected if o.lenient_correct), routed
                    ),
                    "n": routed,
                }
            )
        return rows

    @property
    def brier_score(self) -> float | None:
        """Mean squared gap between confidence and correctness. Lower is better."""
        if not self.outcomes:
            return None
        total = sum((o.confidence - (1.0 if o.correct else 0.0)) ** 2 for o in self.outcomes)
        return total / len(self.outcomes)

    def confusion_matrix(self) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {}
        for outcome in self.outcomes:
            row = matrix.setdefault(outcome.case.expected_intent_id, {})
            row[outcome.predicted] = row.get(outcome.predicted, 0) + 1
        return {expected: dict(sorted(row.items())) for expected, row in sorted(matrix.items())}

    def confusions(self, limit: int = 10) -> list[dict[str, Any]]:
        """The most frequent mistakes, largest first."""
        pairs: dict[tuple[str, str], int] = {}
        for outcome in self.outcomes:
            if not outcome.correct:
                key = (outcome.case.expected_intent_id, outcome.predicted)
                pairs[key] = pairs.get(key, 0) + 1
        ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in ranked[:limit]
        ]

    # -- abstention ----------------------------------------------------------

    @property
    def abstention_scores(self) -> dict[str, Any]:
        """How well the classifier knows what it does not know.

        Three separate facts, because one number cannot express them. Recall is
        the share of genuinely unclassifiable prompts it declined. Precision is
        the share of its abstentions that were warranted. Accuracy is the share
        of all cases where abstaining or not abstaining was the right call.
        """
        abstained = [o for o in self.outcomes if o.decision.abstained]
        should_abstain = [o for o in self.outcomes if o.case.expects_abstention]
        correct_calls = sum(
            1 for o in self.outcomes if o.decision.abstained == o.case.expects_abstention
        )
        return {
            "abstained": len(abstained),
            "should_have_abstained": len(should_abstain),
            "rate": _ratio(len(abstained), self.scored),
            "precision": _ratio(
                sum(1 for o in abstained if o.case.expects_abstention), len(abstained)
            ),
            "unknown_intent_recall": _ratio(
                sum(1 for o in should_abstain if o.decision.abstained), len(should_abstain)
            ),
            "accuracy": _ratio(correct_calls, self.scored),
        }

    # -- calibration ---------------------------------------------------------

    @property
    def expected_calibration_error(self) -> float | None:
        """Weighted mean gap between stated confidence and observed accuracy.

        Zero means a classifier reporting 0.7 is right 70% of the time. A large
        value means the confidence numbers cannot be used as thresholds, which
        matters directly: the cascade gates on them.
        """
        if not self.outcomes:
            return None

        bins: list[list[CaseOutcome]] = [[] for _ in range(CALIBRATION_BINS)]
        for outcome in self.outcomes:
            index = min(int(outcome.confidence * CALIBRATION_BINS), CALIBRATION_BINS - 1)
            bins[index].append(outcome)

        error = 0.0
        for bucket in bins:
            if not bucket:
                continue
            accuracy = sum(1 for o in bucket if o.correct) / len(bucket)
            confidence = sum(o.confidence for o in bucket) / len(bucket)
            error += (len(bucket) / len(self.outcomes)) * abs(accuracy - confidence)
        return error

    def calibration_bins(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in range(CALIBRATION_BINS):
            low = index / CALIBRATION_BINS
            high = (index + 1) / CALIBRATION_BINS
            bucket = [
                o
                for o in self.outcomes
                if min(int(o.confidence * CALIBRATION_BINS), CALIBRATION_BINS - 1) == index
            ]
            if not bucket:
                continue
            rows.append(
                {
                    "range": f"{low:.1f}-{high:.1f}",
                    "count": len(bucket),
                    "mean_confidence": round(sum(o.confidence for o in bucket) / len(bucket), 4),
                    "accuracy": round(sum(1 for o in bucket if o.correct) / len(bucket), 4),
                }
            )
        return rows

    # -- cost, latency, layers -----------------------------------------------

    @property
    def latency_ms(self) -> dict[str, float | None]:
        values = [o.decision.total_latency_ms for o in self.outcomes]
        return {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "p50": _round(percentile(values, 0.50)),
            "p95": _round(percentile(values, 0.95)),
            "p99": _round(percentile(values, 0.99)),
            "max": round(max(values), 4) if values else None,
        }

    @property
    def cost(self) -> dict[str, float | None]:
        values = [o.decision.total_cost_usd for o in self.outcomes]
        total = sum(values)
        return {
            "total_usd": round(total, 6),
            "mean_usd": round(total / len(values), 8) if values else None,
        }

    def layer_distribution(self) -> dict[str, int]:
        counts = {layer.value: 0 for layer in ClassifierLayer}
        for outcome in self.outcomes:
            counts[outcome.decision.layer.value] += 1
        return counts

    @property
    def cache_scores(self) -> dict[str, Any]:
        """Cache behaviour, measured against ground truth.

        The false-hit rate here is a real measurement, not the runtime estimate:
        the benchmark knows the correct label, so it knows whether a cache hit
        served the wrong answer.
        """
        exact = [o for o in self.outcomes if o.decision.layer is ClassifierLayer.L0_EXACT_CACHE]
        semantic = [
            o for o in self.outcomes if o.decision.layer is ClassifierLayer.L1_SEMANTIC_CACHE
        ]
        return {
            "exact_hits": len(exact),
            "semantic_hits": len(semantic),
            "hit_rate": _ratio(len(exact) + len(semantic), self.scored),
            "exact_false_hit_rate": _ratio(
                sum(1 for o in exact if not o.lenient_correct), len(exact)
            ),
            "semantic_false_hit_rate": _ratio(
                sum(1 for o in semantic if not o.lenient_correct), len(semantic)
            ),
        }

    # -- slices --------------------------------------------------------------

    def slice_accuracy(self) -> dict[str, Any]:
        """Accuracy on the subsets that are supposed to be hard."""

        def score(subset: list[CaseOutcome]) -> dict[str, Any]:
            return {
                "count": len(subset),
                "accuracy": _ratio(sum(1 for o in subset if o.correct), len(subset)),
                "lenient_accuracy": _ratio(
                    sum(1 for o in subset if o.lenient_correct), len(subset)
                ),
            }

        return {
            "hard_negatives": score([o for o in self.outcomes if o.case.hard_negative]),
            "multi_intent": score([o for o in self.outcomes if o.case.multi_intent]),
            "expects_abstention": score([o for o in self.outcomes if o.case.expects_abstention]),
            "ordinary": score(
                [
                    o
                    for o in self.outcomes
                    if not (
                        o.case.hard_negative or o.case.multi_intent or o.case.expects_abstention
                    )
                ]
            ),
        }

    # -- rendering -----------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "taxonomy_version": self.taxonomy_version,
            "classifier_version": self.classifier_version,
            "dataset_size": self.dataset_size,
            "scored": self.scored,
            "duration_s": round(self.duration_s, 3),
            "warnings": list(self.warnings),
            "accuracy": _round(self.accuracy),
            "lenient_accuracy": _round(self.lenient_accuracy),
            "top_k_accuracy": {f"top_{k}": _round(self.top_k_accuracy(k)) for k in TOP_K_VALUES},
            "macro_f1": _round(self.macro_f1),
            "micro_f1": _round(self.micro_f1),
            "expected_calibration_error": _round(self.expected_calibration_error),
            "brier_score": _round(self.brier_score),
            "high_confidence_routing": self.routing_precision(),
            "abstention": self.abstention_scores,
            "slices": self.slice_accuracy(),
            "cache": self.cache_scores,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "layer_distribution": self.layer_distribution(),
            "per_intent": {label: score.as_dict() for label, score in self.label_scores().items()},
            "top_confusions": self.confusions(),
            "calibration_bins": self.calibration_bins(),
        }

    def failures(self) -> list[dict[str, Any]]:
        return [
            {
                "id": outcome.case.id,
                "text": outcome.prompt,
                "expected": outcome.case.expected_intent_id,
                "predicted": outcome.predicted,
                "confidence": round(outcome.confidence, 4),
                "layer": outcome.decision.layer.value,
                "hard_negative": outcome.case.hard_negative,
                "multi_intent": outcome.case.multi_intent,
                "notes": outcome.case.notes,
            }
            for outcome in self.outcomes
            if not outcome.lenient_correct
        ]


async def run_benchmark(
    cascade: IntentCascade,
    cases: Sequence[BenchmarkCase],
    *,
    scope: TenantScope | None = None,
    mode: BenchmarkMode = "classifier",
) -> BenchmarkReport:
    """Run a dataset through a cascade and measure what happened.

    The tenant scope defaults to a dedicated benchmark tenant so a run cannot
    populate or read a real tenant's caches.
    """
    resolved_scope = scope or TenantScope(tenant_id="benchmark", user_id="benchmark")
    warnings = validate_dataset(cases, cascade)

    report = BenchmarkReport(
        mode=mode,
        taxonomy_version=cascade.taxonomy.version,
        classifier_version=cascade.version,
        dataset_size=len(cases),
        scored=0,
        warnings=warnings,
    )

    started = time.perf_counter()

    if mode == "cache":
        # Warming pass. Not scored: it measures nothing except that the caches
        # can be written to.
        for case in cases:
            await cascade.classify(resolved_scope, case.to_request())

        for case in cases:
            for paraphrase in case.paraphrases:
                decision = await cascade.classify(resolved_scope, case.to_request(paraphrase))
                report.outcomes.append(CaseOutcome(case=case, prompt=paraphrase, decision=decision))
        if not report.outcomes:
            report.warnings.append(
                "cache mode scored nothing: no case in this dataset has paraphrases"
            )
    else:
        for case in cases:
            decision = await cascade.classify(resolved_scope, case.to_request())
            report.outcomes.append(CaseOutcome(case=case, prompt=case.text, decision=decision))

    report.scored = len(report.outcomes)
    report.duration_s = time.perf_counter() - started
    _feed_back_false_hits(cascade, report)
    return report


def _feed_back_false_hits(cascade: IntentCascade, report: BenchmarkReport) -> None:
    """Tell the caches how many of their hits were actually wrong.

    The runtime false-hit counters have no way to learn this on their own. A
    benchmark run is the one moment ground truth is available, so it is reported
    back rather than discarded.
    """
    semantic_hits = [
        outcome
        for outcome in report.outcomes
        if outcome.decision.layer is ClassifierLayer.L1_SEMANTIC_CACHE
    ]
    if not semantic_hits:
        return
    wrong = sum(1 for outcome in semantic_hits if not outcome.lenient_correct)
    if cascade.semantic_cache is not None:
        cascade.semantic_cache.report_false_hit(reviewed=len(semantic_hits), wrong=wrong)


def render_text(report: BenchmarkReport) -> str:
    """A terminal summary. Deliberately states no verdict."""
    data = report.as_dict()
    lines: list[str] = []

    def row(label: str, value: Any) -> None:
        lines.append(f"  {label:<28} {value}")

    lines.append(f"Intent benchmark — mode: {report.mode}")
    lines.append(f"  taxonomy   {report.taxonomy_version}")
    lines.append(f"  classifier {report.classifier_version}")
    lines.append(
        f"  scored     {report.scored} of {report.dataset_size} cases in {data['duration_s']}s"
    )
    lines.append("")

    if report.warnings:
        lines.append("Warnings")
        for warning in report.warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    lines.append("Accuracy")
    row("strict", _fmt(data["accuracy"]))
    row("lenient", _fmt(data["lenient_accuracy"]))
    for key, value in data["top_k_accuracy"].items():
        row(key, _fmt(value))
    row("macro F1", _fmt(data["macro_f1"]))
    row("micro F1", _fmt(data["micro_f1"]))
    row("calibration error (ECE)", _fmt(data["expected_calibration_error"]))
    if data.get("brier_score") is not None:
        row("Brier score", _fmt(data["brier_score"]))
    lines.append("")

    if data.get("high_confidence_routing"):
        lines.append("High-confidence routing precision")
        for row_data in data["high_confidence_routing"]:
            row(
                f"≥ {row_data['threshold']:.2f}",
                (
                    f"precision={_fmt(row_data['precision'])} "
                    f"coverage={_fmt(row_data['coverage'])} n={row_data['n']}"
                ),
            )
        lines.append("")

    lines.append("Abstention")
    for key in ("rate", "precision", "unknown_intent_recall", "accuracy"):
        row(key, _fmt(data["abstention"][key]))
    lines.append("")

    lines.append("Slices")
    for name, slice_data in data["slices"].items():
        row(f"{name} (n={slice_data['count']})", _fmt(slice_data["accuracy"]))
    lines.append("")

    lines.append("Cache")
    for key, value in data["cache"].items():
        row(key, _fmt(value) if isinstance(value, float | type(None)) else value)
    lines.append("")

    lines.append("Cost and latency")
    row("total cost (USD)", data["cost"]["total_usd"])
    for key, value in data["latency_ms"].items():
        row(f"latency {key} (ms)", value)
    lines.append("")

    lines.append("Answering layer")
    for layer, count in data["layer_distribution"].items():
        if count:
            row(layer, count)
    lines.append("")

    if data["top_confusions"]:
        lines.append("Most frequent confusions")
        for entry in data["top_confusions"]:
            lines.append(f"  {entry['expected']:<24} -> {entry['predicted']:<24} {entry['count']}")
        lines.append("")

    lines.append("These are measurements of this classifier on this dataset. They are not")
    lines.append("a comparison against any other system, and they carry over to production")
    lines.append("traffic only as far as this dataset resembles it.")
    return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> float | None:
    """`None` rather than zero when there is nothing to divide.

    An empty slice has no accuracy. Reporting 0.0 would read as total failure.
    """
    return numerator / denominator if denominator else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarise_dataset(cases: Iterable[BenchmarkCase]) -> dict[str, Any]:
    """Shape of a dataset, for spotting an unbalanced one before trusting it."""
    resolved = list(cases)
    by_label: dict[str, int] = {}
    for case in resolved:
        by_label[case.expected_intent_id] = by_label.get(case.expected_intent_id, 0) + 1
    return {
        "cases": len(resolved),
        "labels": len(by_label),
        "hard_negatives": sum(1 for case in resolved if case.hard_negative),
        "multi_intent": sum(1 for case in resolved if case.multi_intent),
        "expects_abstention": sum(1 for case in resolved if case.expects_abstention),
        "with_paraphrases": sum(1 for case in resolved if case.paraphrases),
        "languages": dict(sorted(Counter(case.language for case in resolved).items())),
        "splits": dict(sorted(Counter(case.split for case in resolved).items())),
        "by_label": dict(sorted(by_label.items())),
    }
