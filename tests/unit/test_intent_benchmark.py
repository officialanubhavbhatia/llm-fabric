"""The benchmark: dataset handling and the arithmetic behind every metric.

The metric code is tested against hand-built outcomes rather than real
classifier output, because a metric that is only ever checked against whatever
the classifier happened to produce is not checked at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_fabric.errors import ConfigurationError
from llm_fabric.intent.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    CaseOutcome,
    load_dataset,
    render_text,
    run_benchmark,
    summarise_dataset,
    validate_dataset,
)
from llm_fabric.intent.bootstrap import bootstrap_taxonomy
from llm_fabric.intent.cache import SemanticCachePolicy
from llm_fabric.intent.cascade import IntentDecision
from llm_fabric.intent.factory import build_offline_cascade
from llm_fabric.intent.schema import (
    UNKNOWN_INTENT_ID,
    ClassifierLayer,
    Complexity,
    ContextClass,
    CostClass,
    IntentAlternative,
    IntentClassification,
    LatencyClass,
    Modality,
    QualityClass,
    ReasoningLevel,
    RiskClass,
)
from llm_fabric.tenancy.cache import TenantScopedCache

DATASET = Path("datasets/intent/bootstrap.jsonl")


def decision(
    intent_id: str,
    confidence: float = 0.9,
    *,
    layer: ClassifierLayer = ClassifierLayer.L2_RULES,
    alternatives: tuple[str, ...] = (),
    latency_ms: float = 1.0,
    cost_usd: float = 0.0,
) -> IntentDecision:
    if intent_id == UNKNOWN_INTENT_ID:
        classification = IntentClassification.unknown(
            classifier_version="test-1",
            taxonomy_version="v1",
            confidence=confidence,
            layer=ClassifierLayer.ABSTAIN,
        )
    else:
        classification = IntentClassification(
            intent_id=intent_id,
            domain=intent_id.split(".")[0],
            complexity=Complexity.MODERATE,
            reasoning_level=ReasoningLevel.LIGHT,
            modality=Modality.TEXT,
            context_class=ContextClass.SHORT,
            risk_class=RiskClass.LOW,
            latency_class=LatencyClass.INTERACTIVE,
            quality_class=QualityClass.STANDARD,
            cost_class=CostClass.LOW,
            confidence=confidence,
            classifier_version="test-1",
            taxonomy_version="v1",
            layer=layer,
            cache_hit=layer.is_cache,
            alternatives=tuple(IntentAlternative(alt, confidence / 2) for alt in alternatives),
        )
    return IntentDecision(
        classification=classification,
        total_latency_ms=latency_ms,
        total_cost_usd=cost_usd,
    )


def outcome(
    expected: str,
    predicted: str,
    confidence: float = 0.9,
    **kwargs: object,
) -> CaseOutcome:
    case = BenchmarkCase(
        id=f"{expected}->{predicted}",
        text="prompt",
        expected_intent_id=expected,
        acceptable_intent_ids=tuple(kwargs.pop("acceptable", ())),  # type: ignore[arg-type]
        hard_negative=bool(kwargs.pop("hard_negative", False)),
        multi_intent=bool(kwargs.pop("multi_intent", False)),
    )
    return CaseOutcome(
        case=case,
        prompt="prompt",
        decision=decision(predicted, confidence, **kwargs),  # type: ignore[arg-type]
    )


def report_of(*outcomes: CaseOutcome, mode: str = "classifier") -> BenchmarkReport:
    return BenchmarkReport(
        mode=mode,  # type: ignore[arg-type]
        taxonomy_version="v1",
        classifier_version="test-1",
        dataset_size=len(outcomes),
        scored=len(outcomes),
        outcomes=list(outcomes),
    )


class TestDatasetLoading:
    def test_the_shipped_dataset_loads(self) -> None:
        cases = load_dataset(DATASET)

        assert len(cases) > 50
        assert all(case.id and case.text and case.expected_intent_id for case in cases)

    def test_the_shipped_dataset_labels_exist_in_the_bootstrap_taxonomy(self) -> None:
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        assert validate_dataset(load_dataset(DATASET), cascade) == []

    def test_the_shipped_dataset_covers_the_hard_shapes(self) -> None:
        summary = summarise_dataset(load_dataset(DATASET))

        assert summary["hard_negatives"] >= 10
        assert summary["multi_intent"] >= 5
        assert summary["expects_abstention"] >= 5
        assert summary["with_paraphrases"] >= 10

    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="no benchmark dataset"):
            load_dataset(tmp_path / "absent.jsonl")

    def test_a_malformed_line_is_refused_rather_than_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "text": "x", "expected_intent_id": "coding"}\nnot json\n')

        with pytest.raises(ConfigurationError, match="line 2"):
            load_dataset(path)

    def test_a_missing_label_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "a", "text": "x"}\n')

        with pytest.raises(ConfigurationError, match="expected_intent_id"):
            load_dataset(path)

    def test_a_duplicate_id_is_refused(self, tmp_path: Path) -> None:
        line = '{"id": "a", "text": "x", "expected_intent_id": "coding"}'
        path = tmp_path / "dupe.jsonl"
        path.write_text(f"{line}\n{line}\n")

        with pytest.raises(ConfigurationError, match="duplicate case id"):
            load_dataset(path)

    def test_an_empty_dataset_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("\n// only a comment\n")

        with pytest.raises(ConfigurationError, match="no cases"):
            load_dataset(path)

    def test_labels_absent_from_the_taxonomy_are_warned_about(self, tmp_path: Path) -> None:
        path = tmp_path / "stale.jsonl"
        path.write_text('{"id": "a", "text": "x", "expected_intent_id": "telepathy"}\n')
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        problems = validate_dataset(load_dataset(path), cascade)
        assert len(problems) == 1
        assert "telepathy" in problems[0]


class TestAccuracyMetrics:
    def test_strict_accuracy_counts_only_the_expected_label(self) -> None:
        report = report_of(
            outcome("coding", "coding"),
            outcome("coding", "writing"),
            outcome("writing", "writing"),
            outcome("writing", "coding"),
        )

        assert report.accuracy == pytest.approx(0.5)

    def test_lenient_accuracy_honours_the_acceptable_set(self) -> None:
        report = report_of(
            outcome("coding", "coding.debug", acceptable=("coding.debug",)),
            outcome("coding", "writing"),
        )

        assert report.accuracy == pytest.approx(0.0)
        assert report.lenient_accuracy == pytest.approx(0.5)

    def test_top_k_accuracy_reads_the_alternatives(self) -> None:
        report = report_of(outcome("writing", "coding", alternatives=("writing",)))

        assert report.top_k_accuracy(1) == pytest.approx(0.0)
        assert report.top_k_accuracy(2) == pytest.approx(1.0)

    def test_per_label_precision_and_recall(self) -> None:
        report = report_of(
            outcome("coding", "coding"),
            outcome("coding", "coding"),
            outcome("writing", "coding"),
            outcome("coding", "writing"),
        )

        scores = report.label_scores()
        # coding: predicted 3 times, right twice; supported 3 times, right twice.
        assert scores["coding"].precision == pytest.approx(2 / 3)
        assert scores["coding"].recall == pytest.approx(2 / 3)
        # writing: predicted once and wrong.
        assert scores["writing"].precision == pytest.approx(0.0)
        assert scores["writing"].recall == pytest.approx(0.0)

    def test_macro_f1_ignores_labels_the_dataset_never_exercises(self) -> None:
        report = report_of(outcome("coding", "coding"), outcome("coding", "writing"))

        scores = report.label_scores()
        assert scores["writing"].support == 0
        # Only 'coding' has support, so macro F1 is coding's F1 alone.
        assert report.macro_f1 == pytest.approx(scores["coding"].f1)

    def test_micro_f1_equals_accuracy_for_single_label_classification(self) -> None:
        report = report_of(
            outcome("coding", "coding"),
            outcome("writing", "coding"),
            outcome("writing", "writing"),
        )

        assert report.micro_f1 == pytest.approx(report.accuracy)

    def test_the_confusion_matrix_records_what_was_mistaken_for_what(self) -> None:
        report = report_of(
            outcome("coding", "writing"),
            outcome("coding", "writing"),
            outcome("coding", "coding"),
        )

        assert report.confusion_matrix() == {"coding": {"coding": 1, "writing": 2}}
        assert report.confusions()[0] == {
            "expected": "coding",
            "predicted": "writing",
            "count": 2,
        }


class TestAbstentionMetrics:
    def test_the_three_abstention_figures_are_distinct(self) -> None:
        report = report_of(
            outcome(UNKNOWN_INTENT_ID, UNKNOWN_INTENT_ID),  # correctly abstained
            outcome(UNKNOWN_INTENT_ID, "coding"),  # should have abstained
            outcome("coding", UNKNOWN_INTENT_ID),  # abstained when it knew
            outcome("coding", "coding"),  # correctly answered
        )

        scores = report.abstention_scores
        assert scores["abstained"] == 2
        assert scores["should_have_abstained"] == 2
        assert scores["precision"] == pytest.approx(0.5)
        assert scores["unknown_intent_recall"] == pytest.approx(0.5)
        assert scores["accuracy"] == pytest.approx(0.5)

    def test_unknown_recall_is_unmeasured_when_nothing_should_abstain(self) -> None:
        report = report_of(outcome("coding", "coding"))

        assert report.abstention_scores["unknown_intent_recall"] is None


class TestCalibration:
    def test_a_perfectly_calibrated_classifier_scores_zero(self) -> None:
        # Ten cases at confidence 0.95, nine of which are right, lands in the
        # 0.9-1.0 bin with accuracy 0.9 against mean confidence 0.95.
        report = report_of(
            *[outcome("coding", "coding", 0.95) for _ in range(9)],
            outcome("coding", "writing", 0.95),
        )

        assert report.expected_calibration_error == pytest.approx(0.05, abs=1e-9)

    def test_overconfidence_is_measured(self) -> None:
        report = report_of(*[outcome("coding", "writing", 0.99) for _ in range(10)])

        assert report.expected_calibration_error == pytest.approx(0.99, abs=1e-9)

    def test_the_bins_are_reported_for_inspection(self) -> None:
        report = report_of(outcome("coding", "coding", 0.95), outcome("coding", "writing", 0.15))

        ranges = {row["range"] for row in report.calibration_bins()}
        assert ranges == {"0.9-1.0", "0.1-0.2"}


class TestCacheAndCostMetrics:
    def test_a_wrong_cache_hit_is_a_false_hit(self) -> None:
        report = report_of(
            outcome("coding", "coding", layer=ClassifierLayer.L1_SEMANTIC_CACHE),
            outcome("coding", "writing", layer=ClassifierLayer.L1_SEMANTIC_CACHE),
            outcome("coding", "coding", layer=ClassifierLayer.L0_EXACT_CACHE),
        )

        scores = report.cache_scores
        assert scores["semantic_hits"] == 2
        assert scores["semantic_false_hit_rate"] == pytest.approx(0.5)
        assert scores["exact_false_hit_rate"] == pytest.approx(0.0)

    def test_the_false_hit_rate_is_unmeasured_without_hits(self) -> None:
        report = report_of(outcome("coding", "coding"))

        assert report.cache_scores["semantic_false_hit_rate"] is None

    def test_cost_and_latency_are_summarised(self) -> None:
        report = report_of(
            outcome("coding", "coding", latency_ms=1.0, cost_usd=0.01),
            outcome("coding", "coding", latency_ms=3.0, cost_usd=0.03),
        )

        assert report.cost["total_usd"] == pytest.approx(0.04)
        assert report.latency_ms["mean"] == pytest.approx(2.0)
        assert report.latency_ms["p50"] == pytest.approx(1.0)


class TestSlices:
    def test_hard_shapes_are_scored_separately(self) -> None:
        report = report_of(
            outcome("coding", "writing", hard_negative=True),
            outcome("coding", "coding"),
            outcome(UNKNOWN_INTENT_ID, UNKNOWN_INTENT_ID, multi_intent=True),
        )

        slices = report.slice_accuracy()
        assert slices["hard_negatives"]["accuracy"] == pytest.approx(0.0)
        assert slices["ordinary"]["accuracy"] == pytest.approx(1.0)
        assert slices["multi_intent"]["count"] == 1

    def test_an_empty_slice_reports_absence_not_zero(self) -> None:
        report = report_of(outcome("coding", "coding"))

        assert report.slice_accuracy()["hard_negatives"]["accuracy"] is None


class TestRunningTheBenchmark:
    async def test_a_classifier_run_scores_every_case(self) -> None:
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
        cases = load_dataset(DATASET)

        report = await run_benchmark(cascade, cases)

        assert report.scored == len(cases)
        assert report.mode == "classifier"
        assert report.taxonomy_version == cascade.taxonomy.version

    async def test_a_classifier_run_starts_from_a_cold_cache(self) -> None:
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())

        report = await run_benchmark(cascade, load_dataset(DATASET))

        assert report.cache_scores["exact_hits"] == 0, (
            "a classifier run that reads its own cache is measuring the cache"
        )

    async def test_a_cache_run_scores_the_paraphrases(self) -> None:
        cascade = build_offline_cascade(
            bootstrap_taxonomy(),
            TenantScopedCache(),
            semantic_policy=SemanticCachePolicy(similarity_threshold=0.5, confidence_threshold=0.5),
        )
        cases = load_dataset(DATASET)
        expected = sum(len(case.paraphrases) for case in cases)

        report = await run_benchmark(cascade, cases, mode="cache")

        assert report.scored == expected
        assert report.cache_scores["semantic_hits"] > 0

    async def test_a_cache_run_on_a_dataset_without_paraphrases_says_so(self) -> None:
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
        case = BenchmarkCase(id="a", text="summarise this", expected_intent_id="summarization")

        report = await run_benchmark(cascade, [case], mode="cache")

        assert report.scored == 0
        assert any("no case" in warning for warning in report.warnings)

    async def test_measured_false_hits_are_reported_back_to_the_cache(self) -> None:
        cascade = build_offline_cascade(
            bootstrap_taxonomy(),
            TenantScopedCache(),
            semantic_policy=SemanticCachePolicy(similarity_threshold=0.3, confidence_threshold=0.3),
        )
        assert cascade.semantic_cache is not None
        assert cascade.semantic_cache.stats.false_hit_rate is None

        await run_benchmark(cascade, load_dataset(DATASET), mode="cache")

        assert cascade.semantic_cache.stats.reviewed_hits > 0
        assert cascade.semantic_cache.stats.false_hit_rate is not None

    async def test_a_run_uses_its_own_tenant(self) -> None:
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
        case = BenchmarkCase(id="a", text="summarise this", expected_intent_id="summarization")

        await run_benchmark(cascade, [case])

        from llm_fabric.tenancy.scope import TenantScope

        production = TenantScope(tenant_id="acme", user_id="alice")
        assert (
            cascade.exact_cache.get(
                production, "summarise this", cascade.discriminators(case.to_request())
            )
            is None
        )

    async def test_the_rendered_summary_states_no_verdict(self) -> None:
        cascade = build_offline_cascade(bootstrap_taxonomy(), TenantScopedCache())
        report = await run_benchmark(cascade, load_dataset(DATASET))

        text = render_text(report).lower()
        assert "accuracy" in text
        for boast in ("best", "superior", "outperform", "state of the art"):
            assert boast not in text
