"""The benchmark CLI: exit codes, gates, and what it refuses to do silently."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_fabric.intent.cli import main

DATASET = Path("datasets/intent/bootstrap.jsonl")


class TestBasicRuns:
    def test_a_default_run_succeeds_without_a_provider(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--dataset", str(DATASET)])

        assert code == 0
        assert "Intent benchmark" in capsys.readouterr().out

    def test_it_says_when_the_model_layers_were_not_measured(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--dataset", str(DATASET)])

        assert "L4/L5 disabled" in capsys.readouterr().out

    def test_json_output_is_machine_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--dataset", str(DATASET), "--format", "json"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["scored"] > 0
        assert "expected_calibration_error" in payload
        assert "per_intent" in payload

    def test_the_report_can_be_written_to_a_file(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "report.json"

        main(["--dataset", str(DATASET), "--output", str(destination)])

        payload = json.loads(destination.read_text())
        assert payload["taxonomy_version"]

    def test_describing_a_dataset_runs_nothing(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--dataset", str(DATASET), "--describe-dataset"])

        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["cases"] > 0
        assert payload["hard_negatives"] > 0

    def test_failures_can_be_listed(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--dataset", str(DATASET), "--show-failures"])

        assert "Failures" in capsys.readouterr().out

    def test_cache_mode_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--dataset", str(DATASET), "--mode", "cache"])

        assert code == 0
        assert "mode: cache" in capsys.readouterr().out


class TestGates:
    def test_an_unreachable_floor_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["--dataset", str(DATASET), "--min-accuracy", "0.999"])

        assert code == 1
        assert "below floor" in capsys.readouterr().err

    def test_a_reachable_floor_passes(self) -> None:
        assert main(["--dataset", str(DATASET), "--min-accuracy", "0.1"]) == 0

    def test_an_unmeasured_metric_cannot_satisfy_a_gate(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ "Not measured" must never be allowed to read as "met the bar"."""
        code = main(
            [
                "--dataset",
                str(DATASET),
                "--max-semantic-false-hit-rate",
                "0.5",
            ]
        )

        assert code == 1
        assert "unmeasured" in capsys.readouterr().err

    def test_a_measured_false_hit_rate_can_satisfy_its_ceiling(self) -> None:
        code = main(
            [
                "--dataset",
                str(DATASET),
                "--mode",
                "cache",
                "--semantic-similarity",
                "0.5",
                "--max-semantic-false-hit-rate",
                "1.0",
            ]
        )

        assert code == 0

    def test_gates_report_every_miss_not_just_the_first(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "--dataset",
                str(DATASET),
                "--min-accuracy",
                "0.999",
                "--min-macro-f1",
                "0.999",
                "--min-unknown-recall",
                "0.999",
            ]
        )

        errors = capsys.readouterr().err
        assert code == 1
        assert errors.count("- ") == 3


class TestThresholdOverrides:
    def test_lowering_the_rules_threshold_changes_the_outcome(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["--dataset", str(DATASET), "--format", "json"])
        strict = json.loads(capsys.readouterr().out)

        main(
            [
                "--dataset",
                str(DATASET),
                "--format",
                "json",
                "--rules-threshold",
                "0.3",
                "--embedding-threshold",
                "0.3",
            ]
        )
        permissive = json.loads(capsys.readouterr().out)

        assert permissive["abstention"]["rate"] < strict["abstention"]["rate"]
        assert (
            permissive["layer_distribution"]["l2_rules"]
            > (strict["layer_distribution"]["l2_rules"])
        )


class TestFailureModes:
    def test_a_missing_dataset_exits_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--dataset", str(tmp_path / "absent.jsonl")])

        assert code == 2
        assert "no benchmark dataset" in capsys.readouterr().err

    def test_enabling_the_model_layers_without_a_model_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--dataset", str(DATASET), "--provider", "mock"])

        assert code == 2
        assert "--structured-model" in capsys.readouterr().err

    def test_an_unknown_provider_is_refused(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "--dataset",
                str(DATASET),
                "--provider",
                "not-a-provider",
                "--structured-model",
                "whatever",
            ]
        )

        assert code == 2
        assert capsys.readouterr().err
