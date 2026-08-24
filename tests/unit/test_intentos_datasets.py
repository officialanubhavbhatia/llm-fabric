"""IntentOS extended sets parse. Scores are only claimed when a bench is run."""

from __future__ import annotations

from pathlib import Path

from llm_fabric.eval.runner import load_examples

REPO = Path(__file__).resolve().parents[2]


def test_intentos_extended_set_loads() -> None:
    examples = load_examples(REPO / "datasets/eval/intentos/extended.jsonl")
    kinds = {row.metadata.get("expected_intent_id") for row in examples}
    tags = {tag for row in examples for tag in (row.metadata.get("tags") or [])}
    assert "unknown" in kinds
    assert "adversarial" in tags
    assert "ood" in tags
    assert "ambiguous" in tags
