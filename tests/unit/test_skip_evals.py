"""SKIP_EVALS must not silently pass a release gate."""

from __future__ import annotations

from llm_fabric.eval.cli import main


def test_skip_evals_env_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("SKIP_EVALS", "true")
    assert main(["gate"]) == 2
    monkeypatch.delenv("SKIP_EVALS")
    monkeypatch.setenv("LLM_FABRIC_SKIP_EVALS", "1")
    assert main(["gate"]) == 2


def test_eval_cli_requires_environment(monkeypatch) -> None:
    monkeypatch.delenv("LLM_FABRIC_ENVIRONMENT", raising=False)
    assert main(["gate"]) == 2
