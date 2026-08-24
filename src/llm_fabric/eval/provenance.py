"""Collect evaluation provenance without inventing any of it.

A missing commit is recorded as missing. Guessing `unknown` or `dirty` would
look like a fact.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from llm_fabric.eval.schema import METRIC_VERSION, EvalProvenance, dataset_version


def current_commit(repo: Path | None = None) -> str | None:
    """`git rev-parse HEAD`, or `None` when git is not available or not a repo."""
    cwd = repo or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def build_provenance(
    *,
    examples: list[dict[str, Any]],
    configuration: dict[str, Any],
    model: str | None = None,
    model_version: str | None = None,
    prompt_version: str | None = None,
    taxonomy_version: str | None = None,
    repo: Path | None = None,
) -> EvalProvenance:
    return EvalProvenance(
        commit=current_commit(repo),
        model=model,
        model_version=model_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy_version,
        dataset_version=dataset_version(examples),
        metric_version=METRIC_VERSION,
        configuration=configuration,
    )
