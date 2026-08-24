"""Versioned benchmark artifacts.

A result file is named by kind, commit and timestamp so two runs cannot
overwrite each other silently. The files live under `artifacts/bench/`, which
is gitignored; the numbers that matter are copied into `docs/BENCHMARKS.md`
after a run, not invented here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llm_fabric.eval.provenance import current_commit

ARTIFACT_ROOT = Path("artifacts/bench")
METRIC_VERSION = "perf-metrics-v1"


def write_artifact(
    kind: str,
    payload: dict[str, Any],
    *,
    root: Path = ARTIFACT_ROOT,
    repo: Path | None = None,
) -> Path:
    commit = current_commit(repo)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = root / stamp
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{kind}-{commit or 'nocommit'}.json"
    path = directory / name
    envelope = {
        "kind": kind,
        "metric_version": METRIC_VERSION,
        "commit": commit,
        "timestamp": stamp,
        "payload": payload,
    }
    path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    latest = root / f"{kind}-latest.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path
