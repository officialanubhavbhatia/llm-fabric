"""Write versioned model-eval artifacts without secrets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path("datasets/eval/models")


def artifact_path(prefix: str, directory: Path = DEFAULT_DIR) -> Path:
    stamp = datetime.now(UTC).strftime("%Y.%m.%d")
    return directory / f"{prefix}-{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
