"""`llm-fabric-perf` — stage benches, profiling, versioned artifacts.

Does not start a gateway. HTTP load remains `llm-fabric-load` against a server
you already started, because how that server is run is part of what is measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_fabric.bench.stages import OPTIMIZATION_FLAGS, STAGE_NAMES, run_stages
from llm_fabric.bench.store import ARTIFACT_ROOT, write_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-perf",
        description="In-process stage benches and a record of which optimizations are off.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stages = sub.add_parser("stages", help="time each named stage in isolation")
    stages.add_argument("--iterations", type=int, default=2000)
    stages.add_argument("--warmup", type=int, default=100)
    stages.add_argument("--output", type=Path)
    stages.add_argument(
        "--artifact-root",
        type=Path,
        default=ARTIFACT_ROOT,
        help="Directory for versioned JSON. Default: artifacts/bench",
    )

    flags = sub.add_parser("optimizations", help="list techniques and whether they are enabled")
    flags.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "optimizations":
        payload = {
            "flags": list(OPTIMIZATION_FLAGS),
            "note": (
                "A technique becomes a production default only after a measured "
                "run on this hardware shows a benefit. Unbuilt backends are not "
                "enabled as placeholders."
            ),
        }
        text = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    results = run_stages(iterations=args.iterations, warmup=args.warmup)
    payload = {
        "stages": [row.as_dict() for row in results],
        "expected": list(STAGE_NAMES),
        "optimizations": list(OPTIMIZATION_FLAGS),
        "note": (
            "In-process, no sockets. Ollama and vLLM are unavailable because "
            "those adapters are not built. Errors are counted, not dropped."
        ),
    }
    path = write_artifact("stages", payload, root=args.artifact_root)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote versioned artifact {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
