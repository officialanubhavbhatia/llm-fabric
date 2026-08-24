"""`llm-fabric-load`: measure gateway throughput and latency.

Points at a already-running gateway rather than starting one, because the way
the server is run — worker count, event loop, reload on or off — is exactly what
is being measured, and a harness that starts the server for you hides it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_fabric.bench.load import (
    WORKLOADS,
    LoadResult,
    LoadSettings,
    calibrate,
    run_load,
    summarise,
)
from llm_fabric.bench.store import ARTIFACT_ROOT, write_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-load",
        description="Measure gateway throughput and latency under a named workload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Workloads:\n"
        + "\n".join(f"  {name:<14} {w.description}" for name, w in WORKLOADS.items()),
    )
    parser.add_argument(
        "--workload",
        default="chat-short",
        choices=sorted(WORKLOADS),
        help="Which request shape to send. Default: chat-short.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=None, help="Bearer token, when auth is enabled.")
    parser.add_argument("--duration", type=float, default=20.0, help="Measured seconds.")
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="Seconds discarded before measuring, so import and JIT costs do not count.",
    )
    parser.add_argument("--connections", type=int, default=64)
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Generator processes. One Python process cannot saturate a fast server.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help=(
            "Offer this many requests per second regardless of how the server "
            "copes (open loop). Without it, load is closed-loop and measures "
            "capacity instead."
        ),
    )
    parser.add_argument(
        "--target-rps",
        type=float,
        default=None,
        help="Fail with exit code 1 if achieved throughput is below this. For CI gates.",
    )
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=None,
        help="Fail with exit code 1 if p99 latency exceeds this.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.0,
        help="Fail with exit code 1 above this error rate. Default: any error fails.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Also measure the generator's own ceiling, to prove it was not the limit.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write the result as JSON.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Also write a versioned copy under this directory (default: artifacts/bench).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = LoadSettings(
        workload=WORKLOADS[args.workload],
        host=args.host,
        port=args.port,
        duration_s=args.duration,
        warmup_s=args.warmup,
        connections=args.connections,
        processes=args.processes,
        token=args.token,
        rate=args.rate,
    )

    result = run_load(settings)
    payload = result.as_dict()

    if args.calibrate:
        payload["calibration"] = calibrate(settings)

    print(summarise(result))

    if args.calibrate:
        ceiling = payload["calibration"]["generator_ceiling_rps"]
        print(f"\nGenerator ceiling on this machine: {ceiling:,.0f} req/s (liveness workload).")
        if result.achieved_rps > 0.8 * ceiling:
            print(
                "WARNING       the result is within 20% of the generator's own "
                "ceiling, so it is measuring the generator as much as the server."
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.output}")

    artifact_root = ARTIFACT_ROOT if args.artifact_root is None else args.artifact_root
    artifact = write_artifact(
        f"load-{args.workload}",
        payload,
        root=artifact_root,
    )
    print(f"Wrote versioned artifact {artifact}")

    return _gate(args, result)


def _gate(args: argparse.Namespace, result: LoadResult) -> int:
    """Apply the thresholds, and say which one failed rather than just exiting."""
    failures: list[str] = []

    if args.target_rps is not None and result.achieved_rps < args.target_rps:
        failures.append(
            f"throughput {result.achieved_rps:,.0f} rps is below the "
            f"{args.target_rps:,.0f} rps gate"
        )
    if args.max_p99_ms is not None and result.p99_ms > args.max_p99_ms:
        failures.append(f"p99 {result.p99_ms:.2f}ms exceeds the {args.max_p99_ms:.2f}ms gate")
    if result.error_rate > args.max_error_rate:
        failures.append(
            f"error rate {result.error_rate:.4%} exceeds the {args.max_error_rate:.4%} gate"
        )
    if result.offered_load_was_met is False:
        failures.append("the generator did not sustain the requested rate, so the run is invalid")

    if not failures:
        return 0
    print("\nFAILED", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
