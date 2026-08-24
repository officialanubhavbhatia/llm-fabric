"""`llm-fabric-usage` — reconcile Redis fast counters with the usage ledger."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from llm_fabric.config import get_settings
from llm_fabric.storage.postgres import create_database_engine, init_schema
from llm_fabric.storage.redis import connect_redis
from llm_fabric.storage.usage import RedisUsageAggregates, UsageLedger

MATCH = "MATCH"
DRIFT = "DRIFT"
REPAIRED = "REPAIRED"
FAILED = "FAILED"


def _compare_bucket(ledger: dict[str, Any], redis: dict[str, int]) -> dict[str, Any]:
    fields = ("invocations", "prompt_tokens", "completion_tokens", "requests")
    diffs = {}
    for field in fields:
        left = int(ledger.get(field) or 0)
        right = int(redis.get(field) or 0)
        if left != right:
            diffs[field] = {"ledger": left, "redis": right}
    return diffs


def reconcile(
    ledger: UsageLedger,
    aggregates: RedisUsageAggregates | None,
    *,
    repair: bool = False,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    try:
        buckets = ledger.day_buckets(tenant_id=tenant_id, observe=tenant_id is None)
    except Exception as exc:  # noqa: BLE001
        return {"status": FAILED, "error": str(exc), "buckets": []}

    reports: list[dict[str, Any]] = []
    drifted = 0
    repaired = 0
    matched = 0
    for bucket in buckets:
        redis_values = aggregates.snapshot(bucket["tenant_id"], bucket["day"]) if aggregates else {}
        diffs = _compare_bucket(bucket, redis_values)
        row: dict[str, Any] = {
            "tenant_id": bucket["tenant_id"],
            "day": bucket["day"],
            "ledger": {
                "invocations": bucket["invocations"],
                "prompt_tokens": bucket["prompt_tokens"],
                "completion_tokens": bucket["completion_tokens"],
                "requests": bucket["requests"],
            },
            "redis": redis_values,
            "diffs": diffs,
        }
        if not diffs:
            row["status"] = MATCH
            matched += 1
        elif repair and aggregates is not None:
            aggregates.replace_day(
                bucket["tenant_id"],
                bucket["day"],
                {
                    "invocations": bucket["invocations"],
                    "prompt_tokens": bucket["prompt_tokens"],
                    "completion_tokens": bucket["completion_tokens"],
                    "requests": bucket["requests"],
                },
            )
            row["status"] = REPAIRED
            repaired += 1
        else:
            row["status"] = DRIFT
            drifted += 1
        reports.append(row)

    if drifted and not repaired or repaired and drifted:
        status = DRIFT
    elif repaired:
        status = REPAIRED
    else:
        status = MATCH
    return {
        "status": status,
        "matched": matched,
        "drifted": drifted,
        "repaired": repaired,
        "failed": 0,
        "buckets": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-fabric-usage",
        description="Reconcile Redis usage counters against the PostgreSQL ledger.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("reconcile", help="compare ledger totals with Redis fast counters")
    rec.add_argument("--repair", action="store_true", help="overwrite Redis from the ledger")
    rec.add_argument("--tenant", help="limit to one tenant (default: all, operator)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    if not settings.database_url:
        print("LLM_FABRIC_DATABASE_URL is required to reconcile usage", file=sys.stderr)
        return 2
    engine = create_database_engine(settings.database_url)
    if settings.environment != "production":
        init_schema(engine)
    ledger = UsageLedger(engine)
    aggregates = None
    if settings.redis_url:
        aggregates = RedisUsageAggregates(connect_redis(settings.redis_url))
    report = reconcile(ledger, aggregates, repair=args.repair, tenant_id=args.tenant)
    print(json.dumps(report, indent=2))
    if report["status"] == FAILED:
        return 2
    if report["status"] == DRIFT and not args.repair:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
