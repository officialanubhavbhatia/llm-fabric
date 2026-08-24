"""Pinned short / medium / large chats against a running Fabric gateway.

Used to check the local Ollama grade ladder. This is not the mock saturation
harness (`llm-fabric-load`): Ollama cannot absorb 64 concurrent generators, and
that tool is duration-based rather than request-count-based.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SHORT_PROMPT = "Reply with one short sentence: what is a hash map?"
MEDIUM_PROMPT = (
    "Explain in about two paragraphs how a hash map handles collisions, "
    "when you would pick chaining versus open addressing, and one coding "
    "pitfall in Python dicts versus a hand-written table."
)
LARGE_PROMPT = " ".join(["Explain the trade-offs in distributed rate limiting."] * 40)

PROMPTS = {
    "short": SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "large": LARGE_PROMPT,
}


def _chat(
    host: str,
    port: int,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    url = f"http://{host}:{port}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode())
            headers = {k.lower(): v for k, v in response.headers.items()}
            return {
                "ok": True,
                "status": response.status,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "served_model": headers.get("x-fabric-served-model"),
                "provider": headers.get("x-fabric-provider"),
                "tier": headers.get("x-fabric-selected-tier"),
                "error": None,
                "id": payload.get("id"),
            }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        return {
            "ok": False,
            "status": exc.code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "served_model": None,
            "provider": None,
            "tier": None,
            "error": detail,
            "id": None,
        }
    except Exception as exc:  # noqa: BLE001 — record transport failures
        return {
            "ok": False,
            "status": 0,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "served_model": None,
            "provider": None,
            "tier": None,
            "error": f"{type(exc).__name__}: {exc}",
            "id": None,
        }


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows if row["ok"]]
    statuses: dict[str, int] = {}
    providers: dict[str, int] = {}
    errors: dict[str, int] = {}
    for row in rows:
        key = str(row["status"])
        statuses[key] = statuses.get(key, 0) + 1
        provider = row.get("provider") or "none"
        providers[provider] = providers.get(provider, 0) + 1
        if not row["ok"]:
            label = (row.get("error") or "unknown")[:80]
            errors[label] = errors.get(label, 0) + 1
    latencies_sorted = sorted(latencies)

    def pct(p: float) -> float | None:
        if not latencies_sorted:
            return None
        last = len(latencies_sorted) - 1
        index = min(last, int(round((p / 100) * last)))
        return latencies_sorted[index]

    return {
        "requests": len(rows),
        "ok": sum(1 for row in rows if row["ok"]),
        "failed": sum(1 for row in rows if not row["ok"]),
        "statuses": statuses,
        "providers": providers,
        "errors": errors,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": max(latencies) if latencies else None,
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else None,
    }


def _models(host: str, port: int, timeout_s: float) -> list[str]:
    url = f"http://{host}:{port}/v1/models"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode())
    return [item["id"] for item in payload.get("data", [])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47317)
    parser.add_argument("--requests", type=int, default=1000, help="Chats per size.")
    parser.add_argument("--model", default="auto", help="Deployment or alias for the load run.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--smoke-model",
        action="append",
        default=[],
        help="Deployment id to smoke (repeatable). Default: every gNN-* from /v1/models.",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ollama-grades/chat-check.json"),
    )
    args = parser.parse_args()

    listed = _models(args.host, args.port, args.timeout)
    smoke_ids = args.smoke_model or [
        item for item in listed if item.startswith("g") and item[1:3].isdigit()
    ]
    payload: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "host": args.host,
        "port": args.port,
        "listed_models": listed,
        "load_model": args.model,
        "smoke": [],
        "workloads": {},
    }

    if not args.skip_smoke:
        for model in smoke_ids:
            row = _chat(
                args.host,
                args.port,
                model=model,
                prompt=SHORT_PROMPT,
                max_tokens=8,
                timeout_s=args.timeout,
            )
            row["model"] = model
            payload["smoke"].append(row)
            status = "PASS" if row["ok"] else "FAIL"
            print(
                f"smoke {status} {model} {row['status']} "
                f"{row['latency_ms']}ms provider={row['provider']}",
                flush=True,
            )

    if not args.skip_load:
        for name, prompt in PROMPTS.items():
            rows: list[dict[str, Any]] = []
            for index in range(args.requests):
                row = _chat(
                    args.host,
                    args.port,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout,
                )
                rows.append(row)
                if (index + 1) % 50 == 0 or not row["ok"]:
                    print(
                        f"{name} {index + 1}/{args.requests} "
                        f"ok={row['ok']} status={row['status']} "
                        f"{row['latency_ms']}ms served={row['served_model']}",
                        flush=True,
                    )
            summary = _summarise(rows)
            payload["workloads"][name] = summary
            print(f"{name} summary {json.dumps(summary)}", flush=True)

    payload["finished_at"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)

    smoke_failed = sum(1 for row in payload["smoke"] if not row["ok"])
    load_failed = sum(int(item.get("failed", 0)) for item in payload["workloads"].values())
    return 1 if smoke_failed or load_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
