#!/usr/bin/env python3
"""Generate deterministic, non-production traffic for README visuals.

Talks to a running MyVista gateway over the real HTTP API. Default identity is
the development anonymous principal (tenant ``public``, user ``anonymous``).
It does not mint ``tenant_demo`` unless a Bearer token is supplied — the
development token issuer is only mounted when ``auth_mode=dev``.

Safe prompts only. No production users, tokens, or private content.

Usage (gateway must already be up):

    python3 scripts/demo/readme_demo.py
    MYVISTA_BASE_URL=http://127.0.0.1:47317 python3 scripts/demo/readme_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.environ.get("MYVISTA_BASE_URL", "http://127.0.0.1:47317").rstrip("/")
API_KEY = os.environ.get("MYVISTA_API_KEY")

CHAT_PROMPTS = (
    "Say hello in one sentence.",
    "Name one reason to measure LLM usage.",
    "What is a fallback graph, in one sentence?",
    "Give a one-line definition of TTFT.",
    "List two reasons eval gates exist.",
)
CLASSIFY_PROMPTS = (
    "Write a Python function that reverses a list.",
    "Explain why 17 is prime.",
    "Draft a polite out-of-office reply.",
)


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"accept": "application/json"}
    if payload is not None:
        headers["content-type"] = "application/json"
    if API_KEY:
        headers["authorization"] = (
            API_KEY if API_KEY.lower().startswith("bearer ") else f"Bearer {API_KEY}"
        )
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
            status = response.status
            request_id = response.headers.get("x-fabric-request-id", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {BASE_URL}{path}: {exc.reason}") from exc
    parsed: Any = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise SystemExit(f"{method} {path} returned a non-object")
    parsed["_http_status"] = status
    parsed["_request_id"] = request_id
    return parsed


def main() -> int:
    health = _request("GET", "/healthz")
    ready = _request("GET", "/readyz")
    print(f"gateway {BASE_URL}")
    print(f"  healthz={health.get('status', health)} readyz={ready.get('status', ready)}")

    print("\nchat")
    for prompt in CHAT_PROMPTS:
        result = _request(
            "POST",
            "/v1/chat/completions",
            {
                "model": os.environ.get("MYVISTA_DEMO_MODEL", "auto"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
            },
        )
        model = result.get("model")
        usage = result.get("usage") or {}
        print(
            f"  {result.get('_request_id') or result.get('id')}  "
            f"model={model}  prompt={usage.get('prompt_tokens')}  "
            f"completion={usage.get('completion_tokens')}"
        )

    print("\nclassify  (API only; serving-path IntentOS routing is OFF)")
    for prompt in CLASSIFY_PROMPTS:
        result = _request("POST", "/v1/intents/classify", {"input": prompt, "language": "en"})
        classification = result.get("classification") or result
        intent = classification.get("intent_id") or classification.get("intent")
        confidence = classification.get("confidence")
        layer = classification.get("layer")
        abstain = classification.get("abstain")
        print(f"  intent={intent}  conf={confidence}  layer={layer}  abstain={abstain}")

    print("\nroute preview  (no inference)")
    preview = _request(
        "POST",
        "/v1/routes/preview",
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        },
    )
    selected = preview.get("selected") or {}
    print(
        f"  selected={selected.get('model_id') or selected.get('id')}  "
        f"provider={selected.get('provider')}  policy={preview.get('policy')}"
    )

    overview = _request("GET", "/v1/observability/dashboards/overview")
    data = overview.get("data") or {}
    print("\noverview")
    print(
        f"  requests={data.get('requests')}  rps={data.get('rps')}  "
        f"errors={data.get('errors')}  tokens={data.get('tokens')}"
    )
    print("  Command Center: " + BASE_URL + "/command-center")
    return 0


if __name__ == "__main__":
    sys.exit(main())
