#!/usr/bin/env python3
"""Generate deterministic, non-production traffic for README visuals.

Talks to a running MyVista gateway with the in-tree ``myvista`` SDK. When
``auth_mode=dev`` (``LLM_FABRIC_DEV_AUTH_SECRET`` set) this script mints a
token for ``tenant_demo`` / ``user_demo`` / ``project_demo``. Otherwise it uses
the anonymous development principal and prints that fact.

Safe prompts only. No production users, tokens, or private content.

Usage (gateway must already be up):

    python3 scripts/demo/readme_demo.py
    MYVISTA_BASE_URL=http://127.0.0.1:47317 python3 scripts/demo/readme_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from myvista import MyVista

BASE_URL = os.environ.get("MYVISTA_BASE_URL", "http://127.0.0.1:47317").rstrip("/")
API_KEY = os.environ.get("MYVISTA_API_KEY")
OUT = Path(__file__).resolve().parents[2] / "artifacts" / "demo"
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


def _mint_demo_token() -> str | None:
    if API_KEY:
        return API_KEY if API_KEY.lower().startswith("bearer ") else f"Bearer {API_KEY}"
    body = {
        "tenant_id": "tenant_demo",
        "user_id": "user_demo",
        "project_id": "project_demo",
        "roles": ["developer"],
        "scopes": ["chat:write"],
        "ttl_seconds": 3600,
    }
    try:
        response = httpx.post(f"{BASE_URL}/v1/dev/token", json=body, timeout=10.0)
    except httpx.HTTPError:
        return None
    if response.status_code == 404:
        return None
    if not response.is_success:
        print(f"dev token issuer returned HTTP {response.status_code}: {response.text[:300]}")
        return None
    token = response.json().get("access_token")
    return str(token) if token else None


def main() -> int:
    token = _mint_demo_token()
    identity = (
        "tenant_demo / user_demo / project_demo"
        if token and not API_KEY
        else ("supplied MYVISTA_API_KEY" if API_KEY else "anonymous development principal")
    )
    if token and not API_KEY:
        print(f"minted dev token for {identity}")
    elif not token:
        print(
            "no /v1/dev/token (auth_mode is not dev). "
            "Traffic is the anonymous development principal (public / anonymous)."
        )

    api_key = API_KEY or token
    if api_key and str(api_key).lower().startswith("bearer "):
        api_key = str(api_key)[7:].strip()
    client = MyVista(api_key=api_key, base_url=BASE_URL)
    print(f"gateway {BASE_URL}")
    health = client.request("GET", "/healthz")
    ready = client.request("GET", "/readyz")
    print(f"  healthz={health.json().get('status')} readyz={ready.json().get('status')}")

    print("\nSDK chat")
    last: dict[str, Any] = {}
    for prompt in CHAT_PROMPTS:
        chat = client.chat.completions.create(
            model=os.environ.get("MYVISTA_DEMO_MODEL", "auto"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
        )
        last = {
            "identity": identity,
            "request_id": chat.request_id,
            "model": chat.model,
            "served_model": chat.fabric.served_model,
            "provider": chat.fabric.provider,
            "intent": chat.fabric.intent,
            "prompt": prompt,
            "text": chat.text,
            "usage": chat.usage,
        }
        print(
            f"  {chat.request_id}  model={chat.model}  "
            f"served={chat.fabric.served_model}  usage={chat.usage}"
        )

    print("\nSDK streaming chat (TTFT path when the gateway records first-byte)")
    chunks = list(
        client.chat.completions.create(
            model=os.environ.get("MYVISTA_DEMO_MODEL", "auto"),
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=16,
            stream=True,
        )
    )
    streamed = "".join(chunk.delta for chunk in chunks)
    print(f"  stream chunks={len(chunks)} text={streamed[:80]!r}")

    print("\nclassify  (API; serving-path routing remains OFF)")
    for prompt in CLASSIFY_PROMPTS:
        result = client.intents.classify(prompt)
        classification = result.get("classification") or result
        print(
            f"  intent={classification.get('intent_id') or classification.get('intent')}  "
            f"conf={classification.get('confidence')}  layer={classification.get('layer')}"
        )

    print("\nroute preview  (no inference)")
    preview = client.routes.preview(
        model="auto",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    selected = preview.get("selected") or {}
    print(
        f"  selected={selected.get('model_id') or selected.get('id')}  "
        f"provider={selected.get('provider')}  policy={preview.get('policy')}"
    )

    overview = client.request("GET", "/v1/observability/dashboards/overview").json()
    data = overview.get("data") or {}
    coverage = data.get("coverage") or {}
    print("\noverview")
    print(
        f"  requests={data.get('requests')}  errors={data.get('errors')}  "
        f"tokens={data.get('tokens')}"
    )
    print(
        f"  intent_serving={coverage.get('intent_serving')}  "
        f"context_record={coverage.get('context_record')}  "
        f"provenance={coverage.get('supported_telemetry_provenance')}"
    )
    print("  Command Center: " + BASE_URL + "/command-center")

    OUT.mkdir(parents=True, exist_ok=True)
    last["coverage"] = coverage
    (OUT / "last-sdk.json").write_text(json.dumps(last, indent=2), encoding="utf-8")
    if token:
        (OUT / "demo.token").write_text(token, encoding="utf-8")
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        "<title>SDK demo request</title><style>"
        "body{margin:0;font:16px/1.4 ui-sans-serif,system-ui,sans-serif;"
        "background:#0f1419;color:#e8eef4;padding:32px}"
        "h1{font-size:20px}.k{color:#8b9aab}"
        "pre{background:#1a222c;padding:16px;border-radius:8px}"
        "</style></head><body>"
        "<h1>MyVista SDK request (real)</h1>"
        f"<p class='k'>{identity}</p>"
        f"<pre>{json.dumps(last, indent=2)}</pre>"
        "<p class='k'>This page is generated from a live SDK call. "
        "It is not a mocked screenshot.</p>"
        "</body></html>"
    )
    (OUT / "sdk-request.html").write_text(html, encoding="utf-8")
    print(f"  wrote {OUT / 'last-sdk.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
