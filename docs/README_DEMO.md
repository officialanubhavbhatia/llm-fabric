# Refreshing README visuals

Reproduce Command Center screenshots and the demo GIF from a running gateway.
No production users, tokens, or private prompts. No private infrastructure is
required for the basic (mock) demo.

The screenshot and GIF in this repository were captured on 2026-08-25 against a
local development gateway on `127.0.0.1:47317` with `auth_mode=dev`, identity
`tenant_demo` / `user_demo` / `project_demo`, and the in-tree `myvista` SDK.
They are that process's meter, not production telemetry. The mock provider is
used unless you point the registry at a live runtime.

## Environment

| Item | Basic demo |
| --- | --- |
| OS | macOS or Linux with Python 3.12+ and Google Chrome |
| Python | `uv sync` in this repo |
| Model runtime | **mock** (no GPU, no Ollama required) |
| Optional | Ollama or vLLM — only if you want those topologies; do not fake vLLM screens on a mock/Ollama-only process |

## 1. Start a gateway

Pick one. Do not mix ports.

**Local mock + demo identity (matches the committed screenshot):**

```bash
git clone https://github.com/officialanubhavbhatia/llm-fabric.git
cd llm-fabric
cp .env.example .env
export LLM_FABRIC_ENVIRONMENT=development
export LLM_FABRIC_AUTH_MODE=dev
export LLM_FABRIC_DEV_AUTH_SECRET=phase4-readme-demo-secret-32chars
export LLM_FABRIC_ALLOW_ANONYMOUS=true
export LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED=true
export LLM_FABRIC_INTENT_ALLOW_HASHING_EMBEDDER=true
unset LLM_FABRIC_API_KEYS LLM_FABRIC_API_CREDENTIALS
unset LLM_FABRIC_DATABASE_URL LLM_FABRIC_REDIS_URL
uv run python -m llm_fabric
```

Gateway: `http://127.0.0.1:47317`  
Command Center: `http://127.0.0.1:47317/command-center`  
Dev token issuer: `POST /v1/dev/token` (mounted only when `auth_mode=dev`)

**Mock Compose (anonymous `public` / `anonymous`, no `.env`):**

```bash
make docker-up
make docker-test
```

The mock Compose profile does not start PostgreSQL; Command Center then uses
the in-process meter. It does not mint `tenant_demo` unless you also set
`LLM_FABRIC_DEV_AUTH_SECRET` on that container.

**Docker Desktop + Ollama** (operator step; live model pull required):

```bash
make docker-desktop-ollama
```

Do **not** capture vLLM KV frames on this stack.

## 2. Demo environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MYVISTA_BASE_URL` | `http://127.0.0.1:47317` | Gateway |
| `MYVISTA_API_KEY` | unset | Skip token mint if already issued |
| `MYVISTA_DEMO_MODEL` | `auto` | Chat model |
| `CHROME_BIN` | macOS Chrome path | Screenshot capture |
| `CDP_PORT` | `9229` | Chrome DevTools port |

## 3. Generate traffic (SDK)

```bash
uv run python3 scripts/demo/readme_demo.py
```

This uses `myvista.MyVista`:

1. `POST /v1/dev/token` for `tenant_demo` / `user_demo` / `project_demo` when the issuer is mounted.
2. Five buffered chat completions.
3. One streaming chat (TTFT path when the gateway records first-byte).
4. Three `intents.classify` calls.
5. One `routes.preview` (no inference).
6. Reads `/v1/observability/dashboards/overview`.

Writes `artifacts/demo/last-sdk.json`, `artifacts/demo/sdk-request.html`, and
`artifacts/demo/demo.token` when a token was minted.

Example chat body:

```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 32
}
```

## 4. Capture screenshots

Requires Google Chrome. Writes real UI pixels; it does not invent metrics.

```bash
uv run --with websockets python3 scripts/demo/capture_command_center.py
```

If `artifacts/demo/demo.token` exists, the capture injects it into the Command
Center token field so the views show `tenant_demo` traffic.

Outputs under `docs/assets/`:

- `sdk-request.png` (real SDK request/response page)
- `command-center.png` (overview)
- `command-center-requests.png`
- `command-center-traces.png`
- `command-center-intents.png`
- `command-center-context.png`
- `command-center-models.png`
- `command-center-kv_cache.png`
- `command-center-routing.png`

On a mock process the KV view must show **unavailable** Ollama/vLLM series.
Do not Photoshop a vLLM scrape.

## 5. Rebuild the GIF

About 28 seconds (8 frames × 3.5 s). Real captures only.

```bash
uv run --with pillow python3 scripts/demo/build_demo_gif.py
```

If GIF quality is poor, keep the PNGs and export an MP4 from them with your
local tools; do not replace frames with invented numbers.

## 6. Regenerating SVGs

```bash
python3 scripts/demo/render_readme_svgs.py
```

Shared palette matches Command Center (`#0f1419` / `#1a222c` / `#6ea8fe`).
System font stack only — no external font files. Do not edit numbers in
`intentos-cascade.svg` by hand — copy them from
`datasets/eval/intentos/final-2026.08.24.json`.

Assets written:

- `docs/assets/myvista-overview.svg`
- `docs/assets/inference-topologies.svg`
- `docs/assets/intentos-cascade.svg`
- `docs/assets/context-pipeline.svg`
- `docs/assets/observability-pipeline.svg`
- `docs/assets/kv-cache-observability.svg`
- `docs/assets/usage-metering.svg`
- `docs/assets/eval-loop.svg`
- `docs/assets/dependency-health.svg`
- plus `command-center-overview.svg`, `routing.svg`, `guardrails.svg`

## Recording steps (manual, optional)

1. Start the gateway as in §1.
2. Run `uv run python3 scripts/demo/readme_demo.py`.
3. Open `http://127.0.0.1:47317/command-center` and paste the demo token.
4. Walk: overview → requests (open a row) → traces → intents → context → models → kv_cache.
5. On mock/Ollama-only, leave KV as unavailable. Do not splice in a vLLM screenshot from another system.

Do not Photoshop metrics. Do not commit production traces.
