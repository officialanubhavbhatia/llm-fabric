# Refreshing README visuals

Reproduce Command Center screenshots and the demo GIF from a running gateway.
No production users, tokens, or private prompts.

The screenshot and GIF in this repository were captured on 2026-08-25 against a
local Docker Desktop stack (gateway on `127.0.0.1:47317`, anonymous development
principal `public` / `anonymous`, PostgreSQL usage ledger). They are not
production telemetry.

## 1. Start a gateway

Pick one. Do not mix ports.

**Mock (matches the README quick start):**

```bash
git clone https://github.com/officialanubhavbhatia/llm-fabric.git
cd llm-fabric
make docker-up
make docker-test
```

No `.env` file is required for Compose. The mock profile does not start
PostgreSQL; Command Center then uses the in-process meter.

**Local process:**

```bash
cp .env.example .env
export LLM_FABRIC_ENVIRONMENT=development
make dev
```

**Docker Desktop + Ollama** (what the committed screenshot used):

```bash
make docker-desktop-ollama
```

Gateway: `http://127.0.0.1:47317`  
Command Center: `http://127.0.0.1:47317/command-center`

## 2. Demo environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MYVISTA_BASE_URL` | `http://127.0.0.1:47317` | Gateway |
| `MYVISTA_API_KEY` | unset | Bearer token if auth is on |
| `MYVISTA_DEMO_MODEL` | `auto` | Chat model |
| `CHROME_BIN` | macOS Chrome path | Screenshot capture |
| `CDP_PORT` | `9229` | Chrome DevTools port |

Development Compose sets `LLM_FABRIC_ALLOW_ANONYMOUS=true`. Traffic is attributed
to tenant `public`, user `anonymous`. The `/v1/dev/token` issuer (for
`tenant_demo`) is only mounted when `auth_mode=dev`.

## 3. Generate traffic

```bash
python3 scripts/demo/readme_demo.py
```

This calls the real HTTP API: five chats, three classify calls, one route
preview. Safe prompts only.

## 4. Capture screenshots

Requires Google Chrome. Writes real UI pixels; it does not invent metrics.

```bash
uv run --with websockets python3 scripts/demo/capture_command_center.py
```

Outputs under `docs/assets/`:

- `command-center.png` (overview)
- `command-center-requests.png`
- `command-center-traces.png`
- `command-center-intents.png`
- `command-center-routing.png`
- `command-center-economics.png`
- `command-center-reliability.png`

## 5. Rebuild the GIF

The committed GIF is a 20-second walk through those real views (~300 KiB). It
does not include a terminal keystroke recording of `curl` (no `ffmpeg` screen
capture in the capture environment).

```bash
uv run --with pillow python3 - <<'PY'
from pathlib import Path
from PIL import Image

assets = Path("docs/assets")
order = [
    "command-center.png",
    "command-center-requests.png",
    "command-center-traces.png",
    "command-center-intents.png",
    "command-center-routing.png",
    "command-center-economics.png",
    "command-center-reliability.png",
]
width = 960
frames = []
for name in order:
    im = Image.open(assets / name).convert("RGB")
    w, h = im.size
    frames.append(
        im.resize((width, int(h * width / w)), Image.Resampling.LANCZOS).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=64
        )
    )
out = assets / "myvista-demo.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=2800, loop=0, optimize=True, disposal=2)
print(out, out.stat().st_size)
PY
```

## 6. Regenerating SVGs

```bash
python3 scripts/demo/render_readme_svgs.py
```

Shared palette matches Command Center (`#0f1419` / `#1a222c` / `#6ea8fe`).
Do not edit numbers in `intentos-cascade.svg` by hand — copy them from
`datasets/eval/intentos/final-2026.08.24.json`.

## Recording steps (manual, optional)

If you want a longer screen recording instead of the GIF:

1. Run the demo script.
2. Open `http://127.0.0.1:47317/command-center`.
3. Click overview → requests → traces → intents → routing → economics.
4. Export a small MP4 or GIF; keep it well under a few megabytes.

Do not Photoshop metrics. Do not commit production traces.
