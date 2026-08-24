# Optional dependency classification

The production Fabric image is `uv sync --frozen --no-dev` from
[`pyproject.toml`](../pyproject.toml). Inference engines are never Python
packages of this process.

| Class | Packages | Production image |
| --- | --- | --- |
| **CORE_RUNTIME** | fastapi, uvicorn, pydantic, pydantic-settings, httpx, pyyaml, pyjwt, OpenTelemetry, prometheus-client, sqlalchemy, psycopg, redis, alembic | Yes |
| **DEV_ONLY** | pytest, pytest-asyncio, ruff, mypy, types-PyYAML, fakeredis | No (`--no-dev`) |
| **EVAL_ONLY** | deepeval, lm-eval (`eval-frameworks` extra) | No |
| **EXPERIMENTAL** | fastembed (`embed` extra) | No. HashingEmbedder is the default. MiniLM stays opt-in. |
| **INFERENCE_ENGINE** | Ollama, vLLM | External images / processes only |

Command Center HTML is packaged in the wheel (`hatch` force-include). Config
YAML, datasets, tests, and docs are not copied into the image.
