# MyVista LLM Fabric

An LLM gateway with policy-based routing across providers. Callers send one
OpenAI-compatible request; the fabric decides which model on which backend serves
it, fails over when a backend breaks, and reports what it chose and what it cost.

Repository: [github.com/officialanubhavbhatia/llm-fabric](https://github.com/officialanubhavbhatia/llm-fabric)

## Run it

No credentials needed. The default registry enables only the `mock` provider,
which performs no inference and returns text assembled from the request — enough
to exercise the whole path end to end.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
python -m llm_fabric
```

The gateway listens on `http://127.0.0.1:47317`. The generated API reference is at
[`/docs`](http://127.0.0.1:47317/docs).

```bash
# Let the fabric choose, then see what it picked
curl -si http://127.0.0.1:47317/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}' \
  | grep -E 'x-fabric-(served-model|policy)'

# Stream
curl -N http://127.0.0.1:47317/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Usage and recent routing decisions
curl -s http://127.0.0.1:47317/v1/usage
```

Because the dialect is OpenAI-compatible, existing clients need only a base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:47317/v1", api_key="unused")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.model)  # the model that actually served the request
```

## Use real providers

Two steps, both required — the fabric will not silently route to an unpriced or
uncredentialed model.

1. Set the key: `LLM_FABRIC_OPENAI_API_KEY` or `LLM_FABRIC_ANTHROPIC_API_KEY`.
2. In `config/models.yaml`, set `enabled: true` on the model and fill in
   `input_cost_per_mtok` / `output_cost_per_mtok` from your provider's current
   pricing page. Prices are operator-supplied inputs, and the `cheapest` policy
   ranks candidates using exactly these numbers.

Copy `.env.example` to `.env` for the full set of settings.

## How it works

```
client → gateway (auth, limits, normalize)
       → router  (registry + policy → decision)
       → serving (provider adapter)
       → metered response with provenance
```

Name a model to pin it, or send an alias like `auto` to route by policy.
`cheapest` orders candidates by blended registry price; `declared` follows
registry order. When a backend fails with a retryable error the router advances
down the fallback chain — except mid-stream, where failing over would splice two
models' output together.

Every response reports the model that served it, the policy that chose it, and how
many candidates failed first, via `x-fabric-*` headers and `/v1/usage`.

[`ARCHITECTURE.md`](ARCHITECTURE.md) covers the layers and, importantly, what is
**not** built. [`docs/CONTRACT.md`](docs/CONTRACT.md) records which request fields
are honoured and which are accepted but inert. Design decisions are in
[`docs/adr/`](docs/adr/).

## Tests

```bash
pytest          # runs offline against injected mock providers
ruff check .
```

The suite needs no credentials and makes no network calls.

## Status

Phase 1. Built: the gateway surface, the routing engine, provider adapters for
OpenAI and Anthropic, and per-request metering.

Not built, and not pretended otherwise: self-hosted model serving, rate limiting
and quotas, circuit breaking, response caching, and durable metering — the
metering sink is in-memory and is lost on restart.

No performance benchmarks have been run, and none are claimed anywhere in this
repository.
