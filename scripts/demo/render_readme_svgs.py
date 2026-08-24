#!/usr/bin/env python3
# ruff: noqa: E501
"""Render the README architecture SVGs with a shared Command Center palette."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets"

BG = "#0f1419"
PANEL = "#1a222c"
LINE = "#2a3542"
TEXT = "#e8eef4"
MUTED = "#8b9aab"
ACCENT = "#6ea8fe"
WARN = "#e6b84d"
BAD = "#e06c75"
OK = "#7fd99a"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


def svg(width: int, height: int, title: str, desc: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <rect width="{width}" height="{height}" fill="{BG}"/>
  {body}
</svg>
'''


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: str | tuple[str, ...] = (),
    stroke: str = LINE,
    dashed: bool = False,
    tag: str | None = None,
    tag_fill: str = MUTED,
) -> str:
    if isinstance(lines, str):
        lines = (lines,)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    text_y = y + 22
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PANEL}" stroke="{stroke}"{dash}/>',
        f'<text x="{x + 12}" y="{text_y}" fill="{TEXT}" font-family="{FONT}" font-size="13" font-weight="600">{title}</text>',
    ]
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 12}" y="{text_y + 18 + i * 15}" fill="{MUTED}" font-family="{FONT}" font-size="11">{line}</text>'
        )
    if tag:
        parts.append(
            f'<rect x="{x + w - 86}" y="{y + 8}" width="74" height="16" rx="8" fill="none" stroke="{tag_fill}"/>'
            f'<text x="{x + w - 49}" y="{y + 20}" text-anchor="middle" fill="{tag_fill}" font-family="{FONT}" font-size="9">{tag}</text>'
        )
    return "\n  ".join(parts)


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = ACCENT) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" marker-end="url(#arrow)"/>'


defs = f'''<defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 z" fill="{ACCENT}"/>
    </marker>
  </defs>'''


def write(name: str, width: int, height: int, title: str, desc: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(svg(width, height, title, desc, defs + "\n  " + body), encoding="utf-8")
    print(name)


def overview() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">MyVista LLM Fabric</text>
  <text x="32" y="56" fill="{MUTED}" font-family="{FONT}" font-size="12">Architecture — serving-path IntentOS routing is OFF. Context compiler is not on the serving path.</text>
  {card(40, 80, 160, 56, title="Developer / SDK", lines=("Python myvista", "curl / OpenAI-shaped"))}
  {arrow(200, 108, 248, 108)}
  {card(250, 80, 170, 56, title="MyVista Gateway", lines=("auth · quotas · /v1/chat", "Command Center"))}
  {arrow(420, 108, 468, 108)}
  {card(470, 72, 200, 72, title="IntentOS", lines=("classify API exists", "serving-path routing OFF"), stroke=WARN, tag="experimental", tag_fill=WARN)}
  {arrow(570, 144, 570, 176)}
  {card(470, 178, 200, 64, title="Context compiler", lines=("not on serving path"), dashed=True, tag="not built", tag_fill=MUTED)}
  {arrow(570, 242, 570, 274)}
  {card(430, 276, 280, 72, title="Route planner", lines=("policy · health · fallback graph", "Grade00–Grade29 declared classes"), tag="built")}
  {arrow(570, 348, 570, 380)}
  {card(250, 382, 640, 52, title="Inference fabric", lines=("Adapters: mock, OpenAI, Anthropic, Ollama, vLLM, generic OpenAI-compatible. LiteLLM is not a native adapter."))}
  {card(250, 444, 145, 52, title="Ollama", lines=("local / Compose"))}
  {card(405, 444, 155, 52, title="OpenAI-compat", lines=("vLLM · custom URLs"))}
  {card(570, 444, 145, 52, title="OpenAI", lines=("hosted API"))}
  {card(725, 444, 165, 52, title="Anthropic", lines=("hosted API"))}
  {card(40, 180, 190, 316, title="Parallel systems", lines=("Guardrails  INPUT+OUTPUT wired", "Usage metering  durable ledger", "Evaluations  first-class gates", "OpenTelemetry  traces/metrics", "Command Center  live views"), stroke=ACCENT)}
  <text x="40" y="538" fill="{MUTED}" font-family="{FONT}" font-size="10">Retrieval / context / execution guardrails are pluggable types. RAG, tools, and KV-cache views are not on the serving path.</text>
"""
    write(
        "myvista-overview.svg",
        920,
        560,
        "MyVista LLM Fabric architecture",
        "SDK to gateway to IntentOS classify, route planner, and inference adapters, with parallel guardrails, usage, evals, OpenTelemetry, and Command Center.",
        body,
    )


def command_center() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">Command Center overview</text>
  <text x="32" y="56" fill="{WARN}" font-family="{FONT}" font-size="12">Observability model — schematic, not a live screenshot. Metrics named here exist in this build unless marked unavailable.</text>
  {card(32, 78, 140, 70, title="Requests/sec", lines=("process buffer rps", "not a fleet SLA"))}
  {card(184, 78, 140, 70, title="Latency", lines=("p50 / p95 / p99 ms", "from usage records"))}
  {card(336, 78, 140, 70, title="Prompt tokens", lines=("ledger + provenance"))}
  {card(488, 78, 140, 70, title="Completion tokens", lines=("ledger + provenance"))}
  {card(640, 78, 140, 70, title="Cost", lines=("registry price × tokens", "estimated vs measured"))}
  {card(792, 78, 140, 70, title="Failovers", lines=("fallback attempts"))}
  {card(32, 168, 280, 150, title="Live views", lines=("overview · users · tenants", "requests · traces · intents", "models · promotion · tiers", "routing · fallbacks · tokens", "economics · evals · drift", "reliability"))}
  {card(328, 168, 280, 150, title="Unavailable views", lines=("threads — no conversation persist", "kv_cache — not scraped", "batching — not scraped", "context — compiler not on path"), dashed=True, tag="not built", tag_fill=MUTED)}
  {card(624, 168, 308, 150, title="Not sourced in this build", lines=("quality / safety scores", "tokens per second (TPS)", "queue depth as a fleet queue", "GPU hours / electricity"), dashed=True)}
  {card(32, 338, 280, 100, title="Route decision", lines=("selected model · provider", "policy · fallback graph"))}
  {card(328, 338, 280, 100, title="IntentOS (frozen eval)", lines=("serving-path routing OFF", "HN gate not cleared"))}
  {card(624, 338, 308, 100, title="Dependency health", lines=("Postgres / Redis", "/healthz vs /readyz"))}
"""
    write(
        "command-center-overview.svg",
        960,
        460,
        "Command Center observability model",
        "Schematic of Command Center metrics that exist: requests, latency, tokens, cost, failovers, plus views that are live versus not built.",
        body,
    )


def observability() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">Observability pipeline</text>
  <text x="32" y="56" fill="{MUTED}" font-family="{FONT}" font-size="12">Built spans: request, auth, input_guardrails, intent, route, llm, output_guardrails. Unbuilt: context, retrieval, tool, eval.</text>
  {card(40, 80, 140, 52, title="SDK request")}
  {arrow(180, 106, 220, 106)}
  {card(222, 80, 140, 52, title="Gateway trace")}
  {arrow(362, 106, 402, 106)}
  {card(404, 80, 140, 52, title="IntentOS span")}
  {arrow(544, 106, 584, 106)}
  {card(586, 80, 140, 52, title="Routing span")}
  {arrow(726, 106, 766, 106)}
  {card(768, 80, 160, 52, title="Provider generation")}
  {arrow(500, 132, 500, 168)}
  {card(40, 170, 200, 64, title="Usage event", lines=("invocation ledger row"))}
  {card(260, 170, 220, 64, title="Guardrail evaluation", lines=("input + output on chat"))}
  {card(500, 170, 200, 64, title="OpenTelemetry", lines=("OTLP HTTP exporter"))}
  {arrow(600, 234, 600, 270)}
  {card(40, 272, 200, 70, title="Traces", lines=("trace_id · request_id"))}
  {card(260, 272, 200, 70, title="Metrics", lines=("bounded Prometheus labels"))}
  {card(480, 272, 200, 70, title="Logs", lines=("structured request logs"))}
  {arrow(600, 342, 600, 378)}
  {card(220, 380, 520, 70, title="Command Center / shared telemetry backend", lines=("local journal is per-process; fleet history belongs in the OTLP backend"))}
  <text x="40" y="480" fill="{MUTED}" font-family="{FONT}" font-size="11">Attributes when present: tenant, intent, route, provider, tokens, latency, cost. Secrets and raw private prompts are not stored by default.</text>
"""
    write(
        "observability-pipeline.svg",
        960,
        500,
        "MyVista observability pipeline",
        "Request flow through gateway, IntentOS, routing, provider, usage, guardrails, and OpenTelemetry traces, metrics, and logs into Command Center.",
        body,
    )


def intentos() -> None:
    layers = [
        (78, "Request", "POST /v1/intents/classify", ACCENT, False, None),
        (148, "L0 Exact cache", "enabled on classify API", ACCENT, False, "default"),
        (218, "L1 Semantic cache", "tenant-isolated cache", ACCENT, False, "default"),
        (288, "L2 Rules", "deterministic rules", ACCENT, False, "default"),
        (358, "L3 Embedding", "default: HashingEmbedder", ACCENT, False, "default"),
        (428, "L4 Structured classifier", "local rerank / model", WARN, True, "optional"),
        (498, "L5 Escalation", "not attached by default", MUTED, True, "off"),
        (568, "ABSTAIN", "unknown / low confidence", WARN, False, "built"),
    ]
    parts = [
        f'<text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">IntentOS cascade</text>',
        f'<text x="32" y="56" fill="{WARN}" font-family="{FONT}" font-size="12">Classify API exists. Serving-path IntentOS routing is OFF. Frozen eval n=98 is a regression tripwire, not production quality.</text>',
    ]
    for i, (y, title, line, color, dashed, tag) in enumerate(layers):
        parts.append(
            card(
                40,
                y,
                360,
                58,
                title=title,
                lines=(line,),
                stroke=color,
                dashed=dashed,
                tag=tag,
                tag_fill=color or MUTED,
            )
        )
        if i < len(layers) - 1:
            parts.append(arrow(220, y + 58, 220, layers[i + 1][0]))
            if title != "Request":
                parts.append(
                    f'<text x="236" y="{y + 70}" fill="{MUTED}" font-family="{FONT}" font-size="10">miss</text>'
                )
    parts.append(
        card(
            430,
            78,
            470,
            220,
            title="Frozen eval (2026-08-24)",
            lines=(
                "accuracy 0.9082",
                "macro-F1 0.9269",
                "high-conf precision 0.964 at threshold 0.9 (n=28, coverage ~0.29)",
                "unknown-intent recall 0.857",
                "hard-negative accuracy 0.50  /  target ≥ 0.58",
                "hard-negative gate not yet cleared",
                "dataset n=98 · artifact datasets/eval/intentos/final-2026.08.24.json",
            ),
            stroke=BAD,
            tag="tripwire",
            tag_fill=WARN,
        )
    )
    parts.append(
        card(
            430,
            318,
            470,
            150,
            title="Also in this cascade",
            lines=(
                "confidence and abstention on every decision",
                "taxonomy version returned by classify",
                "tenant cache isolation",
                "MiniLM embedder is opt-in (--extra embed)",
                "Phase B 30-grade IntentOS planner: not started",
            ),
        )
    )
    write(
        "intentos-cascade.svg",
        940,
        650,
        "IntentOS classification cascade",
        "L0 exact cache through L3 embedding are the default classify path; L4 is optional; L5 is off by default; hard-negative gate is not cleared.",
        "\n  ".join(p for p in parts if p),
    )


def routing() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">Intelligent routing</text>
  <text x="32" y="56" fill="{MUTED}" font-family="{FONT}" font-size="12">Policy-based planner with health, circuit breakers, and a fallback graph. Grade00–Grade29 are declared classes, not 30 production-ranked models.</text>
  {card(32, 80, 150, 64, title="Intent", lines=("classify API only", "not on chat path"), dashed=True, tag="OFF", tag_fill=WARN)}
  {card(194, 80, 150, 64, title="Capabilities", lines=("registry vectors"))}
  {card(356, 80, 150, 64, title="Tenant policy", lines=("quotas · allowlists"))}
  {card(518, 80, 150, 64, title="Latency SLO", lines=("optional preview"))}
  {card(680, 80, 150, 64, title="Cost policy", lines=("seven policies"))}
  {card(842, 80, 140, 64, title="Provider health", lines=("circuit breakers"))}
  {arrow(512, 144, 512, 176)}
  {card(360, 178, 300, 70, title="Route planner", lines=("auditable decision · POST /v1/routes/preview"))}
  {arrow(512, 248, 512, 280)}
  {card(360, 282, 300, 64, title="Primary model", lines=("selected deployment"))}
  {arrow(512, 346, 512, 378)}
  {card(360, 380, 300, 70, title="Fallback graph", lines=("health-aware failover", "each attempt is a usage invocation"))}
  {card(32, 178, 300, 272, title="Honest limits", lines=("Intent-aware serving-path routing: OFF", "Model-grade IntentOS planner: planned", "Context compiler requirement: not built", "Quality cells are declared/measured/unknown", "LiteLLM is not a native adapter"), stroke=WARN)}
"""
    write(
        "routing.svg",
        1000,
        480,
        "MyVista routing and fallback graph",
        "Inputs to the route planner and the primary-then-fallback path. Intent-aware serving-path routing and a 30-grade quality planner are not shipped.",
        body,
    )


def usage() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{
        FONT
    }" font-size="18" font-weight="600">Usage metering</text>
  <text x="32" y="56" fill="{WARN}" font-family="{
        FONT
    }" font-size="12">Durable usage accounting — not billing-grade exactly-once accounting. Crash windows are classified, not hidden.</text>
  {card(40, 80, 180, 60, title="Provider invocation")}
  {arrow(220, 110, 268, 110)}
  {card(270, 80, 180, 60, title="UsageEvent")}
  {arrow(450, 110, 498, 110)}
  {card(500, 80, 220, 60, title="PostgreSQL usage_events")}
  {arrow(720, 110, 768, 110)}
  {card(770, 80, 180, 60, title="Redis counters", lines=("best-effort INCR"))}
  {arrow(610, 140, 610, 176)}
  {card(430, 178, 360, 64, title="Command Center / Economics", lines=("registry prices × tokens"))}
  {
        card(
            40,
            178,
            360,
            200,
            title="What is counted",
            lines=(
                "HTTP request",
                "each provider invocation",
                "fallback attempts",
                "prompt + completion tokens",
                "token provenance",
                "estimated vs measured cost",
            ),
        )
    }
  {
        card(
            40,
            396,
            910,
            84,
            title="Crash window (not exactly-once)",
            lines=(
                "Provider returns then process dies before INSERT → invocation lost.",
                "INSERT succeeds then client retries → a new invocation_id; a second real provider call is counted.",
                "INSERT succeeds and Redis INCR fails → ledger is ahead; reconcile repairs Redis from Postgres, never the reverse.",
            ),
            stroke=WARN,
        )
    }
"""
    write(
        "usage-metering.svg",
        990,
        500,
        "MyVista usage metering architecture",
        "Provider invocations become UsageEvents stored in PostgreSQL with Redis fast counters, shown in Command Center. Not billing-grade exactly-once accounting.",
        body,
    )


def dependency() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">Dependency-aware recovery</text>
  <text x="32" y="56" fill="{MUTED}" font-family="{FONT}" font-size="12">Postgres and Redis are serving dependencies. This is admission control, not fully autonomous self-healing.</text>
  {card(40, 86, 260, 70, title="Postgres / Redis healthy", lines=("serving admitted"), stroke=OK)}
  {arrow(300, 121, 348, 121)}
  {card(350, 86, 200, 70, title="Serving", lines=("chat completions"))}
  {arrow(200, 156, 200, 198)}
  {card(40, 200, 260, 70, title="Dependency fails", lines=("DependencyMonitor"), stroke=BAD)}
  {arrow(300, 235, 348, 235)}
  {card(350, 186, 280, 100, title="UNHEALTHY", lines=("/healthz = 200  (liveness)", "/readyz = 503  (readiness)", "new inference = 503"), stroke=BAD)}
  {arrow(200, 270, 200, 318)}
  {card(40, 320, 260, 70, title="Dependency restored")}
  {arrow(300, 355, 348, 355)}
  {card(350, 306, 200, 52, title="RECOVERING")}
  {arrow(450, 358, 450, 390)}
  {card(350, 392, 280, 70, title="HEALTHY", lines=("serving resumes"), stroke=OK)}
  {card(660, 186, 280, 276, title="Not claimed", lines=("unattended autoscaling", "multi-region failover", "cancelling in-flight provider calls", "CLI heal mutating a live process"), dashed=True)}
"""
    write(
        "dependency-health.svg",
        960,
        490,
        "Dependency-aware recovery",
        "When PostgreSQL or Redis is unhealthy, liveness stays 200, readiness is 503, and new inference is refused until dependencies recover.",
        body,
    )


def eval_loop() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">Eval-first loop</text>
  <text x="32" y="56" fill="{MUTED}" font-family="{FONT}" font-size="12">Evals are first-class. They do not mean every AI behaviour is solved. Agent and safety suites are not built.</text>
  {card(32, 86, 120, 58, title="Change")}
  {arrow(152, 115, 188, 115)}
  {card(190, 86, 120, 58, title="Dataset")}
  {arrow(310, 115, 346, 115)}
  {card(348, 86, 120, 58, title="Eval")}
  {arrow(468, 115, 504, 115)}
  {card(506, 86, 150, 58, title="Regression gate")}
  {arrow(656, 115, 692, 115)}
  {card(694, 86, 160, 58, title="Absolute quality")}
  {arrow(400, 144, 400, 180)}
  {card(250, 182, 140, 58, title="Shadow / canary")}
  {arrow(390, 211, 430, 211)}
  {card(432, 182, 140, 58, title="Promote")}
  {card(32, 270, 220, 140, title="IntentOS", lines=("frozen 98-case tripwire", "HN gate not cleared", "serving-path routing OFF"))}
  {card(268, 270, 220, 140, title="Routing", lines=("planner-match labels", "not routing-quality eval"))}
  {card(504, 270, 220, 140, title="Guardrails", lines=("deterministic engines", "no dedicated safety suite"))}
  {card(740, 270, 200, 140, title="Generation", lines=("named suite ci exists", "DeepEval optional adapter"))}
"""
    write(
        "eval-loop.svg",
        960,
        440,
        "Evaluation loop",
        "Change, dataset, eval, regression and quality gates, then shadow or canary and promote. IntentOS, routing, guardrails, and generation coverage differ.",
        body,
    )


def guardrails() -> None:
    body = f"""
  <text x="32" y="36" fill="{TEXT}" font-family="{FONT}" font-size="18" font-weight="600">Guardrails</text>
  <text x="32" y="56" fill="{MUTED}" font-family="{FONT}" font-size="12">Five-stage model. Only INPUT and OUTPUT run on the chat serving path today. Other stages are pluggable types without RAG or tools to bind to.</text>
  {card(40, 90, 160, 80, title="INPUT", lines=("size limits", "secret/PII-shaped redact", "injection markers"), stroke=OK, tag="wired", tag_fill=OK)}
  {arrow(200, 130, 236, 130)}
  {card(238, 90, 160, 80, title="RETRIEVAL", lines=("no RAG path"), dashed=True, tag="pluggable", tag_fill=MUTED)}
  {arrow(398, 130, 434, 130)}
  {card(436, 90, 160, 80, title="CONTEXT", lines=("compiler not on path"), dashed=True, tag="pluggable", tag_fill=MUTED)}
  {arrow(596, 130, 632, 130)}
  {card(634, 90, 160, 80, title="EXECUTION", lines=("no tool runtime"), dashed=True, tag="pluggable", tag_fill=MUTED)}
  {arrow(794, 130, 830, 130)}
  {card(832, 90, 160, 80, title="OUTPUT", lines=("deterministic engines", "on chat completions"), stroke=OK, tag="wired", tag_fill=OK)}
  {card(40, 210, 952, 90, title="Baseline engines", lines=("Deterministic allow / block / redact / transform / escalate. Model-backed classifiers are adapters and are not on the request path unless configured.", "This is not a claim of solved prompt-injection defence for every flow."))}
"""
    write(
        "guardrails.svg",
        1040,
        330,
        "Five-stage guardrails",
        "INPUT and OUTPUT are wired on chat completions. RETRIEVAL, CONTEXT, and EXECUTION exist as pluggable stages without RAG or tool execution on the serving path.",
        body,
    )


def main() -> None:
    overview()
    command_center()
    observability()
    intentos()
    routing()
    usage()
    dependency()
    eval_loop()
    guardrails()


if __name__ == "__main__":
    main()
