# Model evaluation artifacts

This directory is **not** IntentOS. Do not mix these files with
`datasets/eval/intentos/` (frozen 98).

| File | Meaning |
| --- | --- |
| `workloads.jsonl` | Deterministic category prompts for `llm-fabric eval models` |
| `leaderboard.json` | Latest mock-scored leaderboard (unknown metrics stay `null`) |
| `model-leaderboard.json` | Same payload, name requested by the model-fabric eval phase |
| `model-probes.json` / `model-eval.json` | Latest probe and eval dumps |
| `routing-eval.json` / `routing-shadow.json` | Offline planner quality metrics |
| `index.json` | Deployment → probe/eval/shadow/approval evidence index |
| `promotion-state.json` | CLI-managed lifecycle overlay and audit history (created on first transition) |
| `*-promotion-2026.08.24.json` | Promotion-phase evidence; vLLM files honestly record unavailable when no real endpoint exists |
| `*-YYYY.MM.DD.json` | Versioned probe/eval dumps |

Unknown values are JSON `null`, never `0`.
