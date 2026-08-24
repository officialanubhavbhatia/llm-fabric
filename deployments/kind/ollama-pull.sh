#!/usr/bin/env bash
# Pull one Ollama tag, or every Grade00–Grade29 tag when MODEL=all.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-llama3.2}"
CLUSTER="${KIND_CLUSTER_NAME:-llm-fabric}"
NAMESPACE="${LLM_FABRIC_NAMESPACE:-llm-fabric}"
TAGS_FILE="${OLLAMA_GRADE_TAGS:-$ROOT/config/ollama-grade-tags.txt}"

pull_one() {
  local tag="$1"
  kubectl --context "kind-$CLUSTER" --namespace "$NAMESPACE" \
    exec deployment/llm-fabric-ollama -- ollama pull "$tag" </dev/null
}

if [[ "$MODEL" == "all" ]]; then
  ok=0
  fail=0
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    tag="${raw%%#*}"
    tag="$(printf '%s' "$tag" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$tag" ]] && continue
    echo "pull $tag"
    if pull_one "$tag"; then
      echo "PASS $tag"
      ok=$((ok + 1))
    else
      echo "FAIL $tag"
      fail=$((fail + 1))
    fi
  done <"$TAGS_FILE"
  echo "pulled_ok=$ok failed=$fail"
  [[ "$fail" -eq 0 ]]
else
  pull_one "$MODEL"
fi
