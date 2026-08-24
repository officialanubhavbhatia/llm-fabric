#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${LLM_FABRIC_BASE_URL:-http://127.0.0.1:47317}"
CLUSTER="${KIND_CLUSTER_NAME:-llm-fabric}"
NAMESPACE="${LLM_FABRIC_NAMESPACE:-llm-fabric}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl --fail --silent --show-error "$BASE_URL/healthz" >"$TMP/health.json"
curl --fail --silent --show-error "$BASE_URL/readyz" >"$TMP/ready.json"
curl --fail --silent --show-error "$BASE_URL/command-center" >"$TMP/command-center.html"

curl --fail --silent --show-error \
  --dump-header "$TMP/headers" \
  --output "$TMP/chat.json" \
  -H "content-type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"portable smoke test"}]}' \
  "$BASE_URL/v1/chat/completions"

for header in x-fabric-served-model x-fabric-provider x-fabric-selected-tier; do
  awk -v expected="$header:" 'BEGIN {IGNORECASE=1} index(tolower($0), expected) == 1 {found=1} END {exit !found}' \
    "$TMP/headers" || {
      echo "missing routing header: $header" >&2
      exit 1
    }
done

kubectl --context "kind-$CLUSTER" --namespace "$NAMESPACE" \
  rollout restart deployment/llm-fabric
kubectl --context "kind-$CLUSTER" --namespace "$NAMESPACE" \
  rollout status deployment/llm-fabric --timeout=120s
curl --fail --silent --show-error "$BASE_URL/healthz" >/dev/null

echo "kind smoke PASS: health, readiness, Command Center, mock chat, routing headers, restart"
