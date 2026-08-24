#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER="${KIND_CLUSTER_NAME:-llm-fabric}"
IMAGE="${LLM_FABRIC_IMAGE:-llm-fabric:dev}"
NAMESPACE="${LLM_FABRIC_NAMESPACE:-llm-fabric}"
VALUES="${LLM_FABRIC_HELM_VALUES:-$ROOT/deployments/helm/examples/local-values.yaml}"
MODELS_FILE="${LLM_FABRIC_MODELS_FILE:-}"
REVISION="$(git -C "$ROOT" rev-parse HEAD)"

for command in docker kind kubectl helm; do
  command -v "$command" >/dev/null || {
    echo "required command not found: $command" >&2
    exit 1
  }
done

docker build \
  --build-arg VERSION=dev \
  --build-arg REVISION="$REVISION" \
  -f "$ROOT/deployments/docker/Dockerfile" \
  -t "$IMAGE" \
  "$ROOT"

cluster_exists=false
while IFS= read -r existing; do
  if [[ "$existing" == "$CLUSTER" ]]; then
    cluster_exists=true
  fi
done < <(kind get clusters)

if [[ "$cluster_exists" == true ]]; then
  if ! docker port "${CLUSTER}-control-plane" 30017 >/dev/null 2>&1; then
    echo "existing kind cluster lacks NodePort 30017 mapping; recreating" >&2
    kind delete cluster --name "$CLUSTER"
    cluster_exists=false
  fi
fi

if [[ "$cluster_exists" == false ]]; then
  kind create cluster \
    --name "$CLUSTER" \
    --config "$ROOT/deployments/kind/cluster.yaml"
fi

kind load docker-image "$IMAGE" --name "$CLUSTER"
helm_args=(
  upgrade --install llm-fabric
  "$ROOT/deployments/helm/llm-fabric"
  --namespace "$NAMESPACE"
  --create-namespace
  -f "$VALUES"
  --set image.repository="${IMAGE%:*}"
  --set image.tag="${IMAGE##*:}"
  --set image.pullPolicy=Never
  --wait
  --timeout 180s
)
if [[ -n "$MODELS_FILE" ]]; then
  helm_args+=(--set-file "fabricConfig.models=$MODELS_FILE")
fi
helm "${helm_args[@]}"

kubectl --context "kind-$CLUSTER" \
  --namespace "$NAMESPACE" \
  rollout status deployment/llm-fabric --timeout=120s

echo "LLM Fabric is available at http://127.0.0.1:47317"
echo "Run: make k8s-local-test"
