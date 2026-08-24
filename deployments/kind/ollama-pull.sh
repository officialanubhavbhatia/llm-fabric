#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-llama3.2}"
CLUSTER="${KIND_CLUSTER_NAME:-llm-fabric}"
NAMESPACE="${LLM_FABRIC_NAMESPACE:-llm-fabric}"

kubectl --context "kind-$CLUSTER" --namespace "$NAMESPACE" \
  exec deployment/llm-fabric-ollama -- ollama pull "$MODEL"
