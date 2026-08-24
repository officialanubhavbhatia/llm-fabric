# Azure AKS

**Status:** Helm example rendered. **Not live-tested** on AKS.

Use the same chart as kind/EKS/GKE:
[`deployments/helm/llm-fabric`](../../deployments/helm/llm-fabric) with
[`deployments/helm/examples/aks-values.yaml`](../../deployments/helm/examples/aks-values.yaml).

This repository does not provision an AKS cluster, node pools, Application
Gateway, or Azure OpenAI. Start from a cluster that already exists.

## Image

Push the same multi-arch Fabric image to ACR. Pin `image.digest` in production
values. Do not leave `digest: ""` on a live cluster.

## Identity and secrets

The example sets ServiceAccount annotations for Azure Workload Identity. The
Fabric process does not call Azure SDKs. Use Workload Identity plus External
Secrets / Key Vault CSI to populate Kubernetes Secret `llm-fabric` with
`LLM_FABRIC_DATABASE_URL`, `LLM_FABRIC_MIGRATION_DATABASE_URL`,
`LLM_FABRIC_REDIS_URL`, and provider credentials.

## Inference

Point `providers.baseUrls` at in-cluster vLLM Services or private endpoints.
GPU node pools and the vLLM Deployment are operator-owned
([`deployments/kubernetes/vllm-reference.yaml`](../../deployments/kubernetes/vllm-reference.yaml)).

## Ingress

`ingress.enabled` is false in the example. When enabling it, set
`ingress.className` to the cluster's Application Gateway or NGINX class and
terminate TLS at the ingress.

## Install

```bash
helm upgrade --install llm-fabric deployments/helm/llm-fabric \
  --namespace llm-fabric \
  --create-namespace \
  -f deployments/helm/examples/aks-values.yaml \
  -f path/to/secrets-and-digest.yaml
```
