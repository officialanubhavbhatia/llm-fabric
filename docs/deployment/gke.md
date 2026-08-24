# Google GKE

**Status:** Helm example rendered. **Not live-tested** on GKE.

Use the same chart as kind/AKS/EKS:
[`deployments/helm/llm-fabric`](../../deployments/helm/llm-fabric) with
[`deployments/helm/examples/gke-values.yaml`](../../deployments/helm/examples/gke-values.yaml).

This repository does not provision a GKE cluster, node pools, or Vertex AI.
Start from a cluster that already exists.

## Image

Push the same multi-arch Fabric image to Artifact Registry. Pin `image.digest`
in production values.

## Identity and secrets

The example sets `iam.gke.io/gcp-service-account` (Workload Identity). The
Fabric process does not call GCP SDKs. Use Workload Identity plus External
Secrets / Secret Manager CSI to populate Kubernetes Secret `llm-fabric`.

## Inference

Point `providers.baseUrls` at in-cluster vLLM Services. GPU node pools are
operator-owned
([`deployments/kubernetes/vllm-reference.yaml`](../../deployments/kubernetes/vllm-reference.yaml)).

## Ingress

`ingress.enabled` is false in the example. Set `ingress.className` to the
cluster GCE or Gateway class when enabling it.

## Install

```bash
helm upgrade --install llm-fabric deployments/helm/llm-fabric \
  --namespace llm-fabric \
  --create-namespace \
  -f deployments/helm/examples/gke-values.yaml \
  -f path/to/secrets-and-digest.yaml
```
