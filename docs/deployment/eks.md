# AWS EKS

**Status:** Helm example rendered. **Not live-tested** on EKS.

Use the same chart as kind/AKS/GKE:
[`deployments/helm/llm-fabric`](../../deployments/helm/llm-fabric) with
[`deployments/helm/examples/eks-values.yaml`](../../deployments/helm/examples/eks-values.yaml).

This repository does not provision an EKS cluster, Karpenter, ALB, or Bedrock.
Start from a cluster that already exists.

## Image

Push the same multi-arch Fabric image to ECR. Pin `image.digest` in production
values.

## Identity and secrets

The example sets `eks.amazonaws.com/role-arn` on the ServiceAccount (IRSA).
The Fabric process does not call AWS SDKs. Use IRSA plus External Secrets /
Secrets Store CSI to populate Kubernetes Secret `llm-fabric`.

## Inference

Point `providers.baseUrls` at in-cluster vLLM Services. GPU capacity is
operator-owned
([`deployments/kubernetes/vllm-reference.yaml`](../../deployments/kubernetes/vllm-reference.yaml)).

## Ingress

`ingress.enabled` is false in the example. ALB annotations in the file are
documentation for when ingress is turned on.

## Install

```bash
helm upgrade --install llm-fabric deployments/helm/llm-fabric \
  --namespace llm-fabric \
  --create-namespace \
  -f deployments/helm/examples/eks-values.yaml \
  -f path/to/secrets-and-digest.yaml
```
