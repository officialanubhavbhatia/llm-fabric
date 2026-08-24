# Portable deployment

One Fabric source tree, one OCI image, one Helm chart. Cloud differences belong
in values files and cluster services, not application code.

Detailed runbooks:

- [Production runbook](runbook.md)
- [Azure AKS](aks.md) (Helm-rendered, **not** live-tested)
- [AWS EKS](eks.md) (Helm-rendered, **not** live-tested)
- [Google GKE](gke.md) (Helm-rendered, **not** live-tested)

## Validation matrix

| Environment | Manifest / Helm validated | Live tested |
| --- | --- | --- |
| Docker Compose (`mock`) | yes | yes, local (2026-08-24) |
| Docker Compose (`local` Ollama stack) | yes | stack-up yes; model pull is a separate operator step |
| Docker Compose (`litellm-ollama`) | yes | live chat is an operator step; not claimed from unit tests |
| Docker Compose (`litellm-vllm`) | yes (manifest) | **PENDING** without a real GPU/`VLLM_API_BASE` |
| Docker Compose (`local` + Grade00–Grade29 tags) | yes | pull and chat-check are operator/live steps; not a quality ranking |
| kind | yes | yes (`make k8s-local-test`, 2026-08-24) |
| Azure AKS | yes (`deployments/helm/examples/aks-values.yaml`) | **no** |
| AWS EKS | yes (`deployments/helm/examples/eks-values.yaml`) | **no** |
| Google GKE | yes (`deployments/helm/examples/gke-values.yaml`) | **no** |

Do not treat a successful `helm template` as a cloud go-live.

## Image and chart

- Dockerfile: [`deployments/docker/Dockerfile`](../../deployments/docker/Dockerfile)
- Compose: [`deployments/docker/docker-compose.yml`](../../deployments/docker/docker-compose.yml)
  (`mock`, `local`, `litellm-ollama`, `litellm-vllm`; vLLM is never started by Compose)
- Chart: [`deployments/helm/llm-fabric`](../../deployments/helm/llm-fabric)
  Helm topology examples: `local-ollama-values.yaml` (direct Ollama),
  `local-litellm-ollama-values.yaml`, `vllm-provider-values.yaml` (direct vLLM),
  `local-litellm-vllm-values.yaml`. LiteLLM/Ollama/vLLM Services are ClusterIP.
  Ingress, when enabled, fronts Fabric only.
- Examples: [`deployments/helm/examples`](../../deployments/helm/examples)
- kind: [`deployments/kind`](../../deployments/kind)

The Fabric image contains the gateway, Alembic files for the optional migrate
Job, and the Command Center asset. It does not contain model weights, CUDA,
vLLM, Ollama, or secrets.

Containers must set `LLM_FABRIC_HOST=0.0.0.0` (the process default is
`127.0.0.1`). The Helm ConfigMap already does this.

## Statelessness

| State | Development / test (no DSN) | Production (`DATABASE_URL` + `REDIS_URL`) |
| --- | --- | --- |
| Usage ledger, tenants | Process memory | PostgreSQL |
| Quotas, revocation | Process memory | Redis |
| Circuit breakers, dependency health | Process-local | Process-local (per replica) |
| Promotion overlay | Local file / `/tmp` unless a ConfigMap is mounted | Mount the same evidence ConfigMap on every replica (`promotionEvidence.existingConfigMap`) |

`replicaCount: 2` is safe for mock routing on kind. Production quotas and
usage require Postgres and Redis; do not treat multi-replica mock as a
production capacity result.

## IntentOS and promotion

Helm development values leave serving-path IntentOS off (`SAFE_FALLBACK` still
covers chat). Production examples set `intentClassificationEnabled: "true"`.
The process refuses to start in production if that flag is false. vLLM example
values leave models at `lifecycle: registered`. A Kubernetes Service does not
approve a model.

See also [`DEPENDENCIES.md`](../DEPENDENCIES.md) and [`TOPOLOGY.md`](../TOPOLOGY.md).
