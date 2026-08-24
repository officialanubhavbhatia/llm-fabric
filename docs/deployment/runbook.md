# Production runbook

Exact commands for a cluster that already exists. This chart does not provision
cloud accounts, node pools, or GPUs.

Replace `REGISTRY/llm-fabric@sha256:DIGEST` with the published image pin.
Replace `values.yaml` with an operator file derived from
[`deployments/helm/examples`](../../deployments/helm/examples).

## Deploy

```bash
kubectl create namespace llm-fabric --dry-run=client -o yaml | kubectl apply -f -
kubectl -n llm-fabric create secret generic llm-fabric \
  --from-literal=LLM_FABRIC_DATABASE_URL='postgresql://fabric_app:...@postgres:5432/fabric' \
  --from-literal=LLM_FABRIC_MIGRATION_DATABASE_URL='postgresql://fabric:...@postgres:5432/fabric' \
  --from-literal=LLM_FABRIC_REDIS_URL='redis://redis:6379/0' \
  --from-literal=LLM_FABRIC_API_CREDENTIALS='[...]' \
  --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install llm-fabric deployments/helm/llm-fabric \
  --namespace llm-fabric \
  --create-namespace \
  -f values.yaml \
  --set image.repository=REGISTRY/llm-fabric \
  --set image.digest=sha256:DIGEST \
  --wait --timeout 5m
```

`migrations.enabled: true` requires `secretName` and
`LLM_FABRIC_MIGRATION_DATABASE_URL` in that Secret. The migrate Job runs as a
Helm pre-install/pre-upgrade hook.

## Upgrade

```bash
helm upgrade llm-fabric deployments/helm/llm-fabric \
  --namespace llm-fabric \
  -f values.yaml \
  --set image.digest=sha256:NEW_DIGEST \
  --wait --timeout 5m
kubectl -n llm-fabric rollout status deployment/llm-fabric
```

Do not retag an already-published digest. Ship a new immutable image.

## Rollback (Helm release)

```bash
helm history llm-fabric --namespace llm-fabric
helm rollback llm-fabric  --namespace llm-fabric
kubectl -n llm-fabric rollout status deployment/llm-fabric
```

Application rollback assumes Postgres schema is backward-compatible with the
previous image. If a release applied a forward-only Alembic revision, restore
the database from backup before rolling the image back. See
[`BACKUP_RECOVERY.md`](../BACKUP_RECOVERY.md).

## Rollback (image pin)

```bash
helm upgrade llm-fabric deployments/helm/llm-fabric \
  --namespace llm-fabric \
  -f values.yaml \
  --set image.digest=sha256:PREVIOUS_DIGEST \
  --wait
```

## Scale

```bash
kubectl -n llm-fabric scale deployment/llm-fabric --replicas=3
```

Leave `autoscaling.enabled` false until metrics-server is installed and a
capacity test is measured. YAML HPA is not a verified autoscaler.

## Health

```bash
kubectl -n llm-fabric get pods,deploy,svc
curl -fsS https://fabric.example.internal/healthz
curl -fsS https://fabric.example.internal/readyz
```

Liveness and startup use `/healthz`. Readiness uses `/readyz` so a database
outage does not restart pods in a loop.

## Logs

```bash
kubectl -n llm-fabric logs deploy/llm-fabric --tail=200
kubectl -n llm-fabric logs deploy/llm-fabric -f
```

## Metrics and traces

```bash
curl -fsS https://fabric.example.internal/metrics
```

Set `observability.otlp.endpoint` in values. Put
`LLM_FABRIC_OTEL_EXPORTER_OTLP_HEADERS` in the Secret, never the ConfigMap.
Collector outage is fail-soft.

Command Center: `https://fabric.example.internal/command-center`

## Route explain

```bash
curl -fsS https://fabric.example.internal/v1/routes/preview \
  -H 'content-type: application/json' \
  -H 'Authorization: Bearer ...' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
```

## Provider failure

Circuit breakers are process-local. Confirm the provider Service, then:

```bash
curl -fsS https://fabric.example.internal/v1/routes/preview \
  -H 'content-type: application/json' \
  -H 'Authorization: Bearer ...' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'
```

Disable a failing registry row in the `*-files` ConfigMap (`enabled: false`)
and `helm upgrade` or edit the ConfigMap and restart:

```bash
kubectl -n llm-fabric rollout restart deployment/llm-fabric
```

## Model disable

Set `enabled: false` or `lifecycle: disabled` on the registry row in Helm
`fabricConfig.models`, then `helm upgrade`. Public requests cannot write
promotion state.

## Roll back a model

Revert `fabricConfig.models` and, if used, `promotionEvidence.existingConfigMap`
to the previous evidence ConfigMap. A vLLM Service remaining Ready does not
keep a model approved.

## Roll back the Fabric image

Use Helm rollback or re-pin `image.digest` as above. Confirm:

```bash
kubectl -n llm-fabric get deploy llm-fabric -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
curl -fsS https://fabric.example.internal/healthz
```
