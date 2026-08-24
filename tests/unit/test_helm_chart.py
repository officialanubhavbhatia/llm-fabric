"""Helm chart files exist. Rendering a cluster is not claimed from this test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

CHART = Path(__file__).resolve().parents[2] / "deployments" / "helm" / "llm-fabric"


def test_helm_chart_includes_required_objects() -> None:
    templates = CHART / "templates"
    required = (
        "deployment.yaml",
        "service.yaml",
        "configmap.yaml",
        "pdb.yaml",
        "hpa.yaml",
        "migrate-job.yaml",
        "serviceaccount.yaml",
        "ingress.yaml",
        "networkpolicy.yaml",
        "ollama.yaml",
    )
    assert (CHART / "Chart.yaml").is_file()
    assert (CHART / "values.yaml").is_file()
    for name in required:
        assert (templates / name).is_file(), name
    migrate = (templates / "migrate-job.yaml").read_text(encoding="utf-8")
    assert "python" in migrate
    assert "-m" in migrate
    assert "alembic" in migrate
    assert "upgrade" in migrate
    assert "secretRef" in migrate
    assert "configMapRef" not in migrate
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "migrations:" in values
    assert "LLM_FABRIC_MIGRATION_DATABASE_URL" in values
    assert "secretName:" in values
    assert "YAML alone is not verified autoscaling" in values
    assert "  enabled: false" in values
    assert "readOnlyRootFilesystem: true" in values
    assert "allowPrivilegeEscalation: false" in values
    assert 'drop: ["ALL"]' in values
    assert "automountServiceAccountToken: false" in values
    assert "digest:" in values
    hpa = (templates / "hpa.yaml").read_text(encoding="utf-8")
    assert hpa.lstrip().startswith("{{- if .Values.autoscaling.enabled }}")
    assert "observability:" in values
    assert "LLM_FABRIC_OTEL_EXPORTER_OTLP_HEADERS" in values
    configmap = (templates / "configmap.yaml").read_text(encoding="utf-8")
    assert "LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT" in configmap
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in configmap
    assert "LLM_FABRIC_PROVIDER_BASE_URLS" in configmap
    assert "models.yaml" in configmap
    assert "routing.yaml" in configmap


def test_vllm_pool_example_values_exist() -> None:
    example = Path("examples/helm/vllm-pools-values.yaml")
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    assert "LLM_FABRIC_PROVIDER_BASE_URLS" in text or "vllm-general:" in text
    assert "vllm-coding" in text
    assert "vllm-reasoning" in text
    assert "path: /readyz" in text
    assert "path: /healthz" in text


@pytest.mark.parametrize(
    "name",
    (
        "local-values.yaml",
        "local-ollama-values.yaml",
        "aks-values.yaml",
        "eks-values.yaml",
        "gke-values.yaml",
        "vllm-provider-values.yaml",
        "local-ollama-grades-values.yaml",
    ),
)
def test_portable_values_examples_render(name: str) -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("Helm CLI is validated in the dedicated CI job")
    values = Path("deployments/helm/examples") / name
    command = [helm, "template", "llm-fabric", str(CHART), "-f", str(values)]
    if name == "local-ollama-grades-values.yaml":
        command.extend(["--set-file", "fabricConfig.models=config/models.ollama-grades.yaml"])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "kind: Deployment" in result.stdout
    assert "image:" in result.stdout
    if name == "local-ollama-grades-values.yaml":
        assert "g00-smollm2-135m" in result.stdout
        assert "g29-qwen3-1.7b" in result.stdout
        assert "ollama/ollama:0.32.15" in result.stdout


def test_replica_count_two_and_optional_controls_render() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("Helm CLI is validated in the dedicated CI job")
    two = subprocess.run(
        [
            helm,
            "template",
            "llm-fabric",
            str(CHART),
            "--set",
            "replicaCount=2",
            "--set",
            "podDisruptionBudget.enabled=true",
            "--set",
            "autoscaling.enabled=true",
            "--set",
            "ingress.enabled=true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert two.returncode == 0, two.stderr
    # HPA owns replica count when enabled; the Deployment must not hard-code it.
    assert "replicas: 2" not in two.stdout
    assert "kind: HorizontalPodAutoscaler" in two.stdout
    assert "kind: PodDisruptionBudget" in two.stdout
    assert "kind: Ingress" in two.stdout
    replicas = subprocess.run(
        [helm, "template", "llm-fabric", str(CHART), "--set", "replicaCount=2"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert replicas.returncode == 0, replicas.stderr
    assert "replicas: 2" in replicas.stdout
    off = subprocess.run(
        [helm, "template", "llm-fabric", str(CHART)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert off.returncode == 0, off.stderr
    assert "kind: Ingress" not in off.stdout
    assert "kind: PodDisruptionBudget" not in off.stdout
    assert "kind: HorizontalPodAutoscaler" not in off.stdout
    assert "replicas: 1" in off.stdout
    assert 'LLM_FABRIC_INTENT_CLASSIFICATION_ENABLED: "false"' in off.stdout


def test_vllm_provider_example_does_not_auto_approve() -> None:
    text = Path("deployments/helm/examples/vllm-provider-values.yaml").read_text(encoding="utf-8")
    assert "lifecycle: registered" in text
    assert "lifecycle: approved" not in text


def test_chart_keeps_ollama_separate_and_vllm_external() -> None:
    deployment = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    ollama = (CHART / "templates" / "ollama.yaml").read_text(encoding="utf-8")
    assert "name: gateway" in deployment
    assert "name: ollama" not in deployment
    assert "kind: Deployment" in ollama
    assert "nvidia.com/gpu" not in deployment


def test_helm_liveness_is_healthz_and_readiness_is_readyz() -> None:
    """A database outage must not become a liveness restart storm."""
    values = (CHART / "values.yaml").read_text(encoding="utf-8")
    assert "path: /readyz" in values
    assert "path: /healthz" in values
    deployment = (CHART / "templates" / "deployment.yaml").read_text(encoding="utf-8")
    assert "readinessProbe:" in deployment
    assert "livenessProbe:" in deployment
    assert "startupProbe:" in deployment
    assert "{{ .Values.probes.readiness.path }}" in deployment
    assert "{{ .Values.probes.liveness.path }}" in deployment
    # Liveness and startup share /healthz; readiness is /readyz.
    assert "probes:\n  readiness:\n    path: /readyz" in values.replace("\r\n", "\n")
    assert "liveness:\n    path: /healthz" in values
    assert "startup:\n    path: /healthz" in values
