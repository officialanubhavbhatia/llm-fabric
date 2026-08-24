"""Helm chart files exist. Rendering a cluster is not claimed from this test."""

from __future__ import annotations

from pathlib import Path

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
    assert "fabric_app" in values
    assert "YAML alone is not verified autoscaling" in values
    assert "  enabled: false" in values
    hpa = (templates / "hpa.yaml").read_text(encoding="utf-8")
    assert hpa.lstrip().startswith("{{- if .Values.autoscaling.enabled }}")
    assert "observability:" in values
    assert "LLM_FABRIC_OTEL_EXPORTER_OTLP_HEADERS" in values
    configmap = (templates / "configmap.yaml").read_text(encoding="utf-8")
    assert "LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT" in configmap
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in configmap


def test_vllm_pool_example_values_exist() -> None:
    example = Path("examples/helm/vllm-pools-values.yaml")
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    assert "LLM_FABRIC_PROVIDER_BASE_URLS" in text
    assert "vllm-coding" in text
    assert "vllm-reasoning" in text
    assert "path: /readyz" in text
    assert "path: /healthz" in text


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
