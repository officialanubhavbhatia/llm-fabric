"""Portable distribution invariants that do not require Docker or a cluster."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "deployments/docker/Dockerfile"
COMPOSE = ROOT / "deployments/docker/docker-compose.yml"
RUNTIME = ROOT / "src/llm_fabric"


def test_gateway_image_is_multi_stage_non_root_and_weight_free() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "AS dependency-builder" in text
    assert "AS application-build" in text
    assert "AS runtime" in text
    assert "uv sync --frozen --no-dev" in text
    assert "USER 10001:10001" in text
    assert "STOPSIGNAL SIGTERM" in text
    assert 'CMD ["python", "-m", "llm_fabric"]' in text
    assert "COPY config" not in text
    assert "vllm" not in text.lower()
    assert "ollama" not in text.lower()


def test_compose_keeps_gateway_and_inference_as_separate_workloads() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "gateway-mock:" in text
    assert "gateway-ollama:" in text
    assert "\n  ollama:" in text
    assert "ollama-models:/root/.ollama" in text
    assert 'profiles: ["mock"]' in text
    assert 'profiles: ["local"]' in text
    assert "read_only: true" in text
    assert "cap_drop:" in text
    assert "production-test-key" not in text
    prometheus = (ROOT / "deployments/docker/prometheus.yml").read_text(encoding="utf-8")
    assert "gateway-mock:47317" in prometheus
    assert "gateway-ollama:47317" in prometheus
    assert "ollama/ollama:0.32.15" in text
    assert "models.local.yaml" in text
    assert "models.ollama-grades.yaml" in (ROOT / "Makefile").read_text(encoding="utf-8")


def test_runtime_has_no_cloud_platform_branching() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME.rglob("*.py")).lower()
    for expression in (
        'cloud == "aws"',
        'cloud == "azure"',
        'cloud == "gcp"',
        "llm-fabric-aws",
        "llm-fabric-azure",
        "llm-fabric-gcp",
    ):
        assert expression not in source


def test_kind_entrypoints_exist_and_are_executable() -> None:
    for name in ("up.sh", "smoke.sh", "down.sh", "ollama-pull.sh"):
        path = ROOT / "deployments/kind" / name
        assert path.is_file()
        assert path.stat().st_mode & 0o111
    pull = ROOT / "deployments/docker" / "pull-ollama-grades.sh"
    assert pull.is_file()
    assert pull.stat().st_mode & 0o111
