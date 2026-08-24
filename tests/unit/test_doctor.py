"""The doctor reports authentication prerequisites without starting the server."""

from __future__ import annotations

from llm_fabric.config import Settings
from llm_fabric.doctor import collect_checks, format_report, run_doctor


def test_doctor_fails_closed_for_unauthenticated_production() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        api_keys=[],
    )

    checks = collect_checks(settings)
    statuses = {name: status for name, status, _ in checks}

    assert statuses["startup validation"] == "FAIL"
    assert run_doctor(settings) == 1
    assert "FAIL" in format_report(checks)


def test_doctor_fails_production_when_durable_backends_are_missing() -> None:
    settings = Settings(
        _env_file=None,
        environment="production",
        allow_anonymous=False,
        oidc_issuer="https://issuer.example",
        oidc_audience="myvista-llm-fabric",
    )

    checks = collect_checks(settings)
    statuses = {name: status for name, status, _ in checks}

    assert statuses["environment"] == "PASS"
    assert statuses["startup validation"] == "FAIL"
    assert statuses["database"] == "FAIL"
    assert statuses["redis"] == "FAIL"
    assert statuses["application ddl privileges"] == "FAIL"
    assert statuses["runtime health: database"] == "WARN"
    assert statuses["runtime health: redis"] == "WARN"
    assert run_doctor(settings) == 1


def test_doctor_warns_when_development_is_anonymous() -> None:
    settings = Settings(
        _env_file=None,
        environment="development",
        api_keys=[],
        allow_anonymous=True,
    )

    checks = collect_checks(settings)
    statuses = {name: status for name, status, _ in checks}

    assert statuses["environment"] == "WARN"
    assert statuses["startup validation"] == "WARN"
    assert run_doctor(settings) == 0
    report = format_report(checks)
    assert "CONFIGURATION" in report
    assert "CURRENT RUNTIME HEALTH" in report
    assert "runtime health: database" in {name for name, _, _ in checks}
