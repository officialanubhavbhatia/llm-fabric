"""Inspect whether this process is allowed to start.

Doctor is diagnostic only. It is not the serving health mechanism.

Checks are grouped:

- CONFIGURATION — environment name, auth mode, flags
- STARTUP DEPENDENCY — whether required URLs are present and acceptable
- CURRENT RUNTIME HEALTH — a live probe of those dependencies *now*

Serving readiness after startup is decided by the gateway's dependency
monitor and `/readyz`, not by this CLI.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from llm_fabric.config import Settings, get_settings, validate_startup
from llm_fabric.errors import ConfigurationError

CheckStatus = str  # PASS | WARN | FAIL


def _check_environment(settings: Settings) -> tuple[str, CheckStatus, str]:
    if settings.environment == "production":
        return (
            "environment",
            "PASS",
            "LLM_FABRIC_ENVIRONMENT=production",
        )
    return (
        "environment",
        "WARN",
        f"environment is {settings.environment}; only production is fail-closed for auth",
    )


def _check_startup(settings: Settings) -> tuple[str, CheckStatus, str]:
    try:
        validate_startup(settings)
    except ConfigurationError as exc:
        return ("startup validation", "FAIL", exc.message)
    if settings.environment == "production":
        return (
            "startup validation",
            "PASS",
            f"production authentication is configured ({settings.effective_auth_mode})",
        )
    if settings.auth_required:
        return (
            "startup validation",
            "PASS",
            f"authentication is required ({settings.effective_auth_mode})",
        )
    return (
        "startup validation",
        "WARN",
        "authentication is not required; anonymous callers are admitted",
    )


def _check_dev_secret(settings: Settings) -> tuple[str, CheckStatus, str]:
    if settings.environment == "production" and settings.dev_auth_secret:
        return (
            "development issuer",
            "FAIL",
            "LLM_FABRIC_DEV_AUTH_SECRET is set; production forbids the development issuer",
        )
    if settings.dev_auth_secret:
        return (
            "development issuer",
            "WARN",
            "development issuer is enabled; never set this outside local work",
        )
    return ("development issuer", "PASS", "development issuer is not configured")


def _check_anonymous(settings: Settings) -> tuple[str, CheckStatus, str]:
    if settings.environment == "production" and settings.allow_anonymous:
        return (
            "anonymous bypass",
            "FAIL",
            "LLM_FABRIC_ALLOW_ANONYMOUS is enabled in production",
        )
    if settings.allow_anonymous and not settings.auth_required:
        return (
            "anonymous bypass",
            "WARN",
            "anonymous access is explicitly enabled",
        )
    return ("anonymous bypass", "PASS", "anonymous bypass is not in effect")


def _check_database(settings: Settings) -> tuple[str, CheckStatus, str]:
    """STARTUP DEPENDENCY: is a durable store configured?"""
    if not settings.database_url:
        status: CheckStatus = "FAIL" if settings.environment == "production" else "WARN"
        return ("database", status, "LLM_FABRIC_DATABASE_URL is unset; records are in-memory")
    if settings.database_url.startswith("sqlite"):
        if settings.environment == "production":
            return ("database", "FAIL", "sqlite is not an acceptable production store")
        return ("database", "WARN", "database URL is sqlite; not durable across hosts")
    return ("database", "PASS", "LLM_FABRIC_DATABASE_URL is set")


def _check_database_runtime(settings: Settings) -> tuple[str, CheckStatus, str]:
    """CURRENT RUNTIME HEALTH: can this host reach PostgreSQL right now?"""
    if not settings.database_url or settings.database_url.startswith("sqlite"):
        return (
            "runtime health: database",
            "WARN",
            "not probed; no production PostgreSQL URL is configured",
        )
    try:
        from llm_fabric.storage.postgres import probe_database

        probe_database(settings.database_url)
        return ("runtime health: database", "PASS", "PostgreSQL accepted SELECT 1")
    except ConfigurationError as exc:
        return ("runtime health: database", "FAIL", exc.message)


def _check_redis(settings: Settings) -> tuple[str, CheckStatus, str]:
    """STARTUP DEPENDENCY: is Redis configured?"""
    if not settings.redis_url:
        status: CheckStatus = "FAIL" if settings.environment == "production" else "WARN"
        return ("redis", status, "LLM_FABRIC_REDIS_URL is unset; quotas are per process")
    return ("redis", "PASS", "LLM_FABRIC_REDIS_URL is set")


def _check_redis_runtime(settings: Settings) -> tuple[str, CheckStatus, str]:
    """CURRENT RUNTIME HEALTH: can this host reach Redis right now?"""
    if not settings.redis_url:
        return (
            "runtime health: redis",
            "WARN",
            "not probed; no Redis URL is configured",
        )
    try:
        from llm_fabric.storage.redis import probe_redis

        probe_redis(settings.redis_url)
        return ("runtime health: redis", "PASS", "Redis accepted PING")
    except ConfigurationError as exc:
        return ("runtime health: redis", "FAIL", exc.message)


def collect_checks(settings: Settings) -> list[tuple[str, CheckStatus, str]]:
    return [
        _check_environment(settings),
        _check_startup(settings),
        _check_dev_secret(settings),
        _check_anonymous(settings),
        _check_unsafe_flags(settings),
        _check_database(settings),
        _check_database_runtime(settings),
        _check_database_rls_role(settings),
        _check_application_ddl(settings),
        _check_migrations(settings),
        _check_redis(settings),
        _check_redis_runtime(settings),
        _check_inference(settings),
        _check_guardrails(settings),
        _check_telemetry(settings),
    ]


def _check_unsafe_flags(settings: Settings) -> tuple[str, CheckStatus, str]:
    flags: list[str] = []
    if settings.allow_anonymous:
        flags.append("ALLOW_ANONYMOUS")
    if settings.allow_unsafe_multiworker:
        flags.append("ALLOW_UNSAFE_MULTIWORKER")
    if settings.dev_auth_secret:
        flags.append("DEV_AUTH_SECRET")
    if flags and settings.environment == "production":
        return (
            "unsafe development flags",
            "FAIL",
            "production forbids " + ", ".join(flags),
        )
    if flags:
        return (
            "unsafe development flags",
            "WARN",
            "set: " + ", ".join(flags),
        )
    return ("unsafe development flags", "PASS", "no unsafe development flags are set")


def _check_database_rls_role(settings: Settings) -> tuple[str, CheckStatus, str]:
    if not settings.database_url or not settings.database_url.startswith("postgresql"):
        status: CheckStatus = "FAIL" if settings.environment == "production" else "WARN"
        return ("database RLS role", status, "PostgreSQL application role is not in use")
    try:
        from llm_fabric.storage.postgres import create_database_engine, current_role_bypasses_rls

        engine = create_database_engine(settings.database_url)
        bypasses, role = current_role_bypasses_rls(engine)
        engine.dispose()
    except Exception as exc:  # noqa: BLE001 — doctor must surface any failure
        return ("database RLS role", "FAIL", f"could not inspect database role: {exc}")
    if bypasses:
        return (
            "database RLS role",
            "FAIL",
            f"connected as '{role}', which bypasses row-level security",
        )
    return ("database RLS role", "PASS", f"connected as '{role}', subject to RLS")


def _check_application_ddl(settings: Settings) -> tuple[str, CheckStatus, str]:
    """Production serving role must not be able to mutate schema."""
    if not settings.database_url or not settings.database_url.startswith("postgresql"):
        status: CheckStatus = "FAIL" if settings.environment == "production" else "WARN"
        return (
            "application ddl privileges",
            status,
            "PostgreSQL application role is not in use",
        )
    try:
        from llm_fabric.storage.postgres import (
            create_database_engine,
            current_role_schema_privileges,
        )

        engine = create_database_engine(settings.database_url)
        privileges = current_role_schema_privileges(engine)
        engine.dispose()
    except Exception as exc:  # noqa: BLE001
        return ("application ddl privileges", "FAIL", f"could not inspect privileges: {exc}")
    held = [
        name
        for name in (
            "superuser",
            "createdb",
            "createrole",
            "create_on_public",
            "create_on_database",
        )
        if privileges.get(name)
    ]
    role = str(privileges.get("role", "unknown"))
    if held:
        status = "FAIL" if settings.environment == "production" else "WARN"
        return (
            "application ddl privileges",
            status,
            f"connected as '{role}' with DDL privileges: " + ", ".join(held),
        )
    return (
        "application ddl privileges",
        "PASS",
        f"connected as '{role}', DML/USAGE only on public",
    )


def _check_migrations(settings: Settings) -> tuple[str, CheckStatus, str]:
    if not settings.database_url:
        status: CheckStatus = "FAIL" if settings.environment == "production" else "WARN"
        return ("migrations", status, "no database URL; schema cannot be verified")
    try:
        from sqlalchemy import inspect

        from llm_fabric.storage.postgres import create_database_engine

        engine = create_database_engine(settings.database_url)
        from llm_fabric.storage.schema import current_revision, expected_heads

        tables = set(inspect(engine).get_table_names())
        revision = current_revision(engine)
        heads = expected_heads()
        engine.dispose()
    except Exception as exc:  # noqa: BLE001
        return ("migrations", "FAIL", f"could not inspect schema: {exc}")
    required = {"tenant_records", "tenants", "users", "audit_events", "usage_events"}
    missing = sorted(required - tables)
    if missing:
        return ("migrations", "FAIL", "missing tables: " + ", ".join(missing))
    head = next(iter(heads)) if len(heads) == 1 else ",".join(sorted(heads))
    if settings.environment == "production" and revision not in heads:
        return (
            "migrations",
            "FAIL",
            f"database revision {revision!r} is not head {head!r}; "
            "run `alembic upgrade head` before starting workers",
        )
    return (
        "migrations",
        "PASS",
        f"required tables are present; alembic revision {revision} (head {head})",
    )


def _check_inference(settings: Settings) -> tuple[str, CheckStatus, str]:
    try:
        from llm_fabric.router.registry import ModelRegistry

        registry = ModelRegistry.from_yaml(settings.registry_path)
    except Exception as exc:  # noqa: BLE001
        return ("inference", "FAIL", f"registry unreadable: {exc}")
    enabled = registry.enabled_models()
    live = [spec for spec in enabled if spec.provider != "mock"]
    if not enabled:
        return ("inference", "FAIL", "no models are enabled in the registry")
    if not live:
        return (
            "inference",
            "WARN",
            "only mock providers are enabled; no live inference backend is configured",
        )
    if settings.openai_base_url and "11434" in settings.openai_base_url:
        try:
            import httpx

            response = httpx.get(settings.openai_base_url.rstrip("/") + "/models", timeout=2.0)
            if response.status_code >= 400:
                return (
                    "inference",
                    "FAIL",
                    f"OpenAI-compatible endpoint returned HTTP {response.status_code}",
                )
        except Exception as exc:  # noqa: BLE001
            return ("inference", "FAIL", f"OpenAI-compatible endpoint unreachable: {exc}")
    return (
        "inference",
        "PASS",
        "live providers enabled: " + ", ".join(sorted({spec.provider for spec in live})),
    )


def _check_guardrails(settings: Settings) -> tuple[str, CheckStatus, str]:
    del settings
    from llm_fabric.guardrails import default_engines

    engines = default_engines()
    stages = [engine.stage.value for engine in engines]
    if len(engines) < 5:
        return ("guardrails", "FAIL", "expected five guardrail stages, found " + ", ".join(stages))
    return ("guardrails", "PASS", "five-stage pipeline configured: " + ", ".join(stages))


def _check_telemetry(settings: Settings) -> tuple[str, CheckStatus, str]:
    endpoint = settings.otel_exporter_otlp_endpoint
    if not endpoint:
        return (
            "telemetry",
            "WARN",
            "LLM_FABRIC_OTEL_EXPORTER_OTLP_ENDPOINT is unset; traces stay in-process",
        )
    try:
        import httpx

        response = httpx.get(endpoint, timeout=2.0)
        detail = f"OTLP endpoint {endpoint} responded HTTP {response.status_code}"
        if response.status_code >= 500:
            return ("telemetry", "FAIL", detail)
        return ("telemetry", "PASS", detail)
    except Exception as exc:  # noqa: BLE001
        return (
            "telemetry",
            "FAIL",
            f"OTLP endpoint {endpoint} is unreachable: {exc}",
        )


def format_report(checks: Sequence[tuple[str, CheckStatus, str]]) -> str:
    width = max(len(name) for name, _, _ in checks)
    lines = [
        "MyVista LLM Fabric doctor",
        "",
        "CONFIGURATION / STARTUP DEPENDENCY / CURRENT RUNTIME HEALTH",
        "This report is diagnostic. Serving readiness is /readyz.",
        "",
    ]
    for name, status, detail in checks:
        lines.append(f"{status:<4}  {name:<{width}}  {detail}")
    failures = sum(1 for _, status, _ in checks if status == "FAIL")
    warnings = sum(1 for _, status, _ in checks if status == "WARN")
    lines.append("")
    lines.append(f"{failures} FAIL, {warnings} WARN, {len(checks) - failures - warnings} PASS")
    return "\n".join(lines) + "\n"


def run_doctor(settings: Settings | None = None) -> int:
    """Print PASS/WARN/FAIL for each auth prerequisite. Returns process exit code."""
    resolved = settings if settings is not None else get_settings()
    checks = collect_checks(resolved)
    sys.stdout.write(format_report(checks))
    return 1 if any(status == "FAIL" for _, status, _ in checks) else 0


def main() -> None:
    try:
        raise SystemExit(run_doctor())
    except ConfigurationError as exc:
        sys.stderr.write(exc.message + "\n")
        raise SystemExit(1) from exc
