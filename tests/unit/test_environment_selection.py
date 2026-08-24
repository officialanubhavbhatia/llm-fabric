"""LLM_FABRIC_ENVIRONMENT has no implicit default.

These tests drive the real process entry points, not only Settings().
A missing or unknown value must exit before the process binds a port.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def _clean_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "LLM_FABRIC_ENVIRONMENT" and not key.startswith("LLM_FABRIC_")
    }
    env["PYTHONPATH"] = str(SRC)
    env["HOME"] = str(tmp_path)
    env.update(overrides)
    return env


def _run_module(tmp_path: Path, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_fabric", *args],
        cwd=tmp_path,
        env=_clean_env(tmp_path, **overrides),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_cli_exits_when_environment_is_unset(tmp_path: Path) -> None:
    result = _run_module(tmp_path)
    assert result.returncode == 1
    assert "LLM_FABRIC_ENVIRONMENT is required" in result.stderr
    assert result.returncode != 0


def test_cli_exits_when_environment_is_blank(tmp_path: Path) -> None:
    result = _run_module(tmp_path, LLM_FABRIC_ENVIRONMENT="")
    assert result.returncode == 1
    assert "LLM_FABRIC_ENVIRONMENT is required" in result.stderr


def test_cli_exits_when_environment_is_invalid(tmp_path: Path) -> None:
    result = _run_module(tmp_path, LLM_FABRIC_ENVIRONMENT="staging")
    assert result.returncode == 1
    assert "not valid" in result.stderr


def test_doctor_exits_when_environment_is_unset(tmp_path: Path) -> None:
    result = _run_module(tmp_path, "doctor")
    assert result.returncode == 1
    assert "LLM_FABRIC_ENVIRONMENT is required" in result.stderr


def test_factory_create_app_exits_when_environment_is_unset(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from llm_fabric.gateway.app import create_app; create_app()",
        ],
        cwd=tmp_path,
        env=_clean_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "LLM_FABRIC_ENVIRONMENT is required" in result.stderr


def test_explicit_development_is_accepted(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from llm_fabric.config import Settings, validate_startup; "
            "s = Settings(_env_file=None, environment='development', api_keys=[]); "
            "validate_startup(s); "
            "assert s.environment == 'development'; "
            "assert s.auth_required is False",
        ],
        cwd=tmp_path,
        env=_clean_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_test_is_not_production_fail_closed(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from llm_fabric.config import Settings, validate_startup; "
            "s = Settings(_env_file=None, environment='test', api_keys=[], allow_anonymous=True); "
            "validate_startup(s); "
            "assert s.environment == 'test'; "
            "assert s.auth_required is False",
        ],
        cwd=tmp_path,
        env=_clean_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_production_refuses_anonymous_bypass(tmp_path: Path) -> None:
    script = tmp_path / "check.py"
    script.write_text(
        "from llm_fabric.config import Settings, validate_startup\n"
        "from llm_fabric.errors import ConfigurationError\n"
        "settings = Settings(\n"
        "    _env_file=None,\n"
        "    environment='production',\n"
        "    allow_anonymous=True,\n"
        "    api_keys=[],\n"
        ")\n"
        "try:\n"
        "    validate_startup(settings)\n"
        "except ConfigurationError as exc:\n"
        "    assert 'ALLOW_ANONYMOUS' in exc.message\n"
        "else:\n"
        "    raise SystemExit('production accepted anonymous bypass')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=_clean_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
