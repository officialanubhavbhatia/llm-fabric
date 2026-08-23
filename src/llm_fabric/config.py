"""Typed configuration, loaded from the environment.

Every setting is prefixed `LLM_FABRIC_` except provider credentials, which use
the names the providers themselves document so existing environments work
unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_FABRIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 47317

    registry_path: Path = Path("config/models.yaml")

    # Comma-separated list of accepted client keys. Empty means the gateway is
    # open, which is intended for local development only.
    api_keys: list[str] = Field(default_factory=list)

    # Per-attempt ceiling, not a total request budget.
    request_timeout_s: float = 60.0

    # Total attempts across the whole fallback chain, including the first.
    max_attempts: int = 3

    default_policy: str = "cheapest"
    log_level: str = "INFO"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return [key.strip() for key in value.split(",") if key.strip()]
        return value

    @property
    def auth_required(self) -> bool:
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
