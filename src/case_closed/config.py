"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is unavailable."""


class AppConfig(BaseModel):
    """Validated, immutable configuration for the Anthropic gateway."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    anthropic_api_key: SecretStr = Field(min_length=1, repr=False)
    anthropic_workspace_id: SecretStr | None = Field(default=None, repr=False)
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, min_length=1)
    max_tokens: int = Field(default=2_048, ge=1, le=64_000)
    max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator(
        "anthropic_api_key",
        "anthropic_workspace_id",
        "anthropic_model",
        mode="before",
    )
    @classmethod
    def strip_nonempty_text(cls, value: object) -> object:
        """Normalize environment-sourced strings before validation."""
        if isinstance(value, SecretStr):
            return value.get_secret_value().strip()
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        dotenv_path: str | Path | None = None,
    ) -> Self:
        """Build configuration from an injected mapping or the process environment."""
        if environ is None:
            load_dotenv(dotenv_path=dotenv_path)
            environ = os.environ

        api_key = environ.get("ANTHROPIC_API_KEY")
        if api_key is None or not api_key.strip():
            raise ConfigurationError("ANTHROPIC_API_KEY is required")

        return cls(
            anthropic_api_key=api_key,
            anthropic_workspace_id=environ.get("ANTHROPIC_WORKSPACE_ID"),
            anthropic_model=environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        )


def load_config(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
) -> AppConfig:
    """Load validated application configuration."""
    return AppConfig.from_environment(environ, dotenv_path=dotenv_path)
