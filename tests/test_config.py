"""Tests for application configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from case_closed.config import (
    DEFAULT_ANTHROPIC_MODEL,
    AppConfig,
    ConfigurationError,
    load_config,
)

DUMMY_API_KEY = "anthropic-test-api-key"


def test_load_config_uses_pinned_default_and_redacts_api_key() -> None:
    config = load_config({"ANTHROPIC_API_KEY": f"  {DUMMY_API_KEY}  "})

    assert config.anthropic_model == DEFAULT_ANTHROPIC_MODEL
    assert config.anthropic_api_key.get_secret_value() == DUMMY_API_KEY
    assert DUMMY_API_KEY not in repr(config)
    assert DUMMY_API_KEY not in config.model_dump_json()


def test_load_config_normalizes_model_override() -> None:
    config = load_config(
        {
            "ANTHROPIC_API_KEY": DUMMY_API_KEY,
            "ANTHROPIC_MODEL": "  claude-haiku-4-5  ",
        }
    )

    assert config.anthropic_model == "claude-haiku-4-5"


def test_load_config_redacts_optional_workspace_id() -> None:
    workspace_id = "wrkspc_test_identifier"
    config = load_config(
        {
            "ANTHROPIC_API_KEY": DUMMY_API_KEY,
            "ANTHROPIC_WORKSPACE_ID": f"  {workspace_id}  ",
        }
    )

    assert config.anthropic_workspace_id is not None
    assert config.anthropic_workspace_id.get_secret_value() == workspace_id
    assert workspace_id not in repr(config)
    assert workspace_id not in config.model_dump_json()


def test_load_config_treats_blank_workspace_id_as_unset() -> None:
    config = load_config(
        {
            "ANTHROPIC_API_KEY": DUMMY_API_KEY,
            "ANTHROPIC_WORKSPACE_ID": "   ",
        }
    )

    assert config.anthropic_workspace_id is None


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_load_config_requires_nonempty_api_key(api_key: str | None) -> None:
    environ = {} if api_key is None else {"ANTHROPIC_API_KEY": api_key}

    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY is required"):
        load_config(environ)


def test_app_config_is_frozen_and_forbids_extra_fields() -> None:
    config = AppConfig(anthropic_api_key=DUMMY_API_KEY)

    with pytest.raises(ValidationError):
        config.anthropic_model = "different-model"

    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "anthropic_api_key": DUMMY_API_KEY,
                "unexpected": "value",
            }
        )
