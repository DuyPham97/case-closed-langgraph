"""Tests for player-game boundary schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from case_closed.game_schemas import GameAction, GameActionRoute, PlayerAccusation


def test_visual_game_action_must_be_a_location() -> None:
    with pytest.raises(ValidationError, match="visual actions must be location"):
        GameAction(
            action_id="ask_someone",
            title="Ask someone",
            description="Ask one question.",
            category="interview",
            tool_name="interview_suspect",
            target_id="suspect",
            topic="whereabouts",
            is_visual=True,
        )


def test_successful_route_requires_an_action_id() -> None:
    with pytest.raises(ValidationError, match="successful routes need an action_id"):
        GameActionRoute(
            needs_clarification=False,
            player_message="I know where to look.",
        )


def test_clarification_route_requires_a_question() -> None:
    with pytest.raises(ValidationError, match="clarification routes need a question"):
        GameActionRoute(
            needs_clarification=True,
            player_message="Narrow the lead.",
        )


@pytest.mark.parametrize("missing_field", ["motive", "method"])
def test_accusation_requires_motive_and_method(missing_field: str) -> None:
    payload = {
        "suspect_id": "rowan_pike",
        "motive": "He wanted the sculpture.",
        "method": "He concealed it inside his equipment case.",
    }
    del payload[missing_field]

    with pytest.raises(ValidationError):
        PlayerAccusation.model_validate(payload)
