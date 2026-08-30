"""Tests for public player-game state and reducers."""

from __future__ import annotations

import pytest

from case_closed.case_store import CaseStore
from case_closed.game_state import (
    create_player_game_state,
    merge_action_ids,
    validate_player_game_state,
)


def test_initial_player_state_contains_no_private_solution() -> None:
    state = create_player_game_state(CaseStore().load_public_case())

    assert state["investigation_count"] == 0
    assert state["stage"] == "visual_choice"
    assert "solution" not in state
    validate_player_game_state(state)


def test_action_reducer_is_ordered_and_idempotent() -> None:
    assert merge_action_ids(["first"], ["first", "second"]) == ["first", "second"]


def test_state_rejects_private_solution_data() -> None:
    state = create_player_game_state(CaseStore().load_public_case())
    state["result"] = {"solution_key": {"culprit_id": "someone"}}

    with pytest.raises(ValueError, match="private solution"):
        validate_player_game_state(state)
