"""Tests for SQLite-backed player-game runtime helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from case_closed.config import AppConfig
from case_closed.game_runtime import create_player_game_runtime, get_game_interrupt
from case_closed.game_schemas import (
    AccusationMatch,
    GameAction,
    GameActionRoute,
    PlayerAccusation,
)


class RuntimeFakeGateway:
    """Provide one deterministic route and theory match for runtime testing."""

    def route_free_form(
        self,
        request: str,
        actions: tuple[GameAction, ...],
        completed_action_ids: tuple[str, ...],
        discovered_evidence: tuple[dict[str, object], ...],
    ) -> GameActionRoute:
        del request, actions, completed_action_ids, discovered_evidence
        return GameActionRoute(
            action_id="crosscheck_rowans_case",
            needs_clarification=False,
            clarification_question=None,
            suggested_action_ids=[],
            player_message="The records are ready.",
        )

    def assess_accusation(
        self,
        accusation: PlayerAccusation,
        canonical_story: tuple[str, ...],
    ) -> AccusationMatch:
        del accusation, canonical_story
        return AccusationMatch(motive_match=False, method_match=True)


def test_sqlite_runtime_resumes_optional_dive_and_accusation_boundaries(tmp_path: Path) -> None:
    config = AppConfig(anthropic_api_key="test-key")
    checkpoint_path = tmp_path / "game.sqlite"

    with create_player_game_runtime(
        config,
        checkpoint_path=checkpoint_path,
        gateway=RuntimeFakeGateway(),
    ) as runtime:
        result = runtime.start("midnight_museum", "runtime-flow")
        assert get_game_interrupt(result)["phase"] == "free_form"

        result = runtime.resume(
            "runtime-flow",
            {"request": "Reconcile Rowan's equipment case with the manifest."},
        )
        assert get_game_interrupt(result)["phase"] == "free_form"
        assert result["investigation_count"] == 1

        result = runtime.resume(
            "runtime-flow",
            {"next_step": "accuse"},
        )
        assert get_game_interrupt(result)["phase"] == "accusation"

        result = runtime.resume(
            "runtime-flow",
            {
                "suspect_id": "rowan_pike",
                "motive": "He wanted to preserve Iris Venn's original timing.",
                "method": "He hid the sculpture in his equipment case before the blackout.",
            },
        )
        assert get_game_interrupt(result) is None
        assert result["status"] == "solved"

    assert checkpoint_path.exists()


def test_runtime_rejects_empty_thread_and_resume_payload(tmp_path: Path) -> None:
    config = AppConfig(anthropic_api_key="test-key")
    with create_player_game_runtime(
        config,
        checkpoint_path=tmp_path / "invalid.sqlite",
        gateway=RuntimeFakeGateway(),
    ) as runtime:
        with pytest.raises(ValueError, match="thread_id"):
            runtime.start("midnight_museum", " ")
        with pytest.raises(ValueError, match="must not be empty"):
            runtime.resume("runtime-flow", " ")
