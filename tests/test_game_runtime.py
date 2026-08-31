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
    GameDebrief,
    GameResult,
    PlayerAccusation,
)


class RuntimeFakeGateway:
    """Provide one deterministic route and debrief for runtime testing."""

    def route_free_form(
        self,
        request: str,
        actions: tuple[GameAction, ...],
        completed_action_ids: tuple[str, ...],
        discovered_evidence: tuple[dict[str, object], ...],
    ) -> GameActionRoute:
        del request, actions, completed_action_ids, discovered_evidence
        return GameActionRoute(
            action_id="inspect_security_screening",
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

    def write_debrief(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        accusation: PlayerAccusation,
        result: GameResult,
    ) -> GameDebrief:
        del public_case, discovered_evidence, accusation, result
        return GameDebrief(
            headline="Case closed",
            summary="The physical record and access history align.",
            evidence_analysis=["The cited evidence forms one continuous chain."],
            closing_line="Aurora Circuit is recovered.",
        )


def test_sqlite_runtime_resumes_all_three_player_boundaries(tmp_path: Path) -> None:
    config = AppConfig(anthropic_api_key="test-key")
    checkpoint_path = tmp_path / "game.sqlite"

    with create_player_game_runtime(
        config,
        checkpoint_path=checkpoint_path,
        gateway=RuntimeFakeGateway(),
    ) as runtime:
        result = runtime.start("midnight_museum", "runtime-flow")
        assert get_game_interrupt(result)["phase"] == "visual_choice"

        result = runtime.resume(
            "runtime-flow",
            {"action_id": "inspect_display_case"},
        )
        assert get_game_interrupt(result)["phase"] == "free_form"

        result = runtime.resume(
            "runtime-flow",
            {"request": "Compare the equipment weights."},
        )
        assert get_game_interrupt(result)["phase"] == "accusation"

        result = runtime.resume(
            "runtime-flow",
            {
                "suspect_id": "rowan_pike",
                "motive": "He wanted to possess the sculpture.",
                "method": "He hid it in his equipment case before the blackout.",
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
