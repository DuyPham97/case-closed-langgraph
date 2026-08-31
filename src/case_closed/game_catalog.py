"""Curated deep-dive actions for Midnight at the Lumen Museum."""

from __future__ import annotations

from typing import Final

from case_closed.game_schemas import GameAction

_GAME_ACTIONS: Final[tuple[GameAction, ...]] = (
    GameAction(
        action_id="challenge_mara_on_migration",
        title="Challenge Mara with commissioning note 14-B",
        description=(
            "Press the curator on whether the controller migration can truly be reversed."
        ),
        category="interview",
        tool_name="interview_suspect",
        target_id="mara_vale",
        topic="migration",
        image_path="assets/game/mara-vale.webp",
    ),
    GameAction(
        action_id="ask_theo_who_heard_warning",
        title="Ask Theo who heard the controller warning",
        description=(
            "Find out who knew the original timing would be impossible to restore after migration."
        ),
        category="interview",
        tool_name="interview_suspect",
        target_id="theo_quinn",
        topic="controller_warning",
        image_path="assets/game/theo-quinn.webp",
    ),
    GameAction(
        action_id="ask_mara_about_rowan_and_iris",
        title="Ask Mara about Rowan's history with Iris Venn",
        description=(
            "Establish what Rowan did for the artist and her estate before this exhibition."
        ),
        category="interview",
        tool_name="interview_suspect",
        target_id="mara_vale",
        topic="rowan_history",
        image_path="assets/game/mara-vale.webp",
    ),
    GameAction(
        action_id="audit_rowans_locker_request",
        title="Audit Rowan's locker request",
        description=("Cross-check what Rowan asked Nia before he checked in his equipment case."),
        category="interview",
        tool_name="interview_suspect",
        target_id="nia_brooks",
        topic="locker_request",
        image_path="assets/game/security-screening.webp",
    ),
    GameAction(
        action_id="confront_rowan_with_frame",
        title="Confront Rowan with frame RP_2207",
        description=(
            "Make Rowan reconcile his first statement with the reflection beside the open panel."
        ),
        category="interview",
        tool_name="interview_suspect",
        target_id="rowan_pike",
        topic="camera_frame",
        image_path="assets/game/rowan-pike.webp",
    ),
    GameAction(
        action_id="crosscheck_rowans_case",
        title="Reconcile Rowan's case weight with his manifest",
        description=(
            "Test Rowan's explanation against both weighings and the equipment chain of custody."
        ),
        category="records",
        tool_name="compare_timeline",
        target_id="rowan_case_chain",
        image_path="assets/game/timeline.webp",
    ),
)

_ACTIONS_BY_ID: Final[dict[str, GameAction]] = {
    action.action_id: action for action in _GAME_ACTIONS
}


def all_game_actions() -> tuple[GameAction, ...]:
    """Return every optional deep dive that Claude may map a request onto."""
    return _GAME_ACTIONS


def get_game_action(action_id: str) -> GameAction:
    """Resolve a validated action ID or reject an action outside the case catalog."""
    try:
        return _ACTIONS_BY_ID[action_id]
    except KeyError as exc:
        raise KeyError(f"unknown game action: {action_id!r}") from exc
