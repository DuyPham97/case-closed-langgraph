"""Curated player actions for the Midnight at the Lumen Museum mystery."""

from __future__ import annotations

from typing import Final

from case_closed.game_schemas import GameAction

_GAME_ACTIONS: Final[tuple[GameAction, ...]] = (
    GameAction(
        action_id="inspect_display_case",
        title="Search the display case",
        description=(
            "Examine the plinth, glass, locks, and service panel where Aurora Circuit vanished."
        ),
        category="location",
        tool_name="inspect_location",
        target_id="display_case",
        is_visual=True,
        image_path="assets/game/crime-scene.webp",
    ),
    GameAction(
        action_id="inspect_security_screening",
        title="Audit security screening",
        description=(
            "Compare room access with the equipment checks recorded before and after the gala."
        ),
        category="location",
        tool_name="inspect_location",
        target_id="security_screening",
        is_visual=True,
        image_path="assets/game/security-screening.webp",
    ),
    GameAction(
        action_id="inspect_media_locker",
        title="Open the media locker",
        description=(
            "Examine the checked camera equipment and anything recorded during the critical minute."
        ),
        category="location",
        tool_name="inspect_location",
        target_id="media_locker",
        is_visual=True,
        image_path="assets/game/media-locker.webp",
    ),
    GameAction(
        action_id="inspect_lighting_booth",
        title="Enter the lighting booth",
        description=(
            "Reconstruct how the blackout was triggered and whether someone touched the controls."
        ),
        category="location",
        tool_name="inspect_location",
        target_id="lighting_booth",
        is_visual=True,
        image_path="assets/game/lighting-booth.webp",
    ),
    GameAction(
        action_id="ask_mara_whereabouts",
        title="Question Mara about her whereabouts",
        description="Ask the curator to account for the minutes before the blackout.",
        category="interview",
        tool_name="interview_suspect",
        target_id="mara_vale",
        topic="whereabouts",
        image_path="assets/game/mara-vale.webp",
    ),
    GameAction(
        action_id="ask_mara_case_access",
        title="Ask Mara who could open the display",
        description=(
            "Press the curator on calibration mode, access, and the unlocked service panel."
        ),
        category="interview",
        tool_name="interview_suspect",
        target_id="mara_vale",
        topic="case_access",
        image_path="assets/game/mara-vale.webp",
    ),
    GameAction(
        action_id="ask_theo_whereabouts",
        title="Question Theo about his whereabouts",
        description="Ask the lighting technician where he was during the suspected removal window.",
        category="interview",
        tool_name="interview_suspect",
        target_id="theo_quinn",
        topic="whereabouts",
        image_path="assets/game/theo-quinn.webp",
    ),
    GameAction(
        action_id="ask_theo_blackout",
        title="Ask Theo about the blackout",
        description="Challenge him to explain the ninety-second cue and who could trigger it.",
        category="interview",
        tool_name="interview_suspect",
        target_id="theo_quinn",
        topic="blackout",
        image_path="assets/game/theo-quinn.webp",
    ),
    GameAction(
        action_id="ask_nia_whereabouts",
        title="Question Nia about her whereabouts",
        description="Ask the security lead to account for herself before the alarm was raised.",
        category="interview",
        tool_name="interview_suspect",
        target_id="nia_brooks",
        topic="whereabouts",
        image_path="assets/game/nia-brooks.webp",
    ),
    GameAction(
        action_id="ask_nia_security",
        title="Ask Nia about screening procedures",
        description="Find out how staff equipment was checked, weighed, and stored that night.",
        category="interview",
        tool_name="interview_suspect",
        target_id="nia_brooks",
        topic="security",
        image_path="assets/game/nia-brooks.webp",
    ),
    GameAction(
        action_id="ask_rowan_whereabouts",
        title="Question Rowan about the photo shoot",
        description="Make the photographer describe exactly where he stood and what he touched.",
        category="interview",
        tool_name="interview_suspect",
        target_id="rowan_pike",
        topic="whereabouts",
        image_path="assets/game/rowan-pike.webp",
    ),
    GameAction(
        action_id="ask_rowan_equipment",
        title="Ask Rowan about his equipment case",
        description="Press him on what he carried into the museum and what he checked afterward.",
        category="interview",
        tool_name="interview_suspect",
        target_id="rowan_pike",
        topic="equipment",
        image_path="assets/game/rowan-pike.webp",
    ),
    GameAction(
        action_id="reconstruct_removal_window",
        title="Reconstruct the 9:42 p.m. window",
        description="Align room access, witness whereabouts, and the moment the plinth changed.",
        category="records",
        tool_name="compare_timeline",
        target_id="weight_drop",
        image_path="assets/game/timeline.webp",
    ),
    GameAction(
        action_id="reconstruct_blackout",
        title="Reconstruct the blackout",
        description=(
            "Compare the ninety-second darkness with events that happened earlier that night."
        ),
        category="records",
        tool_name="compare_timeline",
        target_id="blackout",
        image_path="assets/game/timeline.webp",
    ),
)

_ACTIONS_BY_ID: Final[dict[str, GameAction]] = {
    action.action_id: action for action in _GAME_ACTIONS
}


def all_game_actions() -> tuple[GameAction, ...]:
    """Return every action that Claude may map a player's request onto."""
    return _GAME_ACTIONS


def visual_game_actions() -> tuple[GameAction, ...]:
    """Return the four locations offered for the player's first investigation."""
    return tuple(action for action in _GAME_ACTIONS if action.is_visual)


def get_game_action(action_id: str) -> GameAction:
    """Resolve a validated action ID or reject an action outside the case catalog."""
    try:
        return _ACTIONS_BY_ID[action_id]
    except KeyError as exc:
        raise KeyError(f"unknown game action: {action_id!r}") from exc
