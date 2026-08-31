"""Public, checkpoint-safe state for the player-driven mystery."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypedDict, cast

from case_closed.schemas import PublicCase
from case_closed.state import (
    EvidenceState,
    JsonObject,
    TraceEvent,
    append_trace,
    merge_evidence,
)

GameStage = Literal["free_form", "accusation", "complete"]
GameStatus = Literal["playing", "solved", "closed"]

_PRIVATE_KEYS = frozenset(
    {
        "solution",
        "solution_key",
        "private_solution",
        "answer_key",
        "canonical_sequence",
        "acceptable_evidence_sets",
        "correct_culprit_id",
    }
)
_PLAYER_CASE_KEYS = (
    "schema_version",
    "case_id",
    "title",
    "display_timezone",
    "brief",
    "artwork",
    "incident",
    "suspects",
    "locations",
)


def merge_action_ids(
    current: Sequence[str] | None,
    updates: Sequence[str] | None,
) -> list[str]:
    """Append action IDs once so replayed graph nodes remain idempotent."""
    merged: list[str] = []
    for action_id in [*(current or ()), *(updates or ())]:
        normalized = action_id.strip()
        if not normalized:
            raise ValueError("completed action IDs must be non-empty strings")
        if normalized not in merged:
            merged.append(normalized)
    return merged


class PlayerGameState(TypedDict):
    """Every public value persisted in the player game's LangGraph checkpoint."""

    case_id: str
    public_case: JsonObject
    stage: GameStage
    status: GameStatus
    investigation_count: int
    completed_action_ids: Annotated[list[str], merge_action_ids]
    discovered_evidence: Annotated[list[EvidenceState], merge_evidence]
    trace: Annotated[list[TraceEvent], append_trace]
    pending_action_id: str | None
    free_form_request: str | None
    routed_action: JsonObject | None
    last_observation: JsonObject | None
    accusation: JsonObject | None
    result: JsonObject | None
    debrief: JsonObject | None


def create_player_game_state(public_case: PublicCase) -> PlayerGameState:
    """Create public state with the base dossier and two optional deep dives."""
    case_payload = public_case.model_dump(mode="json")
    evidence_by_id = {record.evidence_id: record for record in public_case.observations}
    case_file_evidence = [
        {
            "evidence_id": record.evidence_id,
            "title": record.title,
            "text": record.text,
            "occurred_at": record.occurred_at.isoformat(),
        }
        for evidence_id in public_case.case_file_evidence_ids
        for record in (evidence_by_id[evidence_id],)
    ]
    state = PlayerGameState(
        case_id=public_case.case_id,
        public_case=cast(
            JsonObject,
            {key: case_payload[key] for key in _PLAYER_CASE_KEYS},
        ),
        stage="free_form",
        status="playing",
        investigation_count=0,
        completed_action_ids=[],
        discovered_evidence=case_file_evidence,
        trace=[],
        pending_action_id=None,
        free_form_request=None,
        routed_action=None,
        last_observation=None,
        accusation=None,
        result=None,
        debrief=None,
    )
    validate_player_game_state(state)
    return state


def validate_player_game_state(state: Mapping[str, object]) -> None:
    """Reject private, non-JSON, or impossible player game state."""
    case_id = state.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")
    if state.get("stage") not in {"free_form", "accusation", "complete"}:
        raise ValueError("stage is not supported")
    if state.get("status") not in {"playing", "solved", "closed"}:
        raise ValueError("status is not supported")
    count = state.get("investigation_count")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 2:
        raise ValueError("investigation_count must be between zero and two")
    completed = state.get("completed_action_ids")
    if not isinstance(completed, list) or len(completed) != count:
        raise ValueError("completed actions must match the investigation count")
    _reject_private_data(state)
    try:
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("player game state must be JSON serializable") from exc


def _reject_private_data(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _PRIVATE_KEYS:
                raise ValueError("private solution data cannot enter player game state")
            _reject_private_data(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_private_data(child)
