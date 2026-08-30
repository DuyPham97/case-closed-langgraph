"""JSON-safe LangGraph state and reducers for a detective case."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Required, TypedDict, cast

from case_closed.schemas import CaseStatus, PublicCase

DEFAULT_MAX_ROUNDS = 6
DEFAULT_MAX_INVALID_ACTIONS = 3
DEFAULT_MAX_VERDICT_REVISIONS = 1
MAX_ALLOWED_ROUNDS = 12
MAX_ALLOWED_INVALID_ACTIONS = 5
MAX_ALLOWED_VERDICT_REVISIONS = 3

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_PRIVATE_STATE_KEYS = frozenset(
    {
        "answer_key",
        "acceptable_evidence_sets",
        "canonical_sequence",
        "claim_rules",
        "contradiction_pairs",
        "correct_culprit_id",
        "exoneration_rules",
        "hints",
        "private_solution",
        "required_evidence_ids",
        "solution",
        "solution_key",
        "solution_rules",
    }
)


class EvidenceState(TypedDict, total=False):
    """A JSON representation of one discovered public evidence record."""

    evidence_id: Required[str]
    title: str
    text: str
    source_type: str
    source_reference: str
    occurred_at: str


class TraceEvent(TypedDict, total=False):
    """A compact, public event emitted while the graph runs."""

    node: Required[str]
    event: Required[str]
    round_count: int
    message: str
    data: JsonObject


class PublicValidationError(TypedDict):
    """A redacted validation issue that is safe to checkpoint or show to a model."""

    code: str
    message: str


def merge_evidence(
    current: Sequence[EvidenceState] | None,
    updates: Sequence[EvidenceState] | None,
) -> list[EvidenceState]:
    """Merge discovered evidence idempotently while rejecting changed records."""

    merged: list[EvidenceState] = []
    positions: dict[str, int] = {}

    for record in [*(current or ()), *(updates or ())]:
        normalized = _normalize_evidence(record)
        evidence_id = normalized["evidence_id"]
        existing_position = positions.get(evidence_id)
        if existing_position is None:
            positions[evidence_id] = len(merged)
            merged.append(normalized)
            continue
        if merged[existing_position] != normalized:
            raise ValueError("A discovered evidence record cannot change after it is stored.")

    return merged


def append_trace(
    current: Sequence[TraceEvent] | None,
    updates: Sequence[TraceEvent] | None,
) -> list[TraceEvent]:
    """Append validated public trace events in graph execution order."""

    return [
        *(_normalize_trace_event(event) for event in current or ()),
        *(_normalize_trace_event(event) for event in updates or ()),
    ]


class CaseState(TypedDict):
    """Complete public state persisted by the LangGraph checkpointer."""

    case_id: str
    public_case: JsonObject
    discovered_evidence: Annotated[list[EvidenceState], merge_evidence]
    trace: Annotated[list[TraceEvent], append_trace]
    round_count: int
    max_rounds: int
    invalid_action_count: int
    max_invalid_actions: int
    no_progress_count: int
    last_observation: JsonObject | None
    hypotheses: list[JsonObject]
    assessment: JsonObject | None
    human_review_enabled: bool
    human_review_completed: bool
    human_direction: str | None
    pending_action: JsonObject | None
    proposed_verdict: JsonObject | None
    verdict_revision_count: int
    max_verdict_revisions: int
    validation_errors: list[PublicValidationError]
    status: CaseStatus
    final_report: str | None


def create_initial_state(
    public_case: PublicCase,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_invalid_actions: int = DEFAULT_MAX_INVALID_ACTIONS,
    max_verdict_revisions: int = DEFAULT_MAX_VERDICT_REVISIONS,
    human_review_enabled: bool = True,
) -> CaseState:
    """Create a complete, validated checkpoint state for a public case."""

    _validate_limit("max_rounds", max_rounds, MAX_ALLOWED_ROUNDS)
    _validate_limit(
        "max_invalid_actions",
        max_invalid_actions,
        MAX_ALLOWED_INVALID_ACTIONS,
    )
    _validate_limit(
        "max_verdict_revisions",
        max_verdict_revisions,
        MAX_ALLOWED_VERDICT_REVISIONS,
    )
    if not isinstance(human_review_enabled, bool):
        raise TypeError("human_review_enabled must be a boolean.")

    public_payload = _normalize_json_object(public_case.model_dump(mode="json"))
    state = CaseState(
        case_id=public_case.case_id,
        public_case=public_payload,
        discovered_evidence=[],
        trace=[],
        round_count=0,
        max_rounds=max_rounds,
        invalid_action_count=0,
        max_invalid_actions=max_invalid_actions,
        no_progress_count=0,
        last_observation=None,
        hypotheses=[],
        assessment=None,
        human_review_enabled=human_review_enabled,
        human_review_completed=False,
        human_direction=None,
        pending_action=None,
        proposed_verdict=None,
        verdict_revision_count=0,
        max_verdict_revisions=max_verdict_revisions,
        validation_errors=[],
        status="investigating",
        final_report=None,
    )
    validate_case_state(state)
    return state


def validate_case_state(state: Mapping[str, object]) -> None:
    """Reject non-JSON, private, or out-of-bounds checkpoint state."""

    _require_non_empty_string(state, "case_id")
    max_rounds = _require_bounded_limit(state, "max_rounds", MAX_ALLOWED_ROUNDS)
    max_invalid_actions = _require_bounded_limit(
        state,
        "max_invalid_actions",
        MAX_ALLOWED_INVALID_ACTIONS,
    )
    max_verdict_revisions = _require_bounded_limit(
        state,
        "max_verdict_revisions",
        MAX_ALLOWED_VERDICT_REVISIONS,
    )
    _require_counter(state, "round_count", max_rounds)
    _require_counter(state, "invalid_action_count", max_invalid_actions)
    _require_counter(state, "no_progress_count", max_rounds)
    _require_counter(state, "verdict_revision_count", max_verdict_revisions)

    status = state.get("status")
    if status not in {"investigating", "awaiting_human", "resolved", "inconclusive"}:
        raise ValueError("status is not a supported case status.")

    human_review_enabled = state.get("human_review_enabled")
    human_review_completed = state.get("human_review_completed")
    if not isinstance(human_review_enabled, bool) or not isinstance(human_review_completed, bool):
        raise TypeError("human review flags must be booleans.")
    if human_review_completed and not human_review_enabled:
        raise ValueError("human review cannot be completed when it is disabled.")
    if status == "awaiting_human" and (not human_review_enabled or human_review_completed):
        raise ValueError("awaiting_human requires an incomplete, enabled human review.")

    _reject_private_keys(state)
    try:
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("case state must contain only JSON-serializable values.") from exc


def _normalize_evidence(record: Mapping[str, object]) -> EvidenceState:
    evidence_id = record.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("Every discovered evidence record needs a non-empty evidence_id.")
    normalized = _normalize_json_object(record)
    return cast(EvidenceState, normalized)


def _normalize_trace_event(event: Mapping[str, object]) -> TraceEvent:
    for field in ("node", "event"):
        value = event.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Trace field {field} must be a non-empty string.")
    normalized = _normalize_json_object(event)
    return cast(TraceEvent, normalized)


def _normalize_json_object(value: Mapping[str, object]) -> JsonObject:
    try:
        encoded = json.dumps(value, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError("State values must be JSON-serializable.") from exc
    if not isinstance(normalized, dict):
        raise TypeError("State objects must use string-keyed mappings.")
    _reject_private_keys(normalized)
    return cast(JsonObject, normalized)


def _validate_limit(name: str, value: object, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")


def _require_bounded_limit(
    state: Mapping[str, object],
    name: str,
    maximum: int,
) -> int:
    value = state.get(name)
    _validate_limit(name, value, maximum)
    return cast(int, value)


def _require_counter(state: Mapping[str, object], name: str, maximum: int) -> int:
    value = state.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be between 0 and its configured limit.")
    return value


def _require_non_empty_string(state: Mapping[str, object], name: str) -> str:
    value = state.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _reject_private_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _PRIVATE_STATE_KEYS:
                raise ValueError("Private solution data cannot be stored in case state.")
            _reject_private_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_private_keys(child)
