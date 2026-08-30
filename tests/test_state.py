"""Tests for JSON-safe case state and LangGraph reducers."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from case_closed.schemas import PublicCase
from case_closed.state import (
    MAX_ALLOWED_ROUNDS,
    append_trace,
    create_initial_state,
    merge_evidence,
    validate_case_state,
)


@pytest.fixture
def public_case() -> PublicCase:
    return PublicCase.model_construct(
        schema_version=1,
        case_id="museum-heist",
        title="The Museum Heist",
    )


def test_create_initial_state_is_json_serializable(public_case) -> None:
    state = create_initial_state(public_case)

    serialized = json.dumps(state, allow_nan=False)

    assert json.loads(serialized) == state
    assert state["case_id"] == public_case.case_id
    assert state["status"] == "investigating"
    assert "solution" not in serialized.lower()


def test_merge_evidence_preserves_order_and_is_idempotent() -> None:
    first = {"evidence_id": "ev-1", "text": "A public clue"}
    second = {"evidence_id": "ev-2", "text": "Another public clue"}

    merged = merge_evidence([first], [first, second])

    assert merged == [first, second]
    assert merged[0] is not first


def test_merge_evidence_rejects_changed_record_without_echoing_content() -> None:
    original = {"evidence_id": "ev-1", "text": "public version"}
    changed = {"evidence_id": "ev-1", "text": "sensitive replacement"}

    with pytest.raises(ValueError) as error:
        merge_evidence([original], [changed])

    assert "public version" not in str(error.value)
    assert "sensitive replacement" not in str(error.value)


def test_append_trace_preserves_execution_order() -> None:
    current = [{"node": "plan", "event": "started"}]
    updates = [{"node": "tool", "event": "completed", "round_count": 1}]

    assert append_trace(current, updates) == [*current, *updates]


@pytest.mark.parametrize(
    ("field", "limit_field"),
    [
        ("round_count", "max_rounds"),
        ("invalid_action_count", "max_invalid_actions"),
        ("no_progress_count", "max_rounds"),
        ("verdict_revision_count", "max_verdict_revisions"),
    ],
)
def test_validate_case_state_rejects_counters_above_configured_limits(
    public_case,
    field: str,
    limit_field: str,
) -> None:
    state: dict[str, object] = dict(create_initial_state(public_case))
    configured_limit = state[limit_field]
    assert isinstance(configured_limit, int)
    state[field] = configured_limit + 1

    with pytest.raises(ValueError):
        validate_case_state(state)


def test_create_initial_state_rejects_excessive_limit(public_case) -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        create_initial_state(public_case, max_rounds=MAX_ALLOWED_ROUNDS + 1)


def test_validate_case_state_rejects_unknown_status(public_case) -> None:
    state = create_initial_state(public_case)
    state["status"] = "secretly_solved"  # type: ignore[typeddict-item]

    with pytest.raises(ValueError, match="status"):
        validate_case_state(state)


def test_validate_case_state_rejects_nested_private_solution(public_case) -> None:
    state = deepcopy(create_initial_state(public_case))
    state["assessment"] = {
        "canonical_sequence": ["secret-sequence-item"],
        "culprit_id": "secret-suspect",
    }

    with pytest.raises(ValueError) as error:
        validate_case_state(state)

    assert "secret-suspect" not in str(error.value)
    assert "secret-sequence-item" not in str(error.value)


def test_validate_case_state_rejects_non_json_value(public_case) -> None:
    state = create_initial_state(public_case)
    state["assessment"] = {"bad": object()}  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="JSON"):
        validate_case_state(state)
