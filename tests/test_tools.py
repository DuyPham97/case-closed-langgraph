"""Tests for deterministic public case observations and LangChain tool wrappers."""

from __future__ import annotations

import pytest

from case_closed.case_store import (
    DEFAULT_CASE_ID,
    CaseStore,
    UnknownToolTargetError,
)
from case_closed.schemas import ToolObservation
from case_closed.tools import create_case_tools


@pytest.mark.parametrize(
    ("method_name", "arguments", "expected_evidence_ids"),
    [
        ("inspect_location", ("display_case",), ["E01", "E02", "E03"]),
        ("inspect_location", ("media_locker",), ["E06"]),
        ("inspect_location", ("security_screening",), ["E07"]),
        ("inspect_location", ("lighting_booth",), ["E08"]),
        ("interview_suspect", ("rowan_pike", "whereabouts"), ["E05"]),
        ("compare_timeline", ("weight_drop",), ["E04", "E09"]),
        ("compare_timeline", ("blackout",), ["E01", "E08"]),
    ],
)
def test_public_routes_return_expected_evidence(
    method_name: str,
    arguments: tuple[str, ...],
    expected_evidence_ids: list[str],
) -> None:
    store = CaseStore()
    method = getattr(store, method_name)

    result = method(DEFAULT_CASE_ID, *arguments)

    assert result.evidence_ids == expected_evidence_ids
    assert result.no_new_evidence is False
    assert all(f"[{evidence_id}]" in result.summary for evidence_id in expected_evidence_ids)


def test_scripted_interview_without_evidence_reports_no_new_evidence() -> None:
    result = CaseStore().interview_suspect(
        DEFAULT_CASE_ID,
        "mara_vale",
        "whereabouts",
    )

    assert result.evidence_ids == []
    assert result.no_new_evidence is True
    assert "greenroom" in result.summary


def test_repeated_tool_calls_are_byte_for_byte_deterministic() -> None:
    store = CaseStore()

    first = store.compare_timeline(DEFAULT_CASE_ID, "weight_drop")
    second = store.compare_timeline(DEFAULT_CASE_ID, "weight_drop")

    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    [
        ("inspect_location", ("roof",)),
        ("interview_suspect", ("rowan_pike", "motive")),
        ("compare_timeline", ("midnight",)),
    ],
)
def test_unknown_tool_target_is_rejected(
    method_name: str,
    arguments: tuple[str, ...],
) -> None:
    store = CaseStore()
    method = getattr(store, method_name)

    with pytest.raises(UnknownToolTargetError, match=r"no .* route"):
        method(DEFAULT_CASE_ID, *arguments)


def test_langchain_tools_have_stable_names_and_schemas() -> None:
    inspect_tool, interview_tool, timeline_tool = create_case_tools()

    assert [inspect_tool.name, interview_tool.name, timeline_tool.name] == [
        "inspect_location",
        "interview_suspect",
        "compare_timeline",
    ]
    assert set(inspect_tool.args) == {"location_id"}
    assert set(interview_tool.args) == {"suspect_id", "topic"}
    assert set(timeline_tool.args) == {"anchor_id"}


def test_langchain_tool_returns_structured_public_observation() -> None:
    inspect_tool, _, _ = create_case_tools()

    result = inspect_tool.invoke({"location_id": "display_case"})

    assert isinstance(result, ToolObservation)
    assert result.evidence_ids == ["E01", "E02", "E03"]


def test_tool_outputs_do_not_contain_private_solution_fields() -> None:
    store = CaseStore()
    private_sequence = store.load_solution().canonical_sequence
    outputs = [
        store.inspect_location(DEFAULT_CASE_ID, location_id).model_dump_json()
        for location_id in store.load_public_case().available_actions.location_ids
    ]
    serialized = "\n".join(outputs)

    assert "culprit_id" not in serialized
    assert "acceptable_evidence_sets" not in serialized
    assert all(private_step not in serialized for private_step in private_sequence)
