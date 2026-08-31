"""Tests for bundled case loading, validation, and answer-key isolation."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from case_closed.case_store import (
    DEFAULT_CASE_ID,
    CaseNotFoundError,
    CaseStore,
    InvalidCaseDataError,
)
from case_closed.schemas import EvidenceRecord

CASES_DIR = Path(__file__).parents[1] / "src" / "case_closed" / "cases"
PUBLIC_PATH = CASES_DIR / DEFAULT_CASE_ID / "public.json"
SOLUTION_PATH = CASES_DIR / DEFAULT_CASE_ID / "solution.json"


def test_midnight_museum_public_case_is_complete_and_utc() -> None:
    case = CaseStore().load_public_case()

    assert case.case_id == DEFAULT_CASE_ID
    assert [suspect.suspect_id for suspect in case.suspects] == [
        "mara_vale",
        "theo_quinn",
        "nia_brooks",
        "rowan_pike",
    ]
    assert [record.evidence_id for record in case.observations] == [
        f"E{index:02d}" for index in range(1, 19)
    ]
    assert case.case_file_evidence_ids == [f"E{index:02d}" for index in range(1, 13)]
    timestamps = [
        case.incident.discovered_at,
        case.incident.blackout_started_at,
        case.incident.blackout_ended_at,
        *(record.occurred_at for record in case.observations),
        *(event.starts_at for event in case.timeline_events),
        *(event.ends_at for event in case.timeline_events if event.ends_at is not None),
    ]
    assert all(timestamp.tzinfo is UTC for timestamp in timestamps)


def test_every_evidence_record_is_open_or_reachable() -> None:
    case = CaseStore().load_public_case()

    evidence_ids = {record.evidence_id for record in case.observations}
    reachable_ids = {
        evidence_id for route in case.reveal_routes for evidence_id in route.evidence_ids
    }

    assert reachable_ids | set(case.case_file_evidence_ids) == evidence_ids


def test_camera_frame_occurs_while_panel_is_open_and_before_removal() -> None:
    case = CaseStore().load_public_case()
    evidence = {record.evidence_id: record for record in case.observations}

    assert evidence["E02"].occurred_at < evidence["E06"].occurred_at < evidence["E01"].occurred_at
    assert evidence["E06"].occurred_at == datetime(2026, 8, 30, 4, 42, 14, tzinfo=UTC)
    assert "9:42:14 p.m." in evidence["E06"].text


def test_base_motive_is_inferred_and_explicit_intent_stays_in_follow_ups() -> None:
    store = CaseStore()
    case = store.load_public_case()
    evidence = {record.evidence_id: record for record in case.observations}
    base_motive_text = " ".join(evidence[evidence_id].text for evidence_id in ("E10", "E11", "E12"))

    for explicit_statement in (
        "archive before normalization",
        "rowan requested",
        "wanted to preserve",
        "decided to hide",
        "once it is normalized",
    ):
        assert explicit_statement not in base_motive_text.lower()

    assert "G3-CAT-17" in evidence["E04"].text
    assert "G3-CAT-17" in evidence["E12"].text
    assert "R. Pike" in evidence["E11"].text
    assert "once it is normalized" in evidence["E14"].text.lower()
    assert "rowan entered the hold" in evidence["E15"].text.lower()

    base_ids = set(case.case_file_evidence_ids)
    solution = store.load_solution()
    preservation_rule = next(
        rule for rule in solution.claim_rules if rule.claim_id == "preservation_motive"
    )
    assert any(
        set(evidence_set) <= base_ids for evidence_set in preservation_rule.accepted_evidence_sets
    )


def test_solution_is_valid_against_public_case() -> None:
    store = CaseStore()
    public_case = store.load_public_case()
    solution = store.load_solution()

    assert solution.case_id == public_case.case_id
    assert solution.culprit_id in {suspect.suspect_id for suspect in public_case.suspects}
    assert solution.acceptable_evidence_sets[1] == [
        "E01",
        "E04",
        "E07",
        "E10",
        "E11",
        "E12",
    ]


def test_public_case_has_no_private_answer_key_fields() -> None:
    public_payload = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    public_dump = CaseStore().load_public_case().model_dump(mode="json")
    private_fields = {
        "culprit_id",
        "canonical_sequence",
        "acceptable_evidence_sets",
        "claim_rules",
        "contradiction_pairs",
        "exoneration_rules",
        "hints",
    }

    assert private_fields.isdisjoint(public_payload)
    assert private_fields.isdisjoint(public_dump)


def test_public_tools_work_without_solution_file(tmp_path: Path) -> None:
    case_directory = tmp_path / DEFAULT_CASE_ID
    case_directory.mkdir()
    shutil.copyfile(PUBLIC_PATH, case_directory / "public.json")
    store = CaseStore(tmp_path)

    observation = store.inspect_location(DEFAULT_CASE_ID, "security_screening")

    assert observation.evidence_ids == ["E04", "E07"]
    with pytest.raises(CaseNotFoundError, match=r"solution\.json"):
        store.load_solution(DEFAULT_CASE_ID)


def test_naive_datetime_is_rejected() -> None:
    payload = {
        "evidence_id": "E10",
        "title": "Naive timestamp",
        "text": "This timestamp has no timezone and must fail validation.",
        "source_type": "test",
        "source_reference": "test/naive",
        "occurred_at": "2026-08-30T04:42:18",
    }

    with pytest.raises(ValidationError, match="timezone"):
        EvidenceRecord.model_validate(payload)


def test_aware_datetime_is_coerced_to_utc() -> None:
    record = EvidenceRecord(
        evidence_id="E10",
        title="Aware timestamp",
        text="The timestamp uses an offset and should normalize to UTC.",
        source_type="test",
        source_reference="test/aware",
        occurred_at="2026-08-29T21:42:18-07:00",
    )

    assert record.occurred_at == datetime(2026, 8, 30, 4, 42, 18, tzinfo=UTC)
    assert record.occurred_at.tzinfo is UTC


@pytest.mark.parametrize("case_id", ["../midnight_museum", "Midnight Museum", "/tmp/case"])
def test_invalid_case_id_cannot_escape_case_directory(case_id: str) -> None:
    with pytest.raises(CaseNotFoundError, match="invalid case ID"):
        CaseStore().load_public_case(case_id)


def test_solution_rejects_unknown_evidence(tmp_path: Path) -> None:
    case_directory = tmp_path / DEFAULT_CASE_ID
    case_directory.mkdir()
    shutil.copyfile(PUBLIC_PATH, case_directory / "public.json")
    solution_payload = json.loads(SOLUTION_PATH.read_text(encoding="utf-8"))
    solution_payload["acceptable_evidence_sets"][0].append("E99")
    (case_directory / "solution.json").write_text(
        json.dumps(solution_payload),
        encoding="utf-8",
    )

    with pytest.raises(InvalidCaseDataError, match="unknown evidence"):
        CaseStore(tmp_path).load_solution(DEFAULT_CASE_ID)
