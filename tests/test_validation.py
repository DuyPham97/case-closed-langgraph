"""Tests for deterministic action and verdict validation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from case_closed.schemas import (
    Artwork,
    AvailableActions,
    ClaimRule,
    ContradictionRule,
    EvidenceRecord,
    ExonerationRule,
    Incident,
    InterviewOption,
    InvestigationAction,
    Location,
    PublicCase,
    RevealRoute,
    SolutionKey,
    Suspect,
    TimelineEvent,
    Verdict,
)
from case_closed.validation import validate_action, validate_verdict


@pytest.fixture
def public_case() -> PublicCase:
    occurred_at = datetime(2026, 8, 30, 7, tzinfo=UTC)
    evidence = [
        EvidenceRecord(
            evidence_id=f"ev-{index}",
            title=f"Evidence {index}",
            text=f"Public evidence text {index}",
            source_type="testimony",
            source_reference=f"source-{index}",
            occurred_at=occurred_at + timedelta(minutes=index),
        )
        for index in range(1, 4)
    ]
    return PublicCase(
        schema_version=1,
        case_id="museum-heist",
        title="The Museum Heist",
        display_timezone="America/Los_Angeles",
        brief="A painting vanished during a short blackout.",
        artwork=Artwork(name="Moonlight", mass_kg=4.2, dimensions_cm=(60, 40, 5)),
        incident=Incident(
            blackout_started_at=occurred_at,
            blackout_ended_at=occurred_at + timedelta(minutes=5),
            discovered_at=occurred_at + timedelta(minutes=6),
        ),
        suspects=[
            Suspect(
                suspect_id="suspect-a",
                name="Avery",
                role="Curator",
                public_profile="Closed the gallery.",
            ),
            Suspect(
                suspect_id="suspect-b",
                name="Blake",
                role="Guard",
                public_profile="Patrolled the east wing.",
            ),
        ],
        locations=[
            Location(
                location_id="gallery",
                name="Gallery",
                description="The room where the painting was displayed.",
            )
        ],
        available_actions=AvailableActions(
            location_ids=["gallery"],
            interview_options=[
                InterviewOption(suspect_id="suspect-a", topic_ids=["alibi"]),
                InterviewOption(suspect_id="suspect-b", topic_ids=["alibi"]),
            ],
            timeline_anchor_ids=["blackout"],
        ),
        observations=evidence,
        timeline_events=[
            TimelineEvent(
                event_id="blackout",
                starts_at=occurred_at,
                ends_at=occurred_at + timedelta(minutes=5),
                subject_ids=["suspect-a", "suspect-b"],
                summary="The gallery lost power.",
                source_reference="alarm-log",
            )
        ],
        reveal_routes=[
            RevealRoute(
                tool_name="inspect_location",
                target_id="gallery",
                summary="The frame was inspected.",
                evidence_ids=["ev-1"],
            ),
            RevealRoute(
                tool_name="interview_suspect",
                target_id="suspect-a",
                topic="alibi",
                summary="Avery gave an alibi.",
                evidence_ids=["ev-2"],
            ),
            RevealRoute(
                tool_name="interview_suspect",
                target_id="suspect-b",
                topic="alibi",
                summary="Blake confirmed the alarm sequence.",
                evidence_ids=[],
            ),
            RevealRoute(
                tool_name="compare_timeline",
                target_id="blackout",
                summary="The timeline exposes a contradiction.",
                evidence_ids=["ev-3"],
            ),
        ],
    )


@pytest.fixture
def solution_key() -> SolutionKey:
    return SolutionKey(
        schema_version=1,
        case_id="museum-heist",
        culprit_id="suspect-a",
        canonical_sequence=["ev-1", "ev-2", "ev-3"],
        acceptable_evidence_sets=[["ev-1", "ev-2"]],
        claim_rules=[
            ClaimRule(
                claim_id="access",
                description="The culprit could access the artwork.",
                accepted_evidence_sets=[["ev-1"]],
            )
        ],
        contradiction_pairs=[
            ContradictionRule(
                statement_evidence_id="ev-2",
                rebuttal_evidence_ids=["ev-3"],
            )
        ],
        exoneration_rules=[
            ExonerationRule(
                suspect_id="suspect-b",
                summary="The alarm log supports Blake's account.",
                evidence_ids=["ev-3"],
            )
        ],
        hints=["A private hint naming suspect-a."],
    )


def _action(**changes: object) -> InvestigationAction:
    values: dict[str, object] = {
        "tool_name": "inspect_location",
        "target_id": "gallery",
        "topic": None,
        "reason": "Inspect the scene.",
    }
    values.update(changes)
    return InvestigationAction.model_validate(values)


def _verdict(**changes: object) -> Verdict:
    values: dict[str, object] = {
        "culprit_id": "suspect-a",
        "summary": "Avery took the painting.",
        "case_theory": "The physical clue and false alibi establish the sequence.",
        "confidence": 90,
        "citations": ["ev-1", "ev-2", "ev-3"],
    }
    values.update(changes)
    return Verdict.model_validate(values)


def test_validate_action_accepts_available_unexplored_route(public_case: PublicCase) -> None:
    result = validate_action(
        _action(),
        public_case=public_case,
        discovered_evidence_ids=set(),
    )

    assert result.is_valid
    assert result.as_state_error() is None


def test_validate_action_rejects_unknown_route(public_case: PublicCase) -> None:
    result = validate_action(
        _action(target_id="roof"),
        public_case=public_case,
        discovered_evidence_ids=set(),
    )

    assert not result.is_valid
    assert result.code == "unavailable_action"


def test_validate_action_rejects_exhausted_route(public_case: PublicCase) -> None:
    result = validate_action(
        _action(),
        public_case=public_case,
        discovered_evidence_ids={"ev-1"},
    )

    assert not result.is_valid
    assert result.code == "exhausted_action"


def test_validate_action_rejects_unknown_discovered_evidence(public_case: PublicCase) -> None:
    with pytest.raises(ValueError, match="public case"):
        validate_action(
            _action(),
            public_case=public_case,
            discovered_evidence_ids={"invented-evidence"},
        )


def test_validate_verdict_accepts_supported_discovered_citations(
    public_case: PublicCase,
    solution_key: SolutionKey,
) -> None:
    result = validate_verdict(
        _verdict(),
        public_case=public_case,
        discovered_evidence_ids={"ev-1", "ev-2", "ev-3"},
        solution_key=solution_key,
    )

    assert result.is_valid
    assert result.code == "valid"


def test_validate_verdict_rejects_undiscovered_citation_before_private_scoring(
    public_case: PublicCase,
    solution_key: SolutionKey,
) -> None:
    result = validate_verdict(
        _verdict(),
        public_case=public_case,
        discovered_evidence_ids={"ev-1", "ev-2"},
        solution_key=solution_key,
    )

    assert not result.is_valid
    assert result.code == "undiscovered_citation"


@pytest.mark.parametrize(
    "verdict",
    [
        _verdict(culprit_id="suspect-b"),
        _verdict(citations=["ev-1"]),
        _verdict(citations=["ev-1", "ev-2"]),
    ],
    ids=["wrong-culprit", "missing-support", "unrebutted-contradiction"],
)
def test_private_verdict_failures_use_one_redacted_error(
    public_case: PublicCase,
    solution_key: SolutionKey,
    verdict: Verdict,
) -> None:
    result = validate_verdict(
        verdict,
        public_case=public_case,
        discovered_evidence_ids={"ev-1", "ev-2", "ev-3"},
        solution_key=solution_key,
    )
    serialized = result.model_dump_json()

    assert not result.is_valid
    assert result.code == "unsupported_verdict"
    assert solution_key.culprit_id not in serialized
    assert solution_key.hints[0] not in serialized
    assert "ev-2" not in serialized
    assert "ev-3" not in serialized


def test_validate_verdict_rejects_duplicate_citations(
    public_case: PublicCase,
    solution_key: SolutionKey,
) -> None:
    result = validate_verdict(
        _verdict(citations=["ev-1", "ev-1", "ev-2", "ev-3"]),
        public_case=public_case,
        discovered_evidence_ids={"ev-1", "ev-2", "ev-3"},
        solution_key=solution_key,
    )

    assert not result.is_valid
    assert result.code == "duplicate_citation"


def test_validate_verdict_case_mismatch_error_is_redacted(
    public_case: PublicCase,
    solution_key: SolutionKey,
) -> None:
    mismatched_solution = solution_key.model_copy(update={"case_id": "private-case-id"})

    with pytest.raises(ValueError) as error:
        validate_verdict(
            _verdict(),
            public_case=public_case,
            discovered_evidence_ids={"ev-1", "ev-2", "ev-3"},
            solution_key=mismatched_solution,
        )

    assert "private-case-id" not in str(error.value)
    assert solution_key.culprit_id not in str(error.value)
