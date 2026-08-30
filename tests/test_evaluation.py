"""Tests for deterministic investigation evaluation metrics."""

from __future__ import annotations

from case_closed.case_store import CaseStore
from case_closed.evaluation import evaluate_state
from case_closed.schemas import Verdict
from case_closed.state import create_initial_state


def test_evaluation_accepts_grounded_solution() -> None:
    store = CaseStore()
    state = create_initial_state(store.load_public_case(), human_review_enabled=False)
    state["status"] = "resolved"
    state["round_count"] = 3
    state["discovered_evidence"] = [
        {"evidence_id": "E01"},
        {"evidence_id": "E04"},
        {"evidence_id": "E07"},
    ]
    state["proposed_verdict"] = Verdict(
        culprit_id="rowan_pike",
        summary="Rowan removed the sculpture before the blackout.",
        case_theory="Timing, exclusive access, and matching mass establish the case.",
        confidence=95,
        citations=["E01", "E04", "E07"],
    ).model_dump(mode="json")

    report = evaluate_state(state, store.load_solution())

    assert report.passed is True
    assert all(report.model_dump().values())


def test_evaluation_rejects_unsupported_state() -> None:
    store = CaseStore()
    state = create_initial_state(store.load_public_case(), human_review_enabled=False)

    report = evaluate_state(state, store.load_solution())

    assert report.passed is False
    assert report.status_resolved is False
    assert report.culprit_correct is False
