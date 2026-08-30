"""Output tests for the command-line portfolio interface."""

from __future__ import annotations

from pytest import CaptureFixture

from case_closed.case_store import CaseStore
from case_closed.cli import _json_result, _print_result
from case_closed.schemas import Verdict
from case_closed.state import create_initial_state


def test_cli_prints_grounded_resolved_verdict(capsys: CaptureFixture[str]) -> None:
    store = CaseStore()
    public_case = store.load_public_case()
    state = create_initial_state(public_case, human_review_enabled=False)
    verdict = Verdict(
        culprit_id="rowan_pike",
        summary="Rowan stole the artwork before the blackout.",
        case_theory="Three independent records establish timing, access, and possession.",
        confidence=95,
        citations=["E01", "E04", "E07"],
    )
    state["status"] = "resolved"
    state["round_count"] = 3
    state["proposed_verdict"] = verdict.model_dump(mode="json")

    _print_result(state, public_case)

    output = capsys.readouterr().out
    assert "Status: RESOLVED" in output
    assert "Culprit: Rowan Pike" in output
    assert "Evidence: E01, E04, E07" in output


def test_json_output_omits_interrupt_and_full_case() -> None:
    result = {
        "status": "resolved",
        "public_case": {"brief": "large payload"},
        "__interrupt__": ("internal",),
    }

    assert _json_result(result) == {"status": "resolved"}
