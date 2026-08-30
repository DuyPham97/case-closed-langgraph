"""Offline integration tests for the bounded LangGraph workflow."""

from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from case_closed.case_store import CaseStore
from case_closed.graph import build_investigation_graph
from case_closed.schemas import InvestigationAction, ProgressAssessment, Verdict
from case_closed.state import create_initial_state
from tests.fakes import ScriptedGateway


def _action(
    tool_name: str,
    target_id: str,
    *,
    topic: str | None = None,
) -> InvestigationAction:
    return InvestigationAction(
        tool_name=tool_name,
        target_id=target_id,
        topic=topic,
        reason=f"Check {target_id} next.",
    )


def _assessment(*, ready: bool = False) -> ProgressAssessment:
    return ProgressAssessment(
        summary="The evidence board has been updated.",
        hypotheses=[],
        ready_for_verdict=ready,
        next_leads=["Compare the removal time with access and equipment records."],
    )


def _verdict(
    culprit_id: str = "rowan_pike",
    citations: list[str] | None = None,
) -> Verdict:
    return Verdict(
        culprit_id=culprit_id,
        summary="Rowan Pike removed Aurora Circuit before the blackout.",
        case_theory=(
            "The plinth timing, exclusive access window, and equipment-case weight establish "
            "removal and possession."
        ),
        confidence=96,
        citations=citations or ["E01", "E04", "E07"],
    )


def _gold_gateway(*, verdict: Verdict | None = None) -> ScriptedGateway:
    return ScriptedGateway(
        actions=[
            _action("inspect_location", "display_case"),
            _action("compare_timeline", "weight_drop"),
            _action("inspect_location", "security_screening"),
        ],
        assessments=[_assessment(), _assessment(), _assessment(ready=True)],
        verdicts=[verdict or _verdict()],
    )


def _run_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def test_graph_interrupts_resumes_and_resolves() -> None:
    store = CaseStore()
    gateway = _gold_gateway()
    graph = build_investigation_graph(gateway, store, checkpointer=InMemorySaver())
    initial = create_initial_state(store.load_public_case(), human_review_enabled=True)
    config = _run_config("gold-with-human")

    paused = graph.invoke(initial, config=config)

    assert paused["status"] == "awaiting_human"
    assert paused["round_count"] == 2
    assert paused["__interrupt__"][0].value["question"].startswith("Which lead")

    completed = graph.invoke(Command(resume="Inspect the security screening records."), config)

    assert completed["status"] == "resolved"
    assert completed["human_review_completed"] is True
    assert completed["proposed_verdict"]["culprit_id"] == "rowan_pike"
    assert {item["evidence_id"] for item in completed["discovered_evidence"]} >= {
        "E01",
        "E04",
        "E07",
    }
    assert 'culprit_id": "rowan_pike' not in json.dumps(completed["public_case"])
    assert "acceptable_evidence_sets" not in json.dumps(completed)


def test_graph_resolves_without_interrupt_when_human_review_disabled() -> None:
    store = CaseStore()
    gateway = _gold_gateway()
    graph = build_investigation_graph(gateway, store)
    initial = create_initial_state(store.load_public_case(), human_review_enabled=False)

    completed = graph.invoke(initial)

    assert completed["status"] == "resolved"
    assert "__interrupt__" not in completed
    assert gateway.action_calls == 3
    assert gateway.verdict_calls == 1


def test_invalid_action_is_replanned_without_consuming_a_round() -> None:
    store = CaseStore()
    gateway = ScriptedGateway(
        actions=[
            _action("inspect_location", "moon_vault"),
            _action("inspect_location", "display_case"),
            _action("compare_timeline", "weight_drop"),
            _action("inspect_location", "security_screening"),
        ],
        assessments=[_assessment(), _assessment(), _assessment(ready=True)],
        verdicts=[_verdict()],
    )
    graph = build_investigation_graph(gateway, store)
    initial = create_initial_state(store.load_public_case(), human_review_enabled=False)

    completed = graph.invoke(initial)

    assert completed["status"] == "resolved"
    assert completed["invalid_action_count"] == 1
    assert completed["round_count"] == 3
    rejected = [event for event in completed["trace"] if event["event"] == "action_rejected"]
    assert rejected[0]["data"]["code"] == "unavailable_action"


def test_invalid_action_limit_ends_inconclusive() -> None:
    store = CaseStore()
    gateway = ScriptedGateway(
        actions=[
            _action("inspect_location", "moon_vault"),
            _action("compare_timeline", "midnight"),
        ],
        assessments=[],
        verdicts=[],
    )
    graph = build_investigation_graph(gateway, store)
    initial = create_initial_state(
        store.load_public_case(),
        human_review_enabled=False,
        max_invalid_actions=2,
    )

    completed = graph.invoke(initial)

    assert completed["status"] == "inconclusive"
    assert completed["round_count"] == 0
    assert completed["invalid_action_count"] == 2


def test_round_limit_ends_inconclusive_without_guessing() -> None:
    store = CaseStore()
    gateway = ScriptedGateway(
        actions=[_action("inspect_location", "lighting_booth")],
        assessments=[_assessment()],
        verdicts=[],
    )
    graph = build_investigation_graph(gateway, store)
    initial = create_initial_state(
        store.load_public_case(),
        human_review_enabled=False,
        max_rounds=1,
    )

    completed = graph.invoke(initial)

    assert completed["status"] == "inconclusive"
    assert completed["round_count"] == 1
    assert completed["proposed_verdict"] is None


def test_invalid_citation_is_repaired_once() -> None:
    store = CaseStore()
    gateway = _gold_gateway(verdict=_verdict(citations=["E01", "E04", "E99"]))
    gateway.repaired_verdicts.append(_verdict())
    graph = build_investigation_graph(gateway, store)
    initial = create_initial_state(store.load_public_case(), human_review_enabled=False)

    completed = graph.invoke(initial)

    assert completed["status"] == "resolved"
    assert completed["verdict_revision_count"] == 1
    assert gateway.repair_calls == 1
    assert completed["validation_errors"] == []


def test_unsupported_verdict_repairs_are_bounded() -> None:
    store = CaseStore()
    wrong_verdict = _verdict(culprit_id="theo_quinn", citations=["E01", "E04", "E07"])
    gateway = _gold_gateway(verdict=wrong_verdict)
    gateway.repaired_verdicts.append(wrong_verdict)
    graph = build_investigation_graph(gateway, store)
    initial = create_initial_state(store.load_public_case(), human_review_enabled=False)

    completed = graph.invoke(initial)

    assert completed["status"] == "inconclusive"
    assert completed["verdict_revision_count"] == 1
    assert completed["validation_errors"][0]["code"] == "unsupported_verdict"
    assert "rowan" not in completed["validation_errors"][0]["message"].lower()
