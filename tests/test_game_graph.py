"""Offline integration tests for the player-driven LangGraph workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from case_closed.case_store import CaseStore
from case_closed.game_graph import build_player_game_graph
from case_closed.game_schemas import (
    AccusationMatch,
    GameAction,
    GameActionRoute,
    PlayerAccusation,
)
from case_closed.game_state import create_player_game_state


@dataclass
class ScriptedPlayerGateway:
    """Return deterministic routes and theory matches without network access."""

    routes: list[GameActionRoute] = field(default_factory=list)
    matches: list[AccusationMatch] = field(
        default_factory=lambda: [AccusationMatch(motive_match=False, method_match=True)]
    )
    route_calls: int = 0
    match_calls: int = 0
    routed_requests: list[str] = field(default_factory=list)

    def route_free_form(
        self,
        request: str,
        actions: tuple[GameAction, ...],
        completed_action_ids: tuple[str, ...],
        discovered_evidence: tuple[dict[str, object], ...],
    ) -> GameActionRoute:
        del actions, completed_action_ids, discovered_evidence
        self.routed_requests.append(request)
        route = self.routes[self.route_calls]
        self.route_calls += 1
        return route

    def assess_accusation(
        self,
        accusation: PlayerAccusation,
        canonical_story: tuple[str, ...],
    ) -> AccusationMatch:
        del accusation
        assert "Rowan Pike" in " ".join(canonical_story)
        match = self.matches[self.match_calls]
        self.match_calls += 1
        return match


def _route(action_id: str) -> GameActionRoute:
    return GameActionRoute(
        action_id=action_id,
        needs_clarification=False,
        clarification_question=None,
        suggested_action_ids=[],
        player_message="That line of inquiry has a clear next step.",
    )


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _start(
    gateway: ScriptedPlayerGateway,
    thread_id: str,
) -> tuple[object, dict[str, object]]:
    store = CaseStore()
    graph = build_player_game_graph(gateway, store, checkpointer=InMemorySaver())
    paused = graph.invoke(create_player_game_state(store.load_public_case()), _config(thread_id))
    return graph, paused


def _accusation(
    suspect_id: str = "rowan_pike",
    motive: str = "He wanted to preserve Iris Venn's original timing until the estate reviewed it.",
    method: str = "He used calibration access, a flat decoy, and his heavier camera case.",
) -> dict[str, str]:
    return {"suspect_id": suspect_id, "motive": motive, "method": method}


def test_initial_pause_opens_full_dossier_with_two_optional_credits() -> None:
    gateway = ScriptedPlayerGateway()
    _, paused = _start(gateway, "initial-dossier")
    interrupt_payload = paused["__interrupt__"][0].value

    assert interrupt_payload["phase"] == "free_form"
    assert interrupt_payload["investigations_remaining"] == 2
    assert interrupt_payload["can_accuse"] is True
    assert paused["investigation_count"] == 0
    assert paused["completed_action_ids"] == []
    assert [record["evidence_id"] for record in paused["discovered_evidence"]] == [
        f"E{index:02d}" for index in range(1, 13)
    ]
    assert gateway.route_calls == 0


def test_player_can_solve_without_spending_a_deep_dive() -> None:
    gateway = ScriptedPlayerGateway(matches=[AccusationMatch(motive_match=True, method_match=True)])
    graph, _ = _start(gateway, "zero-credit-solve")
    config = _config("zero-credit-solve")

    accusation_pause = graph.invoke(Command(resume={"next_step": "accuse"}), config)
    assert accusation_pause["__interrupt__"][0].value["phase"] == "accusation"
    assert accusation_pause["investigation_count"] == 0

    completed = graph.invoke(Command(resume=_accusation()), config)

    assert completed["status"] == "solved"
    assert completed["stage"] == "complete"
    assert completed["result"]["tier"] == "solved"
    assert completed["investigation_count"] == 0
    assert gateway.route_calls == 0
    assert gateway.match_calls == 1
    assert completed["debrief"]["headline"] == "The case clicks shut"
    debrief_text = json.dumps(completed["debrief"])
    assert "9:42 p.m., before the automatic 10:00 p.m. blackout" in debrief_text
    assert "exposed the decoy at about 10:01 p.m." in debrief_text
    assert "until morning" not in debrief_text
    assert "while the blackout" not in debrief_text
    assert "three-second" not in debrief_text
    serialized = json.dumps(completed)
    assert "canonical_sequence" not in serialized
    assert "source_reference" not in serialized
    assert "solution_key" not in serialized


def test_one_deep_dive_adds_a_note_then_returns_to_open_dossier() -> None:
    gateway = ScriptedPlayerGateway(routes=[_route("ask_theo_who_heard_warning")])
    graph, _ = _start(gateway, "one-deep-dive")
    config = _config("one-deep-dive")

    next_move = graph.invoke(
        Command(resume={"request": "Ask Theo who heard that the timing could not be restored."}),
        config,
    )

    assert next_move["investigation_count"] == 1
    assert next_move["completed_action_ids"] == ["ask_theo_who_heard_warning"]
    assert next_move["discovered_evidence"][-1]["evidence_id"] == "E14"
    assert next_move["__interrupt__"][0].value["phase"] == "free_form"
    assert next_move["__interrupt__"][0].value["investigations_remaining"] == 1

    accusation_pause = graph.invoke(Command(resume={"next_step": "accuse"}), config)
    assert accusation_pause["__interrupt__"][0].value["phase"] == "accusation"
    assert accusation_pause["investigation_count"] == 1


def test_two_deep_dives_automatically_open_accusation() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[
            _route("ask_theo_who_heard_warning"),
            _route("crosscheck_rowans_case"),
        ]
    )
    graph, _ = _start(gateway, "two-deep-dives")
    config = _config("two-deep-dives")

    graph.invoke(
        Command(resume={"request": "Ask Theo who heard his warning."}),
        config,
    )
    accusation_pause = graph.invoke(
        Command(resume={"request": "Reconcile Rowan's case weight with the manifest."}),
        config,
    )

    assert accusation_pause["investigation_count"] == 2
    assert accusation_pause["completed_action_ids"] == [
        "ask_theo_who_heard_warning",
        "crosscheck_rowans_case",
    ]
    assert accusation_pause["discovered_evidence"][-2]["evidence_id"] == "E14"
    assert accusation_pause["discovered_evidence"][-1]["evidence_id"] == "E18"
    assert accusation_pause["__interrupt__"][0].value["phase"] == "accusation"
    assert accusation_pause["__interrupt__"][0].value["investigations_remaining"] == 0


def test_ambiguous_request_clarifies_without_spending_and_can_be_abandoned() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[
            GameActionRoute(
                action_id=None,
                needs_clarification=True,
                clarification_question="Do you want Theo's warning or Rowan's case records?",
                suggested_action_ids=[
                    "ask_theo_who_heard_warning",
                    "crosscheck_rowans_case",
                ],
                player_message="Two follow-ups could answer that.",
            )
        ]
    )
    graph, _ = _start(gateway, "clarify-or-accuse")
    config = _config("clarify-or-accuse")

    clarification = graph.invoke(Command(resume={"request": "Check what they knew."}), config)

    assert clarification["investigation_count"] == 0
    assert clarification["__interrupt__"][0].value["phase"] == "free_form_clarification"
    assert clarification["__interrupt__"][0].value["investigations_remaining"] == 2

    accusation = graph.invoke(Command(resume={"next_step": "accuse"}), config)
    assert accusation["investigation_count"] == 0
    assert accusation["__interrupt__"][0].value["phase"] == "accusation"


def test_repeated_routed_action_clarifies_without_spending_second_credit() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[
            _route("ask_theo_who_heard_warning"),
            _route("ask_theo_who_heard_warning"),
        ]
    )
    graph, _ = _start(gateway, "repeat-route")
    config = _config("repeat-route")
    graph.invoke(Command(resume={"request": "Ask Theo about the warning."}), config)

    clarification = graph.invoke(
        Command(resume={"request": "Ask Theo the same question again."}),
        config,
    )

    assert clarification["investigation_count"] == 1
    assert clarification["completed_action_ids"] == ["ask_theo_who_heard_warning"]
    assert clarification["__interrupt__"][0].value["phase"] == "free_form_clarification"
    assert clarification["__interrupt__"][0].value["investigations_remaining"] == 1


def test_empty_next_move_pauses_again_without_spending_credit() -> None:
    graph, _ = _start(ScriptedPlayerGateway(), "empty-next-move")
    config = _config("empty-next-move")

    retry = graph.invoke(Command(resume={"request": "  "}), config)

    assert retry["investigation_count"] == 0
    assert retry["__interrupt__"][0].value["phase"] == "free_form"
    assert "error" in retry["__interrupt__"][0].value


def test_accusation_requires_culprit_motive_and_method_before_scoring() -> None:
    gateway = ScriptedPlayerGateway()
    graph, _ = _start(gateway, "invalid-accusation")
    config = _config("invalid-accusation")
    graph.invoke(Command(resume={"next_step": "accuse"}), config)

    retry = graph.invoke(
        Command(resume={"suspect_id": "rowan_pike", "motive": "Preservation."}),
        config,
    )

    assert retry["__interrupt__"][0].value["phase"] == "accusation"
    assert "motive and method" in retry["__interrupt__"][0].value["error"]
    assert gateway.match_calls == 0

    completed = graph.invoke(Command(resume=_accusation()), config)
    assert completed["result"]["tier"] == "solved"
    assert gateway.match_calls == 1


@pytest.mark.parametrize(
    ("suspect_id", "match", "expected_tier", "expected_headline"),
    [
        (
            "rowan_pike",
            AccusationMatch(motive_match=False, method_match=False),
            "partial",
            "The theory stops short",
        ),
        (
            "rowan_pike",
            AccusationMatch(motive_match=True, method_match=False),
            "solved",
            "The case clicks shut",
        ),
        (
            "theo_quinn",
            AccusationMatch(motive_match=False, method_match=True),
            "partial",
            "The theory stops short",
        ),
        (
            "mara_vale",
            AccusationMatch(motive_match=False, method_match=False),
            "failed",
            "The theory does not hold",
        ),
    ],
)
def test_scoring_tiers_work_without_forced_investigations(
    suspect_id: str,
    match: AccusationMatch,
    expected_tier: str,
    expected_headline: str,
) -> None:
    thread_id = f"tier-{suspect_id}-{expected_tier}"
    gateway = ScriptedPlayerGateway(matches=[match])
    graph, _ = _start(gateway, thread_id)
    config = _config(thread_id)
    graph.invoke(Command(resume={"next_step": "accuse"}), config)

    completed = graph.invoke(Command(resume=_accusation(suspect_id=suspect_id)), config)

    assert completed["result"]["tier"] == expected_tier
    assert completed["investigation_count"] == 0
    assert completed["status"] == ("solved" if expected_tier == "solved" else "closed")
    assert completed["debrief"]["headline"] == expected_headline
    debrief_text = json.dumps(completed["debrief"])
    if expected_tier == "solved":
        assert "Rowan Pike" in debrief_text
        assert "9:42" in debrief_text
        assert "10:01 p.m." in debrief_text
    else:
        assert "sealed reconstruction remains unopened" in debrief_text
        for spoiler in (
            "Rowan Pike",
            "Iris Venn",
            "preservation",
            "polarized",
            "decoy",
            "hard case",
            "equipment case",
            "1.84",
            "9:42",
            "10:01",
            "controller migration",
        ):
            assert spoiler.casefold() not in debrief_text.casefold()
