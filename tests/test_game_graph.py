"""Offline integration tests for the player-driven LangGraph workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from case_closed.case_store import CaseStore
from case_closed.game_graph import build_player_game_graph
from case_closed.game_schemas import (
    AccusationMatch,
    GameAction,
    GameActionRoute,
    GameDebrief,
    GameResult,
    PlayerAccusation,
)
from case_closed.game_state import create_player_game_state


@dataclass
class ScriptedPlayerGateway:
    """Return deterministic routes and debriefs without network access."""

    routes: list[GameActionRoute]
    matches: list[AccusationMatch] = field(
        default_factory=lambda: [AccusationMatch(motive_match=False, method_match=True)]
    )
    route_calls: int = 0
    match_calls: int = 0
    debrief_calls: int = 0
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

    def write_debrief(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        accusation: PlayerAccusation,
        result: GameResult,
    ) -> GameDebrief:
        del public_case, discovered_evidence, accusation
        self.debrief_calls += 1
        return GameDebrief(
            headline="Case closed" if result.tier == "solved" else "The file remains open",
            summary="The ending follows only from the evidence on the board.",
            evidence_analysis=["The cited records determine the strength of the accusation."],
            closing_line="The museum falls quiet again.",
        )


def _route(action_id: str) -> GameActionRoute:
    return GameActionRoute(
        action_id=action_id,
        needs_clarification=False,
        player_message="That line of inquiry has a clear next step.",
    )


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _start(gateway: ScriptedPlayerGateway, thread_id: str) -> tuple[object, dict[str, object]]:
    store = CaseStore()
    graph = build_player_game_graph(gateway, store, checkpointer=InMemorySaver())
    paused = graph.invoke(create_player_game_state(store.load_public_case()), _config(thread_id))
    return graph, paused


def test_complete_player_flow_uses_two_investigations_and_solves() -> None:
    gateway = ScriptedPlayerGateway(routes=[_route("inspect_security_screening")])
    graph, first_pause = _start(gateway, "solved-flow")
    config = _config("solved-flow")

    assert first_pause["__interrupt__"][0].value["phase"] == "visual_choice"
    second_pause = graph.invoke(
        Command(resume={"action_id": "inspect_display_case"}),
        config,
    )
    assert second_pause["investigation_count"] == 1
    assert second_pause["__interrupt__"][0].value["phase"] == "free_form"

    accusation_pause = graph.invoke(
        Command(resume={"request": "Compare Rowan's access with his equipment weight."}),
        config,
    )
    assert accusation_pause["investigation_count"] == 2
    assert accusation_pause["completed_action_ids"] == [
        "inspect_display_case",
        "inspect_security_screening",
    ]
    assert accusation_pause["__interrupt__"][0].value["phase"] == "accusation"

    completed = graph.invoke(
        Command(
            resume={
                "suspect_id": "rowan_pike",
                "motive": "He wanted Aurora Circuit for himself.",
                "method": "He hid it in his heavier equipment case before the blackout.",
            }
        ),
        config,
    )

    assert completed["status"] == "solved"
    assert completed["stage"] == "complete"
    assert completed["result"]["tier"] == "solved"
    assert completed["investigation_count"] == 2
    assert gateway.route_calls == 1
    assert gateway.match_calls == 1
    assert gateway.debrief_calls == 1
    assert completed["debrief"]["closing_line"] == (
        "The blackout did not hide the theft. It only told the museum when to start looking."
    )
    serialized = json.dumps(completed)
    assert "acceptable_evidence_sets" not in serialized
    assert "canonical_sequence" not in serialized
    assert (
        "Rowan Pike prepared a polarized insert from a catalog image before the gala"
        not in serialized
    )
    assert "solution_key" not in serialized
    assert "source_reference" not in serialized


def test_ambiguous_free_form_request_clarifies_without_spending_action() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[
            GameActionRoute(
                needs_clarification=True,
                clarification_question="Do you want the access log or equipment weights?",
                suggested_action_ids=[
                    "reconstruct_removal_window",
                    "inspect_security_screening",
                ],
                player_message="Two records could answer that.",
            ),
            _route("inspect_security_screening"),
        ]
    )
    graph, _ = _start(gateway, "clarify-flow")
    config = _config("clarify-flow")
    graph.invoke(Command(resume={"action_id": "inspect_display_case"}), config)

    clarification = graph.invoke(
        Command(resume={"request": "Check the records."}),
        config,
    )

    assert clarification["investigation_count"] == 1
    assert clarification["__interrupt__"][0].value["phase"] == "free_form_clarification"
    accusation = graph.invoke(
        Command(resume={"request": "Compare the equipment weights at security."}),
        config,
    )
    assert accusation["investigation_count"] == 2
    assert accusation["__interrupt__"][0].value["phase"] == "accusation"
    assert gateway.routed_requests == [
        "Check the records.",
        "Compare the equipment weights at security.",
    ]


def test_invalid_visual_choice_pauses_again_without_spending_action() -> None:
    gateway = ScriptedPlayerGateway(routes=[_route("inspect_security_screening")])
    graph, _ = _start(gateway, "invalid-visual")
    config = _config("invalid-visual")

    retry = graph.invoke(Command(resume={"action_id": "secret_tunnel"}), config)

    assert retry["investigation_count"] == 0
    assert retry["__interrupt__"][0].value["phase"] == "visual_choice"
    assert "error" in retry["__interrupt__"][0].value

    free_form = graph.invoke(
        Command(resume={"action_id": "inspect_display_case"}),
        config,
    )
    assert free_form["investigation_count"] == 1
    assert free_form["__interrupt__"][0].value["phase"] == "free_form"


def test_accusation_requires_culprit_motive_and_method_before_scoring() -> None:
    gateway = ScriptedPlayerGateway(routes=[_route("inspect_security_screening")])
    graph, _ = _start(gateway, "invalid-accusation")
    config = _config("invalid-accusation")
    graph.invoke(Command(resume={"action_id": "inspect_display_case"}), config)
    graph.invoke(Command(resume={"request": "Inspect the security weight records."}), config)

    retry = graph.invoke(
        Command(
            resume={
                "suspect_id": "rowan_pike",
                "motive": "He wanted the sculpture.",
            }
        ),
        config,
    )

    assert retry["__interrupt__"][0].value["phase"] == "accusation"
    assert "motive and method" in retry["__interrupt__"][0].value["error"]
    assert gateway.match_calls == 0

    completed = graph.invoke(
        Command(
            resume={
                "suspect_id": "rowan_pike",
                "motive": "He wanted the sculpture.",
                "method": "He hid it in his equipment case.",
            }
        ),
        config,
    )
    assert completed["result"]["tier"] == "solved"
    assert gateway.match_calls == 1


def test_correct_suspect_without_matching_story_gets_partial_tier() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[_route("ask_rowan_whereabouts")],
        matches=[AccusationMatch(motive_match=False, method_match=False)],
    )
    graph, _ = _start(gateway, "weak-flow")
    config = _config("weak-flow")
    graph.invoke(Command(resume={"action_id": "inspect_display_case"}), config)
    graph.invoke(Command(resume={"request": "Question Rowan about where he stood."}), config)

    completed = graph.invoke(
        Command(
            resume={
                "suspect_id": "rowan_pike",
                "motive": "He was jealous of the curator.",
                "method": "He took it somehow during the blackout.",
            }
        ),
        config,
    )

    assert completed["result"]["tier"] == "partial"
    assert completed["result"]["culprit_correct"] is True
    assert completed["result"]["motive_match"] is False
    assert completed["result"]["method_match"] is False
    assert completed["status"] == "closed"


def test_correct_suspect_with_motive_match_alone_solves() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[_route("inspect_security_screening")],
        matches=[AccusationMatch(motive_match=True, method_match=False)],
    )
    graph, _ = _start(gateway, "motive-solved-flow")
    config = _config("motive-solved-flow")
    graph.invoke(Command(resume={"action_id": "inspect_display_case"}), config)
    graph.invoke(Command(resume={"request": "Inspect the security weight records."}), config)

    completed = graph.invoke(
        Command(
            resume={
                "suspect_id": "rowan_pike",
                "motive": "He intended to take Aurora Circuit for himself.",
                "method": "He waited until after the gala and walked out with it.",
            }
        ),
        config,
    )

    assert completed["result"]["tier"] == "solved"
    assert completed["result"]["motive_match"] is True
    assert completed["result"]["method_match"] is False
    assert completed["status"] == "solved"


def test_wrong_suspect_with_matching_method_gets_partial_tier() -> None:
    gateway = ScriptedPlayerGateway(routes=[_route("inspect_security_screening")])
    graph, _ = _start(gateway, "wrong-flow")
    config = _config("wrong-flow")
    graph.invoke(Command(resume={"action_id": "inspect_display_case"}), config)
    graph.invoke(Command(resume={"request": "Inspect the security weight records."}), config)

    completed = graph.invoke(
        Command(
            resume={
                "suspect_id": "theo_quinn",
                "motive": "He wanted the artwork.",
                "method": "He concealed it in an equipment case before the blackout.",
            }
        ),
        config,
    )

    assert completed["result"]["tier"] == "partial"
    assert completed["result"]["culprit_correct"] is False
    assert completed["result"]["method_match"] is True


def test_wrong_suspect_without_matching_story_fails() -> None:
    gateway = ScriptedPlayerGateway(
        routes=[_route("inspect_security_screening")],
        matches=[AccusationMatch(motive_match=False, method_match=False)],
    )
    graph, _ = _start(gateway, "failed-flow")
    config = _config("failed-flow")
    graph.invoke(Command(resume={"action_id": "inspect_display_case"}), config)
    graph.invoke(Command(resume={"request": "Inspect the security weight records."}), config)

    completed = graph.invoke(
        Command(
            resume={
                "suspect_id": "mara_vale",
                "motive": "She wanted publicity.",
                "method": "She carried it away after the blackout.",
            }
        ),
        config,
    )

    assert completed["result"]["tier"] == "failed"
    assert completed["status"] == "closed"
