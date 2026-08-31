"""Player-driven LangGraph workflow with two optional deep dives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from case_closed.case_store import CaseStore
from case_closed.game_catalog import (
    all_game_actions,
    get_game_action,
)
from case_closed.game_schemas import (
    AccusationMatch,
    GameAction,
    GameActionRoute,
    GameDebrief,
    GameResult,
    PlayerAccusation,
)
from case_closed.game_state import PlayerGameState
from case_closed.schemas import PublicCase, ToolObservation
from case_closed.state import JsonObject, TraceEvent
from case_closed.tools import create_case_tools


class PlayerGameGatewayProtocol(Protocol):
    """Structured model operations used by the player game graph."""

    def route_free_form(
        self,
        request: str,
        actions: tuple[GameAction, ...],
        completed_action_ids: tuple[str, ...],
        discovered_evidence: tuple[dict[str, object], ...],
    ) -> GameActionRoute:
        """Map natural language to one allowed scripted action."""

    def assess_accusation(
        self,
        accusation: PlayerAccusation,
        canonical_story: tuple[str, ...],
    ) -> AccusationMatch:
        """Compare the player's motive and method with the private story."""


def build_player_game_graph(
    gateway: PlayerGameGatewayProtocol,
    store: CaseStore,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """Compile the player-led game with optional inquiry and accusation loops."""
    builder = StateGraph(PlayerGameState)

    def request_next_move(state: PlayerGameState) -> dict[str, object]:
        remaining = 2 - state["investigation_count"]
        payload: dict[str, object] = {
            "phase": "free_form",
            "title": "Choose your next move",
            "prompt": (
                "Use a deep-dive credit to pursue one line of inquiry, or make your "
                "accusation whenever your theory is ready."
            ),
            "investigations_remaining": remaining,
            "can_accuse": True,
        }
        while True:
            response = interrupt(payload)
            if _read_text(response, "next_step") == "accuse":
                return {
                    "stage": "accusation",
                    "free_form_request": None,
                    "routed_action": None,
                    "pending_action_id": None,
                    "trace": _trace(
                        "request_next_move",
                        "accusation_requested",
                        state,
                        "The detective is ready to state a theory.",
                    ),
                }
            request = _read_text(response, "request")
            if request and remaining > 0:
                return {
                    "stage": "free_form",
                    "free_form_request": request,
                    "routed_action": None,
                    "trace": _trace(
                        "request_next_move",
                        "free_form_received",
                        state,
                        "The detective submitted a line of inquiry.",
                    ),
                }
            payload["error"] = (
                "Both deep-dive credits are spent. Make your accusation."
                if remaining == 0
                else "Describe one inquiry, or make your accusation."
            )

    def route_free_form(state: PlayerGameState) -> dict[str, object]:
        request = state["free_form_request"]
        if request is None:
            raise ValueError("free-form routing requires a player request")
        routed = gateway.route_free_form(
            request,
            all_game_actions(),
            tuple(state["completed_action_ids"]),
            _object_tuple(state["discovered_evidence"]),
        )
        validated = _validate_routed_action(routed, state["completed_action_ids"])
        action_id = validated.action_id if not validated.needs_clarification else None
        return {
            "pending_action_id": action_id,
            "routed_action": validated.model_dump(mode="json"),
            "trace": _trace(
                "route_free_form",
                "request_clarified" if validated.needs_clarification else "request_understood",
                state,
                validated.player_message,
            ),
        }

    def clarify_free_form(state: PlayerGameState) -> dict[str, object]:
        routed = GameActionRoute.model_validate(state["routed_action"])
        options = [
            _public_action(get_game_action(action_id))
            for action_id in routed.suggested_action_ids
            if action_id not in state["completed_action_ids"]
        ]
        payload: dict[str, object] = {
            "phase": "free_form_clarification",
            "title": "Narrow your investigation",
            "prompt": routed.clarification_question,
            "suggestions": options,
            "investigations_remaining": 2 - state["investigation_count"],
            "can_accuse": True,
        }
        while True:
            response = interrupt(payload)
            if _read_text(response, "next_step") == "accuse":
                return {
                    "stage": "accusation",
                    "free_form_request": None,
                    "routed_action": None,
                    "pending_action_id": None,
                    "trace": _trace(
                        "clarify_free_form",
                        "accusation_requested",
                        state,
                        "The detective set the unfinished inquiry aside.",
                    ),
                }
            request = _read_text(response, "request")
            if request:
                return {
                    "stage": "free_form",
                    "free_form_request": request,
                    "routed_action": None,
                    "pending_action_id": None,
                    "trace": _trace(
                        "clarify_free_form",
                        "clarification_received",
                        state,
                        "The detective refined the line of inquiry.",
                    ),
                }
            payload["error"] = "Clarify what you want to investigate."

    def execute_action(state: PlayerGameState) -> dict[str, object]:
        if state["investigation_count"] >= 2:
            raise ValueError("the investigation budget has already been spent")
        action_id = state["pending_action_id"]
        if action_id is None:
            raise ValueError("an action must be selected before execution")
        action = get_game_action(action_id)
        tools = {tool.name: tool for tool in create_case_tools(store, state["case_id"])}
        raw_observation = tools[action.tool_name].invoke(_tool_arguments(action))
        observation = (
            raw_observation
            if isinstance(raw_observation, ToolObservation)
            else ToolObservation.model_validate(raw_observation)
        )
        public_case = store.load_public_case(state["case_id"])
        evidence_by_id = {evidence.evidence_id: evidence for evidence in public_case.observations}
        existing_ids = {evidence["evidence_id"] for evidence in state["discovered_evidence"]}
        evidence = [
            {
                "evidence_id": record.evidence_id,
                "title": record.title,
                "text": record.text,
                "occurred_at": record.occurred_at.isoformat(),
            }
            for evidence_id in observation.evidence_ids
            if evidence_id not in existing_ids
            for record in (evidence_by_id[evidence_id],)
        ]
        next_count = state["investigation_count"] + 1
        return {
            "investigation_count": next_count,
            "completed_action_ids": [action_id],
            "discovered_evidence": evidence,
            "last_observation": observation.model_dump(mode="json"),
            "pending_action_id": None,
            "free_form_request": None,
            "routed_action": None,
            "stage": "free_form" if next_count < 2 else "accusation",
            "trace": _trace(
                "execute_action",
                "investigation_completed",
                state,
                f"{action.title}: {observation.summary}",
                {
                    "action_id": action_id,
                    "evidence_ids": list(observation.evidence_ids),
                    "investigation_count": next_count,
                },
            ),
        }

    def request_accusation(state: PlayerGameState) -> dict[str, object]:
        public_case = store.load_public_case(state["case_id"])
        payload = _accusation_interrupt_payload(public_case, state)
        while True:
            response = interrupt(payload)
            try:
                accusation = _validate_accusation(response, public_case)
            except (TypeError, ValueError) as exc:
                payload = _accusation_interrupt_payload(public_case, state, str(exc))
                continue
            return {
                "accusation": accusation.model_dump(mode="json"),
                "trace": _trace(
                    "request_accusation",
                    "accusation_received",
                    state,
                    "The detective locked in a final accusation.",
                    {
                        "suspect_id": accusation.suspect_id,
                    },
                ),
            }

    def score_accusation(state: PlayerGameState) -> dict[str, object]:
        accusation = PlayerAccusation.model_validate(state["accusation"])
        solution = store.load_solution(state["case_id"])
        match = gateway.assess_accusation(
            accusation,
            tuple(solution.canonical_sequence),
        )
        correct_suspect = accusation.suspect_id == solution.culprit_id
        narrative_match = match.motive_match or match.method_match
        if correct_suspect and narrative_match:
            tier = "solved"
        elif correct_suspect or narrative_match:
            tier = "partial"
        else:
            tier = "failed"
        result = GameResult(
            tier=tier,
            accused_suspect_id=accusation.suspect_id,
            culprit_correct=correct_suspect,
            motive_match=match.motive_match,
            method_match=match.method_match,
        )
        return {
            "result": result.model_dump(mode="json"),
            "trace": _trace(
                "score_accusation",
                "accusation_scored",
                state,
                _result_message(result),
                {
                    "tier": result.tier,
                    "culprit_correct": result.culprit_correct,
                    "motive_match": result.motive_match,
                    "method_match": result.method_match,
                },
            ),
        }

    def build_debrief(state: PlayerGameState) -> dict[str, object]:
        result = GameResult.model_validate(state["result"])
        public_case = store.load_public_case(state["case_id"])
        debrief = _build_debrief(public_case, result)
        return {
            "debrief": debrief.model_dump(mode="json"),
            "stage": "complete",
            "status": "solved" if result.tier == "solved" else "closed",
            "trace": _trace(
                "build_debrief",
                "case_closed",
                state,
                debrief.headline,
            ),
        }

    builder.add_node("request_next_move", request_next_move)
    builder.add_node("execute_action", execute_action)
    builder.add_node("route_free_form", route_free_form)
    builder.add_node("clarify_free_form", clarify_free_form)
    builder.add_node("request_accusation", request_accusation)
    builder.add_node("score_accusation", score_accusation)
    builder.add_node("build_debrief", build_debrief)

    builder.add_edge(START, "request_next_move")
    builder.add_conditional_edges(
        "request_next_move",
        _route_after_player_choice,
        {"investigate": "route_free_form", "accuse": "request_accusation"},
    )
    builder.add_conditional_edges(
        "execute_action",
        _route_after_execution,
        {"free_form": "request_next_move", "accusation": "request_accusation"},
    )
    builder.add_conditional_edges(
        "route_free_form",
        _route_after_intent,
        {"execute": "execute_action", "clarify": "clarify_free_form"},
    )
    builder.add_conditional_edges(
        "clarify_free_form",
        _route_after_player_choice,
        {"investigate": "route_free_form", "accuse": "request_accusation"},
    )
    builder.add_edge("request_accusation", "score_accusation")
    builder.add_edge("score_accusation", "build_debrief")
    builder.add_edge("build_debrief", END)

    return builder.compile(checkpointer=checkpointer)


def _route_after_execution(state: PlayerGameState) -> str:
    return "free_form" if state["investigation_count"] < 2 else "accusation"


def _route_after_player_choice(state: PlayerGameState) -> str:
    return "accuse" if state["stage"] == "accusation" else "investigate"


def _route_after_intent(state: PlayerGameState) -> str:
    routed = GameActionRoute.model_validate(state["routed_action"])
    return "clarify" if routed.needs_clarification else "execute"


def _validate_routed_action(
    routed: GameActionRoute,
    completed_action_ids: list[str],
) -> GameActionRoute:
    available = {
        action.action_id: action
        for action in all_game_actions()
        if action.action_id not in completed_action_ids
    }
    if not routed.needs_clarification and routed.action_id in available:
        return routed
    suggested = [action_id for action_id in routed.suggested_action_ids if action_id in available][
        :3
    ]
    if not suggested:
        suggested = list(available)[:3]
    return GameActionRoute(
        action_id=None,
        needs_clarification=True,
        clarification_question=(
            routed.clarification_question or "Which lead should we pursue with this deep dive?"
        ),
        suggested_action_ids=suggested,
        player_message=(
            routed.player_message
            if routed.needs_clarification
            else "That lead is too broad. Narrow the inquiry."
        ),
    )


def _validate_accusation(
    response: object,
    public_case: PublicCase,
) -> PlayerAccusation:
    if not isinstance(response, Mapping):
        raise TypeError("Choose a suspect and explain both motive and method.")
    suspect_id = _read_text(response, "suspect_id")
    motive = _read_text(response, "motive")
    method = _read_text(response, "method")
    if not suspect_id or not motive or not method:
        raise ValueError("Choose a suspect and explain both motive and method.")
    suspect_ids = {suspect.suspect_id for suspect in public_case.suspects}
    if suspect_id not in suspect_ids:
        raise ValueError("Choose one of the four suspects.")
    return PlayerAccusation(
        suspect_id=suspect_id,
        motive=motive,
        method=method,
    )


def _accusation_interrupt_payload(
    public_case: PublicCase,
    state: PlayerGameState,
    error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "phase": "accusation",
        "title": "Name the culprit",
        "prompt": "Choose a suspect, then explain their motive and how they did it.",
        "suspects": [
            {
                "suspect_id": suspect.suspect_id,
                "name": suspect.name,
                "role": suspect.role,
            }
            for suspect in public_case.suspects
        ],
        "evidence": [
            {
                key: record[key]
                for key in ("evidence_id", "title", "text", "occurred_at")
                if key in record
            }
            for record in state["discovered_evidence"]
        ],
        "investigations_remaining": max(0, 2 - state["investigation_count"]),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _public_action(action: GameAction) -> dict[str, object]:
    return {
        "action_id": action.action_id,
        "title": action.title,
        "description": action.description,
        "category": action.category,
        "image_path": action.image_path,
    }


def _read_text(response: object, key: str) -> str:
    value: object
    if isinstance(response, str):
        value = response
    elif isinstance(response, Mapping):
        value = response.get(key)
    else:
        return ""
    return value.strip() if isinstance(value, str) else ""


def _tool_arguments(action: GameAction) -> dict[str, str]:
    if action.tool_name == "inspect_location":
        return {"location_id": action.target_id}
    if action.tool_name == "compare_timeline":
        return {"anchor_id": action.target_id}
    if action.topic is None:
        raise ValueError("interview actions require a topic")
    return {"suspect_id": action.target_id, "topic": action.topic}


def _result_message(result: GameResult) -> str:
    if result.tier == "solved":
        return "The culprit and the theory fit the case."
    if result.tier == "partial":
        return "Part of the accusation fits, but the complete theory does not."
    return "The accusation does not fit the case."


def _build_debrief(public_case: PublicCase, result: GameResult) -> GameDebrief:
    suspect = next(
        (
            suspect
            for suspect in public_case.suspects
            if suspect.suspect_id == result.accused_suspect_id
        ),
        None,
    )
    if suspect is None:
        raise ValueError("game result references an unknown suspect")

    if result.tier == "solved":
        return GameDebrief(
            headline="The case clicks shut",
            summary=" ".join(
                (
                    f"{suspect.name} was the only recorded occupant of Gallery 3 when the plinth "
                    "fell from 1.84 kilograms to zero at 9:42 p.m., before the automatic 10:00 "
                    "p.m. blackout.",
                    "The taped contact and polarized insert kept the empty display convincing "
                    "until the smart glass rebooted and exposed the decoy at about 10:01 p.m.",
                    "The hard equipment case then gained exactly 1.84 kilograms while its listed "
                    "contents stayed unchanged.",
                    _solved_match_summary(result),
                )
            ),
            evidence_analysis=[
                "The plinth changed at 9:42:18 p.m.; the preloaded blackout did not begin until "
                "10:00 p.m.",
                "The taped rear contact masked an open panel, and the polarized insert looked "
                "flat after the smart glass rebooted.",
                "The hard case gained exactly the sculpture's 1.84-kilogram mass with no manifest "
                "change or recorded handoff.",
                "The migration note, preservation annotations, and rejected delay request "
                "document the threatened loss of Iris Venn's timing.",
            ],
            closing_line=_closing_line(result),
        )

    if result.tier == "partial":
        headline = "The theory stops short"
        summary = " ".join(
            (
                "Some part of the accusation is supported, but the three claims do not yet form "
                "one coherent case.",
                "The verdict cards show only which categories matched; the sealed reconstruction "
                "remains unopened.",
                "Review the dossier for independent records that connect opportunity, reason, "
                "and physical sequence to the same person.",
            )
        )
    else:
        headline = "The theory does not hold"
        summary = " ".join(
            (
                "The accusation does not yet align with the case record.",
                "No conclusion follows from it, and the sealed reconstruction remains unopened.",
                "Revisit the dossier and build a single chain from opportunity, reason, and "
                "physical evidence before accusing again.",
            )
        )

    return GameDebrief(
        headline=headline,
        summary=summary,
        evidence_analysis=[
            "Build the timeline from independent records before relying on anyone's account.",
            "Test each personal statement against a separate physical or timestamped record.",
            "A complete motive should explain why the event mattered at that moment, not merely "
            "why the object was valuable.",
            "A complete method should account for access, timing, and aftermath without "
            "unsupported steps.",
        ],
        closing_line=_closing_line(result),
    )


def _solved_match_summary(result: GameResult) -> str:
    if result.motive_match and result.method_match:
        return "Both the preservation motive and the physical sequence match your theory."
    if result.motive_match:
        return "Your preservation motive matches the record, although the proposed method does not."
    return "Your method matches the physical trail, although the proposed motive does not."


def _closing_line(result: GameResult) -> str:
    if result.tier == "solved":
        return "The blackout did not hide the theft. It only told the museum when to start looking."
    if result.tier == "partial":
        return "A suspicion is not a case. The gallery waits for a theory that fits the facts."
    return "The lights are back, but this theory leaves the real trail in shadow."


def _object_tuple(values: list[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    return tuple(dict(value) for value in values)


def _trace(
    node: str,
    event: str,
    state: PlayerGameState,
    message: str,
    data: Mapping[str, object] | None = None,
) -> list[TraceEvent]:
    return [
        TraceEvent(
            node=node,
            event=event,
            round_count=state["investigation_count"],
            message=message,
            data=cast(JsonObject, dict(data or {})),
        )
    ]
