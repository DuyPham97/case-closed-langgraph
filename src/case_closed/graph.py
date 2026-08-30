"""Bounded LangGraph workflow for the evidence-grounded investigation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from case_closed.case_store import CaseStore
from case_closed.schemas import (
    InvestigationAction,
    ProgressAssessment,
    PublicCase,
    ToolObservation,
    Verdict,
)
from case_closed.state import CaseState, JsonObject, TraceEvent
from case_closed.tools import create_case_tools
from case_closed.validation import validate_action, validate_verdict

HUMAN_REVIEW_AFTER_ROUNDS = 2
NO_PROGRESS_INTERRUPT_THRESHOLD = 2


class InvestigationGateway(Protocol):
    """Structured model operations required by the graph."""

    def plan_action(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        hypotheses: tuple[dict[str, object], ...],
        round_count: int,
        max_rounds: int,
        human_direction: str | None,
    ) -> InvestigationAction:
        """Choose the next public investigation action."""

    def assess_progress(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        previous_hypotheses: tuple[dict[str, object], ...],
        round_count: int,
        max_rounds: int,
    ) -> ProgressAssessment:
        """Update grounded hypotheses and suggest remaining leads."""

    def draft_verdict(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        hypotheses: tuple[dict[str, object], ...],
    ) -> Verdict:
        """Draft a verdict using discovered evidence only."""

    def repair_verdict(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        previous_verdict: dict[str, object],
        validation_errors: tuple[dict[str, object], ...],
    ) -> Verdict:
        """Repair a rejected verdict without receiving private solution data."""


def build_investigation_graph(
    gateway: InvestigationGateway,
    store: CaseStore,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph:
    """Compile the detective workflow with injected model and persistence."""

    builder = StateGraph(CaseState)

    def load_case(state: CaseState) -> dict[str, object]:
        public_case = store.load_public_case(state["case_id"])
        return {
            "public_case": public_case.model_dump(mode="json"),
            "trace": _trace("load_case", "case_loaded", state, public_case.title),
        }

    def plan_action(state: CaseState) -> dict[str, object]:
        action = gateway.plan_action(
            _object_dict(state["public_case"]),
            _object_tuple(state["discovered_evidence"]),
            _object_tuple(state["hypotheses"]),
            state["round_count"],
            state["max_rounds"],
            _planner_guidance(state),
        )
        return {
            "pending_action": action.model_dump(mode="json"),
            "human_direction": None,
            "validation_errors": [],
            "trace": _trace(
                "plan_action",
                "action_planned",
                state,
                f"{action.tool_name}: {action.target_id}",
                {"reason": action.reason},
            ),
        }

    def validate_planned_action(state: CaseState) -> dict[str, object]:
        public_case = PublicCase.model_validate(state["public_case"])
        action = InvestigationAction.model_validate(state["pending_action"])
        result = validate_action(
            action,
            public_case=public_case,
            discovered_evidence_ids=_discovered_ids(state),
        )
        error = result.as_state_error()
        if error is None:
            return {
                "validation_errors": [],
                "trace": _trace(
                    "validate_action",
                    "action_accepted",
                    state,
                    result.message,
                ),
            }
        return {
            "invalid_action_count": state["invalid_action_count"] + 1,
            "pending_action": None,
            "validation_errors": [error],
            "trace": _trace(
                "validate_action",
                "action_rejected",
                state,
                result.message,
                {"code": result.code},
            ),
        }

    def execute_tool(state: CaseState) -> dict[str, object]:
        action = InvestigationAction.model_validate(state["pending_action"])
        tools = {tool.name: tool for tool in create_case_tools(store, state["case_id"])}
        tool = tools[action.tool_name]
        arguments = _tool_arguments(action)
        raw_observation = tool.invoke(arguments)
        observation = (
            raw_observation
            if isinstance(raw_observation, ToolObservation)
            else ToolObservation.model_validate(raw_observation)
        )
        next_round = state["round_count"] + 1
        return {
            "last_observation": observation.model_dump(mode="json"),
            "round_count": next_round,
            "trace": _trace(
                "execute_tool",
                "tool_completed",
                state,
                observation.summary,
                {
                    "tool_name": action.tool_name,
                    "target_id": action.target_id,
                    "topic": action.topic,
                    "evidence_ids": list(observation.evidence_ids),
                    "round_count": next_round,
                },
            ),
        }

    def record_evidence(state: CaseState) -> dict[str, object]:
        public_case = PublicCase.model_validate(state["public_case"])
        observation = ToolObservation.model_validate(state["last_observation"])
        existing_ids = _discovered_ids(state)
        evidence_by_id = {evidence.evidence_id: evidence for evidence in public_case.observations}
        newly_discovered = [
            evidence_by_id[evidence_id].model_dump(mode="json")
            for evidence_id in observation.evidence_ids
            if evidence_id not in existing_ids
        ]
        no_progress_count = 0 if newly_discovered else state["no_progress_count"] + 1
        message = (
            f"Added {len(newly_discovered)} evidence item(s) to the case board."
            if newly_discovered
            else "No new evidence was added."
        )
        return {
            "discovered_evidence": newly_discovered,
            "no_progress_count": no_progress_count,
            "trace": _trace(
                "record_evidence",
                "evidence_recorded",
                state,
                message,
                {
                    "evidence_ids": [item["evidence_id"] for item in newly_discovered],
                    "no_progress_count": no_progress_count,
                },
            ),
        }

    def assess_progress(state: CaseState) -> dict[str, object]:
        assessment = gateway.assess_progress(
            _object_dict(state["public_case"]),
            _object_tuple(state["discovered_evidence"]),
            _object_tuple(state["hypotheses"]),
            state["round_count"],
            state["max_rounds"],
        )
        hypotheses = [item.model_dump(mode="json") for item in assessment.hypotheses]
        return {
            "assessment": assessment.model_dump(mode="json"),
            "hypotheses": hypotheses,
            "trace": _trace(
                "assess_progress",
                "progress_assessed",
                state,
                assessment.summary,
                {
                    "ready_for_verdict": assessment.ready_for_verdict,
                    "next_leads": list(assessment.next_leads),
                },
            ),
        }

    def prepare_human_review(state: CaseState) -> dict[str, object]:
        return {
            "status": "awaiting_human",
            "trace": _trace(
                "prepare_human_review",
                "human_review_requested",
                state,
                "The graph paused for the player's choice of lead.",
            ),
        }

    def request_direction(state: CaseState) -> dict[str, object]:
        assessment = ProgressAssessment.model_validate(state["assessment"])
        response = interrupt(
            {
                "question": "Which lead should the detective prioritize next?",
                "suggested_leads": assessment.next_leads[:3],
                "discovered_evidence_ids": sorted(_discovered_ids(state)),
                "round_count": state["round_count"],
            }
        )
        direction = str(response).strip() or "Continue with the strongest remaining lead."
        return {
            "human_direction": direction,
            "human_review_completed": True,
            "no_progress_count": 0,
            "status": "investigating",
            "trace": _trace(
                "request_direction",
                "human_direction_received",
                state,
                direction,
            ),
        }

    def draft_verdict(state: CaseState) -> dict[str, object]:
        verdict = gateway.draft_verdict(
            _object_dict(state["public_case"]),
            _object_tuple(state["discovered_evidence"]),
            _object_tuple(state["hypotheses"]),
        )
        return {
            "proposed_verdict": verdict.model_dump(mode="json"),
            "validation_errors": [],
            "trace": _trace(
                "draft_verdict",
                "verdict_drafted",
                state,
                verdict.summary,
                {"culprit_id": verdict.culprit_id, "citations": verdict.citations},
            ),
        }

    def validate_proposed_verdict(state: CaseState) -> dict[str, object]:
        public_case = PublicCase.model_validate(state["public_case"])
        verdict = Verdict.model_validate(state["proposed_verdict"])
        result = validate_verdict(
            verdict,
            public_case=public_case,
            discovered_evidence_ids=_discovered_ids(state),
            solution_key=store.load_solution(state["case_id"]),
        )
        error = result.as_state_error()
        if error is None:
            return {
                "status": "resolved",
                "validation_errors": [],
                "final_report": f"{verdict.summary}\n\n{verdict.case_theory}",
                "trace": _trace(
                    "validate_verdict",
                    "verdict_accepted",
                    state,
                    result.message,
                ),
            }
        return {
            "validation_errors": [error],
            "trace": _trace(
                "validate_verdict",
                "verdict_rejected",
                state,
                result.message,
                {"code": result.code},
            ),
        }

    def repair_verdict(state: CaseState) -> dict[str, object]:
        verdict = gateway.repair_verdict(
            _object_dict(state["public_case"]),
            _object_tuple(state["discovered_evidence"]),
            _object_dict(state["proposed_verdict"] or {}),
            tuple(dict(error) for error in state["validation_errors"]),
        )
        revision_count = state["verdict_revision_count"] + 1
        return {
            "proposed_verdict": verdict.model_dump(mode="json"),
            "verdict_revision_count": revision_count,
            "trace": _trace(
                "repair_verdict",
                "verdict_repaired",
                state,
                verdict.summary,
                {"revision_count": revision_count},
            ),
        }

    def inconclusive(state: CaseState) -> dict[str, object]:
        return {
            "status": "inconclusive",
            "final_report": (
                "The investigation reached its configured safety limit without a fully "
                "supported verdict."
            ),
            "trace": _trace(
                "inconclusive",
                "investigation_stopped",
                state,
                "A configured retry or round limit was reached.",
            ),
        }

    builder.add_node("load_case", load_case)
    builder.add_node("plan_action", plan_action)
    builder.add_node("validate_action", validate_planned_action)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("record_evidence", record_evidence)
    builder.add_node("assess_progress", assess_progress)
    builder.add_node("prepare_human_review", prepare_human_review)
    builder.add_node("request_direction", request_direction)
    builder.add_node("draft_verdict", draft_verdict)
    builder.add_node("validate_verdict", validate_proposed_verdict)
    builder.add_node("repair_verdict", repair_verdict)
    builder.add_node("inconclusive", inconclusive)

    builder.add_edge(START, "load_case")
    builder.add_edge("load_case", "plan_action")
    builder.add_edge("plan_action", "validate_action")
    builder.add_conditional_edges(
        "validate_action",
        _route_after_action_validation,
        {
            "execute_tool": "execute_tool",
            "retry": "plan_action",
            "inconclusive": "inconclusive",
        },
    )
    builder.add_edge("execute_tool", "record_evidence")
    builder.add_edge("record_evidence", "assess_progress")
    builder.add_conditional_edges(
        "assess_progress",
        lambda state: _route_after_assessment(state, store),
        {
            "draft_verdict": "draft_verdict",
            "human_review": "prepare_human_review",
            "investigate": "plan_action",
            "inconclusive": "inconclusive",
        },
    )
    builder.add_edge("prepare_human_review", "request_direction")
    builder.add_edge("request_direction", "plan_action")
    builder.add_edge("draft_verdict", "validate_verdict")
    builder.add_conditional_edges(
        "validate_verdict",
        _route_after_verdict_validation,
        {
            "end": END,
            "repair": "repair_verdict",
            "inconclusive": "inconclusive",
        },
    )
    builder.add_edge("repair_verdict", "validate_verdict")
    builder.add_edge("inconclusive", END)

    return builder.compile(checkpointer=checkpointer)


def _route_after_action_validation(state: CaseState) -> str:
    if not state["validation_errors"]:
        return "execute_tool"
    if state["invalid_action_count"] >= state["max_invalid_actions"]:
        return "inconclusive"
    return "retry"


def _route_after_assessment(state: CaseState, store: CaseStore) -> str:
    evidence_ids = _discovered_ids(state)
    solution = store.load_solution(state["case_id"])
    complete_path = any(set(path) <= evidence_ids for path in solution.acceptable_evidence_sets)
    if complete_path:
        return "draft_verdict"
    if state["round_count"] >= state["max_rounds"]:
        return "inconclusive"
    human_review_due = state["round_count"] >= HUMAN_REVIEW_AFTER_ROUNDS
    stalled = state["no_progress_count"] >= NO_PROGRESS_INTERRUPT_THRESHOLD
    if (
        state["human_review_enabled"]
        and not state["human_review_completed"]
        and (human_review_due or stalled)
    ):
        return "human_review"
    return "investigate"


def _route_after_verdict_validation(state: CaseState) -> str:
    if not state["validation_errors"]:
        return "end"
    if state["verdict_revision_count"] >= state["max_verdict_revisions"]:
        return "inconclusive"
    return "repair"


def _tool_arguments(action: InvestigationAction) -> dict[str, str]:
    if action.tool_name == "inspect_location":
        return {"location_id": action.target_id}
    if action.tool_name == "compare_timeline":
        return {"anchor_id": action.target_id}
    if action.topic is None:
        raise ValueError("An interview action requires a topic.")
    return {"suspect_id": action.target_id, "topic": action.topic}


def _discovered_ids(state: CaseState) -> set[str]:
    return {record["evidence_id"] for record in state["discovered_evidence"]}


def _object_dict(value: Mapping[str, object]) -> dict[str, object]:
    return dict(value)


def _object_tuple(values: list[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    return tuple(dict(value) for value in values)


def _planner_guidance(state: CaseState) -> str | None:
    guidance: list[str] = []
    if state["human_direction"]:
        guidance.append(f"Player direction: {state['human_direction']}")

    completed_actions: list[str] = []
    for event in state["trace"]:
        if event["event"] != "tool_completed":
            continue
        data = event.get("data", {})
        tool_name = data.get("tool_name")
        target_id = data.get("target_id")
        topic = data.get("topic")
        if isinstance(tool_name, str) and isinstance(target_id, str):
            action = f"{tool_name}({target_id}"
            if isinstance(topic, str):
                action = f"{action}, {topic}"
            completed_actions.append(f"{action})")
    if completed_actions:
        guidance.append(
            "Already completed; do not repeat: " + ", ".join(dict.fromkeys(completed_actions))
        )

    if state["validation_errors"]:
        latest_action = next(
            (
                event["message"]
                for event in reversed(state["trace"])
                if event["event"] == "action_planned"
            ),
            "the previous action",
        )
        feedback = "; ".join(error["message"] for error in state["validation_errors"])
        guidance.append(f"Rejected action {latest_action}: {feedback} Choose a different action.")
    return "\n".join(guidance) or None


def _trace(
    node: str,
    event: str,
    state: CaseState,
    message: str,
    data: Mapping[str, object] | None = None,
) -> list[TraceEvent]:
    payload = cast(JsonObject, dict(data or {}))
    return [
        TraceEvent(
            node=node,
            event=event,
            round_count=state["round_count"],
            message=message,
            data=payload,
        )
    ]
