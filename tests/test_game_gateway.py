"""Offline tests for the LangChain player-game gateway."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, call

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from case_closed.game_catalog import all_game_actions
from case_closed.game_gateway import PlayerGameGateway
from case_closed.game_schemas import (
    AccusationMatch,
    GameActionRoute,
    PlayerAccusation,
)


def _message_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_gateway_uses_native_structured_output_for_model_calls() -> None:
    route = GameActionRoute(
        action_id="crosscheck_rowans_case",
        needs_clarification=False,
        clarification_question=None,
        suggested_action_ids=[],
        player_message="The screening desk can answer that.",
    )
    route_model = MagicMock()
    route_model.invoke.return_value = route.model_dump(mode="json")
    accusation_model = MagicMock()
    accusation_model.invoke.return_value = AccusationMatch(
        motive_match=False,
        method_match=True,
    ).model_dump(mode="json")
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.side_effect = [route_model, accusation_model]
    gateway = PlayerGameGateway(cast(BaseChatModel, model))
    accusation = PlayerAccusation(
        suspect_id="rowan_pike",
        motive="He wanted to take the sculpture.",
        method="He hid it in the equipment case before the blackout.",
    )

    assert (
        gateway.route_free_form(
            "Check the equipment weights.",
            all_game_actions(),
            ("ask_theo_who_heard_warning",),
            ({"evidence_id": "E01", "title": "Weight change", "source_reference": "hidden"},),
        )
        == route
    )
    match = gateway.assess_accusation(
        accusation,
        ("Rowan moved the sculpture into his hard equipment case.",),
    )
    assert match == AccusationMatch(motive_match=False, method_match=True)

    assert model.with_structured_output.call_args_list == [
        call(GameActionRoute, method="json_schema"),
        call(AccusationMatch, method="json_schema"),
    ]
    route_prompt = _message_text(route_model.invoke.call_args.args[0])
    assert "clarification_question=null" in route_prompt
    assert "action_id=null" in route_prompt
    assert "hidden" not in route_prompt
    accusation_prompt = _message_text(accusation_model.invoke.call_args.args[0])
    assert "hard equipment case" in accusation_prompt
    assert "He hid it" in accusation_prompt
