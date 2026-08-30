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
    GameDebrief,
    GameResult,
    PlayerAccusation,
)


def _message_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_gateway_uses_native_structured_output_for_all_calls() -> None:
    route = GameActionRoute(
        action_id="inspect_security_screening",
        needs_clarification=False,
        player_message="The screening desk can answer that.",
    )
    debrief = GameDebrief(
        headline="The case clicks shut",
        summary="The discovered records support the accusation.",
        evidence_analysis=["The weight record closes the transport gap."],
        closing_line="The gallery lights come back on.",
    )
    route_model = MagicMock()
    route_model.invoke.return_value = route.model_dump(mode="json")
    accusation_model = MagicMock()
    accusation_model.invoke.return_value = AccusationMatch(
        motive_match=False,
        method_match=True,
    ).model_dump(mode="json")
    debrief_model = MagicMock()
    debrief_model.invoke.return_value = debrief.model_dump(mode="json")
    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.side_effect = [route_model, accusation_model, debrief_model]
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
            ("inspect_display_case",),
            ({"evidence_id": "E01", "title": "Weight change", "source_reference": "hidden"},),
        )
        == route
    )
    match = gateway.assess_accusation(
        accusation,
        ("Rowan moved the sculpture into his hard equipment case.",),
    )
    assert match == AccusationMatch(motive_match=False, method_match=True)
    assert (
        gateway.write_debrief(
            {
                "suspects": [
                    {"suspect_id": "rowan_pike", "name": "Rowan Pike", "role": "Photographer"}
                ]
            },
            ({"evidence_id": "E01", "title": "Weight change", "source_reference": "hidden"},),
            accusation,
            GameResult(
                tier="solved",
                accused_suspect_id="rowan_pike",
                culprit_correct=True,
                motive_match=False,
                method_match=True,
            ),
        )
        == debrief
    )

    assert model.with_structured_output.call_args_list == [
        call(GameActionRoute, method="json_schema"),
        call(AccusationMatch, method="json_schema"),
        call(GameDebrief, method="json_schema"),
    ]
    assert "hidden" not in _message_text(route_model.invoke.call_args.args[0])
    assert "hidden" not in _message_text(debrief_model.invoke.call_args.args[0])
    accusation_prompt = _message_text(accusation_model.invoke.call_args.args[0])
    assert "hard equipment case" in accusation_prompt
    assert "He hid it" in accusation_prompt
