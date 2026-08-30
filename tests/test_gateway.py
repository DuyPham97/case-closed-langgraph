"""Tests for the structured Claude gateway."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, call

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import SecretStr, ValidationError

from case_closed.config import DEFAULT_ANTHROPIC_MODEL, AppConfig
from case_closed.gateway import ClaudeGateway, create_anthropic_model
from case_closed.schemas import Hypothesis, InvestigationAction, ProgressAssessment, Verdict

DUMMY_API_KEY = "anthropic-test-api-key"


def _model_with_runnables(
    action_response: object,
    progress_response: object,
    *verdict_responses: object,
) -> tuple[BaseChatModel, MagicMock, MagicMock, MagicMock]:
    action_model = MagicMock()
    action_model.invoke.return_value = action_response
    progress_model = MagicMock()
    progress_model.invoke.return_value = progress_response
    verdict_model = MagicMock()
    verdict_model.invoke.side_effect = verdict_responses

    model = MagicMock(spec=BaseChatModel)
    model.with_structured_output.side_effect = [
        action_model,
        progress_model,
        verdict_model,
    ]
    return cast(BaseChatModel, model), action_model, progress_model, verdict_model


def _public_case() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "midnight-museum",
        "title": "The Midnight Museum Heist",
        "brief": "A painting vanished during a blackout.",
        "suspects": [{"suspect_id": "suspect_a", "name": "Avery"}],
        "locations": [{"location_id": "gallery", "name": "Gallery"}],
        "available_actions": {"location_ids": ["gallery"]},
        "observations": [{"text": "UNDISCOVERED_EVIDENCE_NEEDLE"}],
        "timeline_events": [{"summary": "UNDISCOVERED_TIMELINE_NEEDLE"}],
        "reveal_routes": [{"summary": "PRIVATE_ROUTE_NEEDLE"}],
        "solution_key": {"culprit_id": "PRIVATE_SOLUTION_NEEDLE"},
    }


def _message_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_gateway_uses_native_structured_output_and_validates_results() -> None:
    action = InvestigationAction(
        tool_name="inspect_location",
        target_id="gallery",
        reason="Inspect the scene before interviewing suspects.",
    )
    hypothesis = Hypothesis(
        suspect_id="suspect_a",
        summary="Avery had access.",
        confidence=35,
        evidence_ids=["evidence_1"],
    )
    assessment = ProgressAssessment(
        summary="One plausible lead remains.",
        hypotheses=[hypothesis],
        ready_for_verdict=False,
        next_leads=["Inspect the gallery."],
    )
    verdict = Verdict(
        culprit_id="suspect_a",
        summary="Avery committed the theft.",
        case_theory="The access record and physical trace align.",
        confidence=80,
        citations=["evidence_1"],
    )
    repaired = verdict.model_copy(update={"confidence": 75})
    model, action_model, progress_model, verdict_model = _model_with_runnables(
        action.model_dump(mode="json"),
        assessment.model_dump(mode="json"),
        verdict.model_dump(mode="json"),
        repaired.model_dump(mode="json"),
    )

    gateway = ClaudeGateway(model)

    assert gateway.plan_action(_public_case(), (), (), 0, 4, None) == action
    assert (
        gateway.assess_progress(
            _public_case(),
            ({"evidence_id": "evidence_1", "text": "A trace."},),
            (),
            1,
            4,
        )
        == assessment
    )
    assert (
        gateway.draft_verdict(
            _public_case(),
            ({"evidence_id": "evidence_1", "text": "A trace."},),
            (hypothesis.model_dump(mode="json"),),
        )
        == verdict
    )
    assert (
        gateway.repair_verdict(
            _public_case(),
            ({"evidence_id": "evidence_1", "text": "A trace."},),
            verdict.model_dump(mode="json"),
            (
                {
                    "code": "overstated_confidence",
                    "message": "confidence is overstated",
                },
            ),
        )
        == repaired
    )

    structured_calls = cast(MagicMock, model).with_structured_output.call_args_list
    assert structured_calls == [
        call(InvestigationAction, method="json_schema"),
        call(ProgressAssessment, method="json_schema"),
        call(Verdict, method="json_schema"),
    ]
    assert action_model.invoke.call_count == 1
    assert progress_model.invoke.call_count == 1
    assert verdict_model.invoke.call_count == 2


def test_gateway_prompts_exclude_undiscovered_and_private_case_data() -> None:
    action = InvestigationAction(
        tool_name="inspect_location",
        target_id="gallery",
        reason="Inspect the scene.",
    )
    assessment = ProgressAssessment(
        summary="Keep investigating.",
        hypotheses=[],
        ready_for_verdict=False,
        next_leads=["gallery"],
    )
    verdict = Verdict(
        culprit_id="suspect_a",
        summary="Supported conclusion.",
        case_theory="Evidence supports this suspect.",
        confidence=60,
        citations=["evidence_1"],
    )
    model, action_model, _, verdict_model = _model_with_runnables(action, assessment, verdict)
    gateway = ClaudeGateway(model)

    gateway.plan_action(_public_case(), (), (), 0, 4, "Focus on physical access.")
    gateway.repair_verdict(
        _public_case(),
        ({"evidence_id": "evidence_1", "text": "A trace."},),
        verdict.model_dump(mode="json"),
        (
            {
                "code": "unsupported_citation",
                "message": "Use only discovered citations.",
                "solution_key": "PRIVATE_VALIDATOR_NEEDLE",
            },
        ),
    )

    messages = action_model.invoke.call_args.args[0]
    prompt_text = _message_text(messages)
    assert "The Midnight Museum Heist" in prompt_text
    assert "Focus on physical access." in prompt_text
    assert "UNDISCOVERED_EVIDENCE_NEEDLE" not in prompt_text
    assert "UNDISCOVERED_TIMELINE_NEEDLE" not in prompt_text
    assert "PRIVATE_ROUTE_NEEDLE" not in prompt_text
    assert "PRIVATE_SOLUTION_NEEDLE" not in prompt_text
    repair_messages = verdict_model.invoke.call_args.args[0]
    repair_prompt_text = _message_text(repair_messages)
    assert "unsupported_citation" in repair_prompt_text
    assert "Use only discovered citations." in repair_prompt_text
    assert "PRIVATE_VALIDATOR_NEEDLE" not in repair_prompt_text


def test_gateway_rejects_malformed_structured_output() -> None:
    assessment = ProgressAssessment(
        summary="Keep investigating.",
        hypotheses=[],
        ready_for_verdict=False,
        next_leads=[],
    )
    verdict = Verdict(
        culprit_id="suspect_a",
        summary="Supported conclusion.",
        case_theory="Evidence supports this suspect.",
        confidence=60,
        citations=["evidence_1"],
    )
    model, _, _, _ = _model_with_runnables({"tool_name": "made_up_tool"}, assessment, verdict)
    gateway = ClaudeGateway(model)

    with pytest.raises(ValidationError):
        gateway.plan_action(_public_case(), (), (), 0, 4, None)


@pytest.mark.parametrize(
    ("workspace_id", "expected_headers"),
    [
        (None, None),
        ("wrkspc_test_identifier", {"anthropic-workspace-id": "wrkspc_test_identifier"}),
    ],
)
def test_create_anthropic_model_passes_safe_configuration(
    workspace_id: str | None,
    expected_headers: dict[str, str] | None,
) -> None:
    captured: dict[str, object] = {}
    sentinel = cast(BaseChatModel, object())

    def fake_factory(**kwargs: object) -> BaseChatModel:
        captured.update(kwargs)
        return sentinel

    config = AppConfig(
        anthropic_api_key=DUMMY_API_KEY,
        anthropic_workspace_id=workspace_id,
    )

    assert create_anthropic_model(config, model_factory=fake_factory) is sentinel
    assert captured["model"] == DEFAULT_ANTHROPIC_MODEL
    assert isinstance(captured["api_key"], SecretStr)
    assert captured["api_key"].get_secret_value() == DUMMY_API_KEY
    assert captured["max_tokens"] == 2_048
    assert captured["max_retries"] == 2
    if expected_headers is None:
        assert "default_headers" not in captured
    else:
        assert captured["default_headers"] == expected_headers
