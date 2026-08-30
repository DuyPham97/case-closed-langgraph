"""LangChain gateway for structured Claude model calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from case_closed.config import AppConfig
from case_closed.prompts import (
    assess_progress_messages,
    draft_verdict_messages,
    plan_action_messages,
    repair_verdict_messages,
)
from case_closed.schemas import InvestigationAction, ProgressAssessment, Verdict

ModelFactory = Callable[..., BaseChatModel]


def create_anthropic_model(
    config: AppConfig,
    *,
    model_factory: ModelFactory = ChatAnthropic,
) -> BaseChatModel:
    """Create the configured Anthropic model without retaining configuration secrets."""
    options: dict[str, object] = {
        "model": config.anthropic_model,
        "api_key": config.anthropic_api_key,
        "max_tokens": config.max_tokens,
        "max_retries": config.max_retries,
    }
    if config.anthropic_workspace_id is not None:
        options["default_headers"] = {
            "anthropic-workspace-id": config.anthropic_workspace_id.get_secret_value()
        }
    return model_factory(**options)


class ClaudeGateway:
    """Invoke injected Claude-compatible models through typed structured-output boundaries."""

    def __init__(self, model: BaseChatModel) -> None:
        self._action_model = model.with_structured_output(
            InvestigationAction,
            method="json_schema",
        )
        self._progress_model = model.with_structured_output(
            ProgressAssessment,
            method="json_schema",
        )
        self._verdict_model = model.with_structured_output(
            Verdict,
            method="json_schema",
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        model_factory: ModelFactory = ChatAnthropic,
    ) -> Self:
        """Create a gateway backed by a configured ChatAnthropic model."""
        return cls(create_anthropic_model(config, model_factory=model_factory))

    def plan_action(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        hypotheses: tuple[dict[str, object], ...],
        round_count: int,
        max_rounds: int,
        human_direction: str | None,
    ) -> InvestigationAction:
        """Select one valid next action from public investigation context."""
        result = self._action_model.invoke(
            plan_action_messages(
                public_case,
                discovered_evidence,
                hypotheses,
                round_count,
                max_rounds,
                human_direction,
            )
        )
        return _validate_output(result, InvestigationAction)

    def assess_progress(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        previous_hypotheses: tuple[dict[str, object], ...],
        round_count: int,
        max_rounds: int,
    ) -> ProgressAssessment:
        """Assess current evidence and produce updated hypotheses for graph routing."""
        result = self._progress_model.invoke(
            assess_progress_messages(
                public_case,
                discovered_evidence,
                previous_hypotheses,
                round_count,
                max_rounds,
            )
        )
        return _validate_output(result, ProgressAssessment)

    def draft_verdict(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        hypotheses: tuple[dict[str, object], ...],
    ) -> Verdict:
        """Draft a typed verdict using only public, discovered evidence."""
        result = self._verdict_model.invoke(
            draft_verdict_messages(public_case, discovered_evidence, hypotheses)
        )
        return _validate_output(result, Verdict)

    def repair_verdict(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        previous_verdict: dict[str, object],
        validation_errors: tuple[dict[str, object], ...],
    ) -> Verdict:
        """Repair a verdict according to deterministic validation errors."""
        result = self._verdict_model.invoke(
            repair_verdict_messages(
                public_case,
                discovered_evidence,
                previous_verdict,
                validation_errors,
            )
        )
        return _validate_output(result, Verdict)


def _validate_output[OutputModelT: BaseModel](
    value: object,
    schema: type[OutputModelT],
) -> OutputModelT:
    if isinstance(value, schema):
        return value
    return schema.model_validate(value)
