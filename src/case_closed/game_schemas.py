"""Typed boundaries for the player-driven detective game."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from case_closed.schemas import FrozenModel, ToolName

GameActionCategory = Literal["location", "interview", "records"]
GameResultTier = Literal["solved", "partial", "failed"]


class GameAction(FrozenModel):
    """A player-facing choice backed by one deterministic case tool route."""

    action_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: GameActionCategory
    tool_name: ToolName
    target_id: str = Field(min_length=1)
    topic: str | None = Field(default=None, min_length=1)
    is_visual: bool = False
    image_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_route(self) -> GameAction:
        """Require interview topics and keep visual choices location-based."""
        if self.tool_name == "interview_suspect" and self.topic is None:
            raise ValueError("interview actions require a topic")
        if self.tool_name != "interview_suspect" and self.topic is not None:
            raise ValueError(f"{self.tool_name} actions do not accept a topic")
        if self.is_visual and self.category != "location":
            raise ValueError("visual actions must be location choices")
        return self


class GameActionRoute(FrozenModel):
    """Claude's bounded interpretation of a player's free-form investigation."""

    action_id: str | None = Field(min_length=1)
    needs_clarification: bool
    clarification_question: str | None = Field(min_length=1)
    suggested_action_ids: list[str] = Field(max_length=3)
    player_message: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> GameActionRoute:
        """Keep successful routes and clarification responses mutually exclusive."""
        if self.needs_clarification:
            if self.clarification_question is None:
                raise ValueError("clarification routes need a question")
        elif self.action_id is None:
            raise ValueError("successful routes need an action_id")
        return self


class PlayerAccusation(FrozenModel):
    """The player's final suspect, motive, and method theory."""

    suspect_id: str = Field(min_length=1)
    motive: str = Field(min_length=1, max_length=1_000)
    method: str = Field(min_length=1, max_length=1_000)


class AccusationMatch(FrozenModel):
    """Strict semantic matches against the transient canonical story."""

    motive_match: bool
    method_match: bool


class GameResult(FrozenModel):
    """A public score derived transiently from the private solution key."""

    tier: GameResultTier
    accused_suspect_id: str = Field(min_length=1)
    culprit_correct: bool
    motive_match: bool
    method_match: bool


class GameDebrief(FrozenModel):
    """A grounded, player-facing closing analysis."""

    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_analysis: list[str] = Field(min_length=1, max_length=5)
    closing_line: str = Field(min_length=1)
