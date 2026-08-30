"""Validated domain schemas for cases, investigation actions, and verdicts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

ToolName = Literal["inspect_location", "interview_suspect", "compare_timeline"]
CaseStatus = Literal["investigating", "awaiting_human", "resolved", "inconclusive"]
PositiveFloat = Annotated[float, Field(gt=0)]


def _coerce_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("timestamp must be valid ISO 8601") from exc
    else:
        raise ValueError("timestamp must be an ISO 8601 string or datetime")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


UtcDatetime = Annotated[datetime, BeforeValidator(_coerce_utc)]


class FrozenModel(BaseModel):
    """Base configuration for immutable boundary schemas."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class InvestigationAction(FrozenModel):
    """A single structured action selected by the investigator model."""

    tool_name: ToolName
    target_id: str = Field(min_length=1)
    topic: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_topic(self) -> InvestigationAction:
        if self.tool_name == "interview_suspect" and self.topic is None:
            raise ValueError("interview_suspect requires a topic")
        if self.tool_name != "interview_suspect" and self.topic is not None:
            raise ValueError(f"{self.tool_name} does not accept a topic")
        return self


class Hypothesis(FrozenModel):
    """A bounded, evidence-cited theory about one suspect."""

    suspect_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    confidence: int = Field(ge=0, le=100)
    evidence_ids: list[str] = Field(default_factory=list)


class ProgressAssessment(FrozenModel):
    """A structured assessment used to route the investigation graph."""

    summary: str = Field(min_length=1)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    ready_for_verdict: bool
    next_leads: list[str] = Field(default_factory=list)


class Verdict(FrozenModel):
    """The model-authored case conclusion before deterministic validation."""

    culprit_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    case_theory: str = Field(min_length=1)
    confidence: int = Field(ge=0, le=100)
    citations: list[str] = Field(min_length=1)


class ToolObservation(FrozenModel):
    """A public-only result returned by every deterministic case tool."""

    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    no_new_evidence: bool


class Artwork(FrozenModel):
    """Public physical details for the missing artwork."""

    name: str = Field(min_length=1)
    mass_kg: float = Field(gt=0)
    dimensions_cm: tuple[PositiveFloat, PositiveFloat, PositiveFloat]


class Incident(FrozenModel):
    """Public timing information for the apparent theft."""

    discovered_at: UtcDatetime
    blackout_started_at: UtcDatetime
    blackout_ended_at: UtcDatetime

    @model_validator(mode="after")
    def validate_chronology(self) -> Incident:
        if self.blackout_ended_at <= self.blackout_started_at:
            raise ValueError("blackout must end after it starts")
        if self.discovered_at < self.blackout_ended_at:
            raise ValueError("discovery cannot predate the end of the blackout")
        return self


class Suspect(FrozenModel):
    """A suspect profile safe to show to the model and player."""

    suspect_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    public_profile: str = Field(min_length=1)


class Location(FrozenModel):
    """An inspectable location in the museum."""

    location_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class InterviewOption(FrozenModel):
    """The allowed deterministic interview topics for one suspect."""

    suspect_id: str = Field(min_length=1)
    topic_ids: list[str] = Field(min_length=1)


class AvailableActions(FrozenModel):
    """The complete public menu of valid tool targets."""

    location_ids: list[str] = Field(min_length=1)
    interview_options: list[InterviewOption] = Field(min_length=1)
    timeline_anchor_ids: list[str] = Field(min_length=1)


class EvidenceRecord(FrozenModel):
    """A discoverable, source-attributed public observation."""

    evidence_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    occurred_at: UtcDatetime


class TimelineEvent(FrozenModel):
    """A source-backed event available to the timeline comparison tool."""

    event_id: str = Field(min_length=1)
    starts_at: UtcDatetime
    ends_at: UtcDatetime | None = None
    subject_ids: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chronology(self) -> TimelineEvent:
        if self.ends_at is not None and self.ends_at < self.starts_at:
            raise ValueError("timeline event cannot end before it starts")
        return self


class RevealRoute(FrozenModel):
    """A deterministic mapping from a valid tool call to public evidence."""

    tool_name: ToolName
    target_id: str = Field(min_length=1)
    topic: str | None = Field(default=None, min_length=1)
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topic(self) -> RevealRoute:
        if self.tool_name == "interview_suspect" and self.topic is None:
            raise ValueError("interview route requires a topic")
        if self.tool_name != "interview_suspect" and self.topic is not None:
            raise ValueError(f"{self.tool_name} route does not accept a topic")
        return self


class PublicCase(FrozenModel):
    """All case information that public tools may inspect or return."""

    schema_version: Literal[1]
    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    display_timezone: str = Field(min_length=1)
    brief: str = Field(min_length=1)
    artwork: Artwork
    incident: Incident
    suspects: list[Suspect] = Field(min_length=1)
    locations: list[Location] = Field(min_length=1)
    available_actions: AvailableActions
    observations: list[EvidenceRecord] = Field(min_length=1)
    timeline_events: list[TimelineEvent] = Field(min_length=1)
    reveal_routes: list[RevealRoute] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> PublicCase:
        suspect_ids = [suspect.suspect_id for suspect in self.suspects]
        location_ids = [location.location_id for location in self.locations]
        evidence_ids = [record.evidence_id for record in self.observations]
        timeline_event_ids = [event.event_id for event in self.timeline_events]

        _require_unique("suspect IDs", suspect_ids)
        _require_unique("location IDs", location_ids)
        _require_unique("evidence IDs", evidence_ids)
        _require_unique("timeline event IDs", timeline_event_ids)
        _require_unique(
            "interview suspect IDs",
            [option.suspect_id for option in self.available_actions.interview_options],
        )

        if set(self.available_actions.location_ids) != set(location_ids):
            raise ValueError("available location IDs must match declared locations")
        if set(self.available_actions.timeline_anchor_ids) - set(timeline_event_ids):
            raise ValueError("timeline anchors must reference declared timeline events")

        interview_topics = {
            option.suspect_id: set(option.topic_ids)
            for option in self.available_actions.interview_options
        }
        if set(interview_topics) != set(suspect_ids):
            raise ValueError("every suspect must have interview options")

        evidence_id_set = set(evidence_ids)
        route_keys: list[tuple[ToolName, str, str | None]] = []
        reachable_evidence: set[str] = set()
        for route in self.reveal_routes:
            route_keys.append((route.tool_name, route.target_id, route.topic))
            unknown_evidence = set(route.evidence_ids) - evidence_id_set
            if unknown_evidence:
                raise ValueError(f"route references unknown evidence: {sorted(unknown_evidence)}")
            reachable_evidence.update(route.evidence_ids)

            if route.tool_name == "inspect_location":
                if route.target_id not in self.available_actions.location_ids:
                    raise ValueError(f"unknown inspection location: {route.target_id}")
            elif route.tool_name == "interview_suspect":
                if route.topic not in interview_topics.get(route.target_id, set()):
                    raise ValueError(
                        f"unknown interview target/topic: {route.target_id}/{route.topic}"
                    )
            elif route.target_id not in self.available_actions.timeline_anchor_ids:
                raise ValueError(f"unknown timeline anchor: {route.target_id}")

        _require_unique("tool routes", route_keys)
        if reachable_evidence != evidence_id_set:
            unreachable = sorted(evidence_id_set - reachable_evidence)
            raise ValueError(f"all evidence must be reachable; unreachable: {unreachable}")
        return self


class ClaimRule(FrozenModel):
    """Private allowed citations for one canonical claim."""

    claim_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    accepted_evidence_sets: list[list[str]] = Field(min_length=1)


class ContradictionRule(FrozenModel):
    """Private relationship between a statement and rebutting evidence."""

    statement_evidence_id: str = Field(min_length=1)
    rebuttal_evidence_ids: list[str] = Field(min_length=1)


class ExonerationRule(FrozenModel):
    """Private evidence supporting elimination of one innocent suspect."""

    suspect_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class SolutionKey(FrozenModel):
    """Private answer key loaded only by deterministic verdict validation."""

    schema_version: Literal[1]
    case_id: str = Field(min_length=1)
    culprit_id: str = Field(min_length=1)
    canonical_sequence: list[str] = Field(min_length=1)
    acceptable_evidence_sets: list[list[str]] = Field(min_length=1)
    claim_rules: list[ClaimRule] = Field(min_length=1)
    contradiction_pairs: list[ContradictionRule] = Field(default_factory=list)
    exoneration_rules: list[ExonerationRule] = Field(min_length=1)
    hints: list[str] = Field(min_length=1)


def _require_unique(label: str, values: list[object]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
