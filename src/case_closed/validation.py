"""Deterministic public-safe validation for investigation actions and verdicts."""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from case_closed.schemas import InvestigationAction, PublicCase, SolutionKey, Verdict
from case_closed.state import PublicValidationError

type ValidationCode = Literal[
    "valid",
    "unavailable_action",
    "exhausted_action",
    "unknown_suspect",
    "duplicate_citation",
    "undiscovered_citation",
    "unsupported_verdict",
]

_VALID_MESSAGE = "Validation passed."
_UNAVAILABLE_ACTION_MESSAGE = "That investigation action is not available in this case."
_EXHAUSTED_ACTION_MESSAGE = "That lead has already been exhausted; choose another action."
_UNKNOWN_SUSPECT_MESSAGE = "The verdict must name a suspect from the public case file."
_DUPLICATE_CITATION_MESSAGE = "Each evidence citation may appear only once."
_UNDISCOVERED_CITATION_MESSAGE = (
    "Every cited evidence item must be discovered before it can support a verdict."
)
_UNSUPPORTED_VERDICT_MESSAGE = (
    "The submitted verdict is not yet supported by the evidence on the case board."
)
_CASE_CONFIGURATION_MESSAGE = "Verdict validation could not run for this case."


class ValidationResult(BaseModel):
    """A redacted validation outcome safe to expose to the graph or model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool
    code: ValidationCode
    message: str = Field(min_length=1)

    def as_state_error(self) -> PublicValidationError | None:
        """Return the public error payload used by CaseState, if validation failed."""

        if self.is_valid:
            return None
        return PublicValidationError(code=self.code, message=self.message)


def validate_action(
    action: InvestigationAction,
    *,
    public_case: PublicCase,
    discovered_evidence_ids: Collection[str],
) -> ValidationResult:
    """Validate a structured action against public routes and prior discoveries."""

    discovered = _validate_discovered_ids(discovered_evidence_ids, public_case)
    matching_route = next(
        (
            route
            for route in public_case.reveal_routes
            if route.tool_name == action.tool_name
            and route.target_id == action.target_id
            and route.topic == action.topic
        ),
        None,
    )
    if matching_route is None:
        return _invalid("unavailable_action", _UNAVAILABLE_ACTION_MESSAGE)

    route_evidence = set(matching_route.evidence_ids)
    if route_evidence and route_evidence <= discovered:
        return _invalid("exhausted_action", _EXHAUSTED_ACTION_MESSAGE)
    return _valid()


def validate_verdict(
    verdict: Verdict,
    *,
    public_case: PublicCase,
    discovered_evidence_ids: Collection[str],
    solution_key: SolutionKey,
) -> ValidationResult:
    """Validate a verdict without returning any private answer-key details."""

    _validate_solution_case(public_case, solution_key)
    discovered = _validate_discovered_ids(discovered_evidence_ids, public_case)
    known_suspects = {suspect.suspect_id for suspect in public_case.suspects}
    if verdict.culprit_id not in known_suspects:
        return _invalid("unknown_suspect", _UNKNOWN_SUSPECT_MESSAGE)

    citations = set(verdict.citations)
    if len(citations) != len(verdict.citations):
        return _invalid("duplicate_citation", _DUPLICATE_CITATION_MESSAGE)
    if not citations <= discovered:
        return _invalid("undiscovered_citation", _UNDISCOVERED_CITATION_MESSAGE)

    supported = verdict.culprit_id == solution_key.culprit_id
    supported = supported and _matches_acceptable_evidence_set(
        citations,
        solution_key.acceptable_evidence_sets,
    )
    supported = supported and _satisfies_cited_contradictions(citations, solution_key)
    if not supported:
        return _invalid("unsupported_verdict", _UNSUPPORTED_VERDICT_MESSAGE)
    return _valid()


def _validate_discovered_ids(
    discovered_evidence_ids: Collection[str],
    public_case: PublicCase,
) -> set[str]:
    if any(not isinstance(item, str) or not item.strip() for item in discovered_evidence_ids):
        raise ValueError("Discovered evidence IDs must be non-empty strings.")
    discovered = set(discovered_evidence_ids)
    known = {record.evidence_id for record in public_case.observations}
    if not discovered <= known:
        raise ValueError("Discovered evidence does not belong to the public case file.")
    return discovered


def _validate_solution_case(public_case: PublicCase, solution_key: SolutionKey) -> None:
    if public_case.case_id != solution_key.case_id:
        raise ValueError(_CASE_CONFIGURATION_MESSAGE)


def _matches_acceptable_evidence_set(
    citations: set[str],
    acceptable_sets: Collection[Collection[str]],
) -> bool:
    return any(
        bool(required := set(option)) and required <= citations for option in acceptable_sets
    )


def _satisfies_cited_contradictions(citations: set[str], solution_key: SolutionKey) -> bool:
    return all(
        rule.statement_evidence_id not in citations
        or bool(set(rule.rebuttal_evidence_ids) & citations)
        for rule in solution_key.contradiction_pairs
    )


def _valid() -> ValidationResult:
    return ValidationResult(is_valid=True, code="valid", message=_VALID_MESSAGE)


def _invalid(code: ValidationCode, message: str) -> ValidationResult:
    return ValidationResult(is_valid=False, code=code, message=message)
