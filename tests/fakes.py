"""Scripted model gateway used by offline graph tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from case_closed.schemas import InvestigationAction, ProgressAssessment, Verdict


@dataclass
class ScriptedGateway:
    """Return predefined structured model outputs without network access."""

    actions: list[InvestigationAction]
    assessments: list[ProgressAssessment]
    verdicts: list[Verdict]
    repaired_verdicts: list[Verdict] = field(default_factory=list)
    action_calls: int = 0
    assessment_calls: int = 0
    verdict_calls: int = 0
    repair_calls: int = 0

    def plan_action(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        hypotheses: tuple[dict[str, object], ...],
        round_count: int,
        max_rounds: int,
        human_direction: str | None,
    ) -> InvestigationAction:
        """Return the next scripted action."""
        del public_case, discovered_evidence, hypotheses, round_count, max_rounds, human_direction
        action = self.actions[self.action_calls]
        self.action_calls += 1
        return action

    def assess_progress(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        previous_hypotheses: tuple[dict[str, object], ...],
        round_count: int,
        max_rounds: int,
    ) -> ProgressAssessment:
        """Return the next scripted assessment."""
        del public_case, discovered_evidence, previous_hypotheses, round_count, max_rounds
        assessment = self.assessments[self.assessment_calls]
        self.assessment_calls += 1
        return assessment

    def draft_verdict(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        hypotheses: tuple[dict[str, object], ...],
    ) -> Verdict:
        """Return the next scripted initial verdict."""
        del public_case, discovered_evidence, hypotheses
        verdict = self.verdicts[self.verdict_calls]
        self.verdict_calls += 1
        return verdict

    def repair_verdict(
        self,
        public_case: dict[str, object],
        discovered_evidence: tuple[dict[str, object], ...],
        previous_verdict: dict[str, object],
        validation_errors: tuple[dict[str, object], ...],
    ) -> Verdict:
        """Return the next scripted repaired verdict."""
        del public_case, discovered_evidence, previous_verdict, validation_errors
        verdict = self.repaired_verdicts[self.repair_calls]
        self.repair_calls += 1
        return verdict
