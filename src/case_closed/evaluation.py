"""Deterministic metrics for completed detective graph runs."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from case_closed.schemas import SolutionKey, Verdict


class EvaluationReport(BaseModel):
    """Portfolio-friendly pass/fail metrics for one investigation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status_resolved: bool
    culprit_correct: bool
    citations_discovered: bool
    evidence_path_complete: bool
    within_round_limit: bool

    @property
    def passed(self) -> bool:
        """Return whether every deterministic acceptance metric passed."""
        return all(
            (
                self.status_resolved,
                self.culprit_correct,
                self.citations_discovered,
                self.evidence_path_complete,
                self.within_round_limit,
            )
        )


def evaluate_state(
    state: Mapping[str, object],
    solution: SolutionKey,
) -> EvaluationReport:
    """Score a final public state against a transient private solution key."""
    raw_verdict = state.get("proposed_verdict")
    verdict = Verdict.model_validate(raw_verdict) if isinstance(raw_verdict, dict) else None
    raw_evidence = state.get("discovered_evidence", [])
    discovered_ids = {
        item["evidence_id"]
        for item in raw_evidence
        if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
    }
    citations = set(verdict.citations) if verdict is not None else set()
    round_count = state.get("round_count")
    max_rounds = state.get("max_rounds")
    return EvaluationReport(
        status_resolved=state.get("status") == "resolved",
        culprit_correct=(verdict is not None and verdict.culprit_id == solution.culprit_id),
        citations_discovered=bool(citations) and citations <= discovered_ids,
        evidence_path_complete=any(
            set(path) <= citations for path in solution.acceptable_evidence_sets
        ),
        within_round_limit=(
            isinstance(round_count, int)
            and isinstance(max_rounds, int)
            and round_count <= max_rounds
        ),
    )
