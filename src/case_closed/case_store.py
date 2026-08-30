"""Load bundled case data and expose public-only deterministic observations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from case_closed.schemas import PublicCase, SolutionKey, ToolName, ToolObservation

DEFAULT_CASE_ID: Final = "midnight_museum"
_CASE_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class CaseStoreError(RuntimeError):
    """Base exception for case repository failures."""


class CaseNotFoundError(CaseStoreError):
    """Raised when a requested bundled case does not exist."""


class InvalidCaseDataError(CaseStoreError):
    """Raised when bundled case JSON fails decoding or validation."""


class UnknownToolTargetError(CaseStoreError):
    """Raised when a tool call does not match a declared public route."""


class CaseStore:
    """Read case files while keeping answer keys out of normal tool paths."""

    def __init__(self, cases_dir: Path | None = None) -> None:
        self._cases_dir = cases_dir if cases_dir is not None else Path(__file__).parent / "cases"

    def load_public_case(self, case_id: str = DEFAULT_CASE_ID) -> PublicCase:
        """Load and validate the public portion of a case."""
        path = self._case_directory(case_id) / "public.json"
        payload = self._read_json_object(path, case_id)
        try:
            case = PublicCase.model_validate(payload)
        except ValidationError as exc:
            raise InvalidCaseDataError(f"invalid public case data for {case_id}") from exc
        if case.case_id != case_id:
            raise InvalidCaseDataError(
                f"public case ID {case.case_id!r} does not match directory {case_id!r}"
            )
        return case

    def load_solution(self, case_id: str = DEFAULT_CASE_ID) -> SolutionKey:
        """Load the private answer key for deterministic verdict validation only."""
        path = self._case_directory(case_id) / "solution.json"
        payload = self._read_json_object(path, case_id)
        try:
            solution = SolutionKey.model_validate(payload)
        except ValidationError as exc:
            raise InvalidCaseDataError(f"invalid solution data for {case_id}") from exc

        public_case = self.load_public_case(case_id)
        self._validate_solution(solution, public_case)
        return solution

    def inspect_location(
        self,
        case_id: str,
        location_id: str,
    ) -> ToolObservation:
        """Return the deterministic public observation for an inspectable location."""
        return self._run_public_route(case_id, "inspect_location", location_id, None)

    def interview_suspect(
        self,
        case_id: str,
        suspect_id: str,
        topic: str,
    ) -> ToolObservation:
        """Return the scripted public response for a suspect and topic."""
        return self._run_public_route(case_id, "interview_suspect", suspect_id, topic)

    def compare_timeline(
        self,
        case_id: str,
        anchor_id: str,
    ) -> ToolObservation:
        """Return a deterministic comparison around a declared timeline anchor."""
        return self._run_public_route(case_id, "compare_timeline", anchor_id, None)

    def _run_public_route(
        self,
        case_id: str,
        tool_name: ToolName,
        target_id: str,
        topic: str | None,
    ) -> ToolObservation:
        public_case = self.load_public_case(case_id)
        route = next(
            (
                candidate
                for candidate in public_case.reveal_routes
                if candidate.tool_name == tool_name
                and candidate.target_id == target_id
                and candidate.topic == topic
            ),
            None,
        )
        if route is None:
            target = f"{target_id}/{topic}" if topic is not None else target_id
            raise UnknownToolTargetError(f"no {tool_name} route for {target!r}")

        evidence_by_id = {record.evidence_id: record for record in public_case.observations}
        evidence_lines = [
            f"[{evidence_id}] {evidence_by_id[evidence_id].title}: "
            f"{evidence_by_id[evidence_id].text}"
            for evidence_id in route.evidence_ids
        ]
        summary = route.summary
        if evidence_lines:
            summary = f"{summary}\n\n" + "\n".join(evidence_lines)
        return ToolObservation(
            summary=summary,
            evidence_ids=list(route.evidence_ids),
            no_new_evidence=not route.evidence_ids,
        )

    def _case_directory(self, case_id: str) -> Path:
        if _CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise CaseNotFoundError(f"invalid case ID: {case_id!r}")
        return self._cases_dir / case_id

    @staticmethod
    def _read_json_object(path: Path, case_id: str) -> dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise CaseNotFoundError(f"case file not found for {case_id}: {path.name}") from exc
        except OSError as exc:
            raise CaseStoreError(f"could not read case file for {case_id}: {path.name}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidCaseDataError(f"invalid JSON for {case_id}: {path.name}") from exc
        if not isinstance(payload, dict):
            raise InvalidCaseDataError(f"case file must contain an object: {path.name}")
        return payload

    @staticmethod
    def _validate_solution(solution: SolutionKey, public_case: PublicCase) -> None:
        if solution.case_id != public_case.case_id:
            raise InvalidCaseDataError("solution case ID does not match public case ID")

        suspect_ids = {suspect.suspect_id for suspect in public_case.suspects}
        evidence_ids = {record.evidence_id for record in public_case.observations}
        if solution.culprit_id not in suspect_ids:
            raise InvalidCaseDataError("solution culprit is not a public suspect")

        referenced_evidence: set[str] = set()
        for evidence_set in solution.acceptable_evidence_sets:
            if not evidence_set:
                raise InvalidCaseDataError("acceptable evidence sets cannot be empty")
            referenced_evidence.update(evidence_set)
        for rule in solution.claim_rules:
            for evidence_set in rule.accepted_evidence_sets:
                if not evidence_set:
                    raise InvalidCaseDataError("claim evidence sets cannot be empty")
                referenced_evidence.update(evidence_set)
        for pair in solution.contradiction_pairs:
            referenced_evidence.add(pair.statement_evidence_id)
            referenced_evidence.update(pair.rebuttal_evidence_ids)
        for rule in solution.exoneration_rules:
            if rule.suspect_id not in suspect_ids:
                raise InvalidCaseDataError(
                    f"exoneration references unknown suspect: {rule.suspect_id}"
                )
            referenced_evidence.update(rule.evidence_ids)

        unknown_evidence = referenced_evidence - evidence_ids
        if unknown_evidence:
            raise InvalidCaseDataError(
                f"solution references unknown evidence: {sorted(unknown_evidence)}"
            )

        claim_ids = [rule.claim_id for rule in solution.claim_rules]
        if len(claim_ids) != len(set(claim_ids)):
            raise InvalidCaseDataError("solution claim IDs must be unique")

        expected_exonerations = suspect_ids - {solution.culprit_id}
        actual_exonerations = {rule.suspect_id for rule in solution.exoneration_rules}
        if actual_exonerations != expected_exonerations:
            raise InvalidCaseDataError(
                "solution must exonerate every innocent suspect exactly once"
            )
