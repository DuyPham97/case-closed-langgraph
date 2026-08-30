"""Tests for SQLite persistence and runtime invocation helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from case_closed.case_store import CaseStore
from case_closed.config import AppConfig
from case_closed.runtime import create_runtime, get_interrupt_payload
from case_closed.schemas import InvestigationAction, ProgressAssessment, Verdict
from case_closed.state import create_initial_state
from tests.fakes import ScriptedGateway


@dataclass(frozen=True)
class _UntrustedCheckpointValue:
    value: str


def _action(tool_name: str, target_id: str) -> InvestigationAction:
    return InvestigationAction(
        tool_name=tool_name,
        target_id=target_id,
        reason=f"Investigate {target_id}.",
    )


def _assessment() -> ProgressAssessment:
    return ProgressAssessment(
        summary="Continue collecting source-backed evidence.",
        hypotheses=[],
        ready_for_verdict=False,
        next_leads=["Inspect the security screening records."],
    )


def _verdict() -> Verdict:
    return Verdict(
        culprit_id="rowan_pike",
        summary="Rowan removed Aurora Circuit before the blackout.",
        case_theory="The timing, access, and matching weight establish the theft.",
        confidence=95,
        citations=["E01", "E04", "E07"],
    )


def _config() -> AppConfig:
    return AppConfig(anthropic_api_key="test-key")


def test_runtime_resumes_from_sqlite_after_reconstruction(tmp_path: Path) -> None:
    store = CaseStore()
    checkpoint_path = tmp_path / "case.sqlite"
    first_gateway = ScriptedGateway(
        actions=[
            _action("inspect_location", "display_case"),
            _action("compare_timeline", "weight_drop"),
        ],
        assessments=[_assessment(), _assessment()],
        verdicts=[],
    )
    initial = create_initial_state(store.load_public_case(), human_review_enabled=True)

    with create_runtime(
        _config(),
        checkpoint_path=checkpoint_path,
        store=store,
        gateway=first_gateway,
    ) as first_runtime:
        paused = first_runtime.start(initial, "persisted-case")

    assert get_interrupt_payload(paused) is not None

    second_gateway = ScriptedGateway(
        actions=[_action("inspect_location", "security_screening")],
        assessments=[_assessment()],
        verdicts=[_verdict()],
    )
    with create_runtime(
        _config(),
        checkpoint_path=checkpoint_path,
        store=store,
        gateway=second_gateway,
    ) as second_runtime:
        completed = second_runtime.resume(
            "persisted-case",
            "Inspect the security screening records.",
        )

    assert completed["status"] == "resolved"
    assert completed["human_review_completed"] is True
    assert completed["round_count"] == 3


def test_runtime_rejects_blank_identifiers_and_direction() -> None:
    store = CaseStore()
    gateway = ScriptedGateway(actions=[], assessments=[], verdicts=[])

    with create_runtime(
        _config(),
        checkpoint_path=":memory:",
        store=store,
        gateway=gateway,
    ) as runtime:
        with pytest.raises(ValueError, match="thread_id"):
            runtime.start(create_initial_state(store.load_public_case()), " ")
        with pytest.raises(ValueError, match="direction"):
            runtime.resume("thread", " ")


def test_runtime_context_manager_closes_connection() -> None:
    store = CaseStore()
    gateway = ScriptedGateway(actions=[], assessments=[], verdicts=[])

    with create_runtime(
        _config(),
        checkpoint_path=":memory:",
        store=store,
        gateway=gateway,
    ) as runtime:
        connection = runtime.connection

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_runtime_uses_a_strict_checkpoint_serializer() -> None:
    store = CaseStore()
    gateway = ScriptedGateway(actions=[], assessments=[], verdicts=[])

    with create_runtime(
        _config(),
        checkpoint_path=":memory:",
        store=store,
        gateway=gateway,
    ) as runtime:
        checkpointer = runtime.graph.checkpointer
        assert checkpointer is not None
        serializer = checkpointer.serde

        assert isinstance(serializer, JsonPlusSerializer)
        safe_payload = {"case_id": "midnight_museum", "evidence_ids": ["E01"]}
        assert serializer.loads_typed(serializer.dumps_typed(safe_payload)) == safe_payload

        encoded = serializer.dumps_typed(_UntrustedCheckpointValue(value="blocked"))
        decoded = serializer.loads_typed(encoded)
        assert decoded == {"value": "blocked"}
        assert not isinstance(decoded, _UntrustedCheckpointValue)
