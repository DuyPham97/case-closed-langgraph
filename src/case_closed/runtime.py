"""Runtime assembly and checkpoint-aware graph invocation helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import cast

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from case_closed.case_store import CaseStore
from case_closed.config import AppConfig
from case_closed.gateway import ClaudeGateway
from case_closed.graph import InvestigationGateway, build_investigation_graph
from case_closed.state import CaseState, JsonObject

DEFAULT_CHECKPOINT_PATH = Path("checkpoints.sqlite")


class InvestigationRuntime:
    """Own a compiled graph and the SQLite connection backing its checkpoints."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        connection: sqlite3.Connection,
        store: CaseStore,
    ) -> None:
        self.graph = graph
        self.connection = connection
        self.store = store

    def start(self, initial_state: CaseState, thread_id: str) -> dict[str, object]:
        """Run a new investigation until completion or an interrupt."""
        return cast(dict[str, object], self.graph.invoke(initial_state, _thread_config(thread_id)))

    def resume(self, thread_id: str, direction: str) -> dict[str, object]:
        """Resume a paused investigation with player direction."""
        normalized = direction.strip()
        if not normalized:
            raise ValueError("direction must be a non-empty string")
        return cast(
            dict[str, object],
            self.graph.invoke(Command(resume=normalized), _thread_config(thread_id)),
        )

    def close(self) -> None:
        """Close the checkpoint database connection."""
        self.connection.close()

    def __enter__(self) -> InvestigationRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def create_runtime(
    config: AppConfig,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    store: CaseStore | None = None,
    gateway: InvestigationGateway | None = None,
) -> InvestigationRuntime:
    """Create the production runtime with Claude and SQLite by default."""
    case_store = store if store is not None else CaseStore()
    model_gateway = gateway if gateway is not None else ClaudeGateway.from_config(config)
    connection = _open_checkpoint_connection(checkpoint_path)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    checkpointer = SqliteSaver(connection, serde=serializer)
    graph = build_investigation_graph(model_gateway, case_store, checkpointer=checkpointer)
    return InvestigationRuntime(graph, connection, case_store)


def get_interrupt_payload(result: Mapping[str, object]) -> JsonObject | None:
    """Extract the first JSON interrupt payload from an invocation result."""
    interrupts = result.get("__interrupt__")
    if not isinstance(interrupts, (tuple, list)) or not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    if not isinstance(value, dict):
        return None
    return cast(JsonObject, value)


def _open_checkpoint_connection(checkpoint_path: str | Path) -> sqlite3.Connection:
    raw_path = str(checkpoint_path)
    if raw_path != ":memory:":
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = str(path)
    return sqlite3.connect(raw_path, check_same_thread=False)


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    normalized = thread_id.strip()
    if not normalized:
        raise ValueError("thread_id must be a non-empty string")
    return {"configurable": {"thread_id": normalized}}
