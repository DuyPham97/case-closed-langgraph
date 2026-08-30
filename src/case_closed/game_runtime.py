"""Checkpointed runtime helpers for the player-driven game graph."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import cast

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from case_closed.case_store import DEFAULT_CASE_ID, CaseStore
from case_closed.config import AppConfig
from case_closed.game_gateway import PlayerGameGateway
from case_closed.game_graph import PlayerGameGatewayProtocol, build_player_game_graph
from case_closed.game_state import create_player_game_state
from case_closed.runtime import DEFAULT_CHECKPOINT_PATH
from case_closed.state import JsonObject


class PlayerGameRuntime:
    """Own the player game graph and its SQLite checkpoint connection."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        connection: sqlite3.Connection,
        store: CaseStore,
    ) -> None:
        self.graph = graph
        self.connection = connection
        self.store = store

    def start(
        self,
        case_id: str = DEFAULT_CASE_ID,
        thread_id: str = "",
    ) -> dict[str, object]:
        """Start a fresh case and pause for the first visual choice."""
        public_case = self.store.load_public_case(case_id)
        initial_state = create_player_game_state(public_case)
        return cast(
            dict[str, object],
            self.graph.invoke(initial_state, _thread_config(thread_id)),
        )

    def resume(
        self,
        thread_id: str,
        payload: str | Mapping[str, object],
    ) -> dict[str, object]:
        """Resume the current human boundary with a JSON-safe player payload."""
        _validate_resume_payload(payload)
        normalized: str | dict[str, object]
        normalized = payload.strip() if isinstance(payload, str) else dict(payload)
        return cast(
            dict[str, object],
            self.graph.invoke(Command(resume=normalized), _thread_config(thread_id)),
        )

    def close(self) -> None:
        """Close the checkpoint database connection."""
        self.connection.close()

    def __enter__(self) -> PlayerGameRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def create_player_game_runtime(
    config: AppConfig,
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    store: CaseStore | None = None,
    gateway: PlayerGameGatewayProtocol | None = None,
) -> PlayerGameRuntime:
    """Create the production player game with Claude and SQLite checkpoints."""
    case_store = store if store is not None else CaseStore()
    game_gateway = gateway if gateway is not None else PlayerGameGateway.from_config(config)
    connection = _open_checkpoint_connection(checkpoint_path)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    checkpointer = SqliteSaver(connection, serde=serializer)
    graph = build_player_game_graph(game_gateway, case_store, checkpointer=checkpointer)
    return PlayerGameRuntime(graph, connection, case_store)


def get_game_interrupt(result: Mapping[str, object]) -> JsonObject | None:
    """Extract the current player-facing interrupt payload from a graph result."""
    interrupts = result.get("__interrupt__")
    if not isinstance(interrupts, (tuple, list)) or not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    if not isinstance(value, dict):
        return None
    return cast(JsonObject, value)


def _validate_resume_payload(payload: str | Mapping[str, object]) -> None:
    if isinstance(payload, str):
        if not payload.strip():
            raise ValueError("resume payload must not be empty")
        return
    try:
        json.dumps(dict(payload), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("resume payload must be JSON serializable") from exc


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
