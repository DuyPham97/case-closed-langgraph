"""LangChain tools backed exclusively by deterministic public case data."""

from __future__ import annotations

from langchain.tools import BaseTool, tool

from case_closed.case_store import DEFAULT_CASE_ID, CaseStore
from case_closed.schemas import ToolObservation


def create_case_tools(
    store: CaseStore | None = None,
    case_id: str = DEFAULT_CASE_ID,
) -> tuple[BaseTool, BaseTool, BaseTool]:
    """Build the three case tools with an injectable repository for offline tests."""
    case_store = store if store is not None else CaseStore()

    @tool("inspect_location")
    def inspect_location(location_id: str) -> ToolObservation:
        """Inspect a declared museum location by its public location ID."""
        return case_store.inspect_location(case_id, location_id)

    @tool("interview_suspect")
    def interview_suspect(suspect_id: str, topic: str) -> ToolObservation:
        """Interview a declared suspect about one of their available public topics."""
        return case_store.interview_suspect(case_id, suspect_id, topic)

    @tool("compare_timeline")
    def compare_timeline(anchor_id: str) -> ToolObservation:
        """Compare source-backed events around a declared public timeline anchor."""
        return case_store.compare_timeline(case_id, anchor_id)

    return inspect_location, interview_suspect, compare_timeline
