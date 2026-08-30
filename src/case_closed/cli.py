"""Command-line interface for running the museum investigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from case_closed.case_store import DEFAULT_CASE_ID, CaseStore
from case_closed.config import ConfigurationError, load_config
from case_closed.runtime import create_runtime, get_interrupt_payload
from case_closed.schemas import PublicCase, Verdict
from case_closed.state import DEFAULT_MAX_ROUNDS, CaseState, create_initial_state


def main() -> None:
    """Run the checkpoint-aware detective CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_config(dotenv_path=args.env_file)
    except ConfigurationError as exc:
        parser.error(str(exc))

    store = CaseStore()
    public_case = store.load_public_case(args.case_id)
    initial_state = create_initial_state(
        public_case,
        max_rounds=args.max_rounds,
        human_review_enabled=not args.autonomous,
    )
    thread_id = args.thread_id or f"case-{uuid4().hex[:12]}"

    with create_runtime(
        config,
        checkpoint_path=args.checkpoint_path,
        store=store,
    ) as runtime:
        print(f"\n{public_case.title}")
        print(f"Investigation ID: {thread_id}\n")
        result = runtime.start(initial_state, thread_id)
        while payload := get_interrupt_payload(result):
            _print_interrupt(payload)
            direction = input("Your direction: ").strip()
            if not direction:
                direction = "Continue with the strongest remaining lead."
            result = runtime.resume(thread_id, direction)

    if args.json_output:
        print(json.dumps(_json_result(result), indent=2, ensure_ascii=False, sort_keys=True))
        return
    _print_result(result, public_case)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="case-closed",
        description="Solve a fictional museum heist with a bounded LangGraph investigator.",
    )
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--thread-id")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints.sqlite"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Disable the mid-investigation human direction pause.",
    )
    parser.add_argument("--json", dest="json_output", action="store_true")
    return parser


def _print_interrupt(payload: dict[str, object]) -> None:
    print("\nThe graph paused for human direction.")
    print(payload.get("question", "Choose the next lead."))
    leads = payload.get("suggested_leads", [])
    if isinstance(leads, list):
        for index, lead in enumerate(leads, start=1):
            print(f"  {index}. {lead}")
    print()


def _print_result(result: dict[str, object], public_case: PublicCase) -> None:
    state = cast(CaseState, result)
    print(f"\nStatus: {state['status'].upper()}")
    print(f"Rounds used: {state['round_count']} / {state['max_rounds']}")
    if state["status"] != "resolved" or state["proposed_verdict"] is None:
        print(state["final_report"] or "No supported verdict was reached.")
        return

    verdict = Verdict.model_validate(state["proposed_verdict"])
    suspect_names = {suspect.suspect_id: suspect.name for suspect in public_case.suspects}
    print(f"Culprit: {suspect_names.get(verdict.culprit_id, verdict.culprit_id)}")
    print(f"Confidence: {verdict.confidence}%")
    print(f"Evidence: {', '.join(verdict.citations)}")
    print(f"\n{verdict.summary}\n{verdict.case_theory}")


def _json_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in result.items() if key not in {"__interrupt__", "public_case"}
    }
