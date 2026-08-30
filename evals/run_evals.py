"""Run the live Claude investigation and print deterministic evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from case_closed.case_store import DEFAULT_CASE_ID, CaseStore
from case_closed.config import load_config
from case_closed.evaluation import evaluate_state
from case_closed.runtime import create_runtime
from case_closed.state import create_initial_state


def main() -> None:
    """Execute one live case and exit unsuccessfully if an acceptance metric fails."""
    store = CaseStore()
    public_case = store.load_public_case(DEFAULT_CASE_ID)
    initial_state = create_initial_state(public_case, human_review_enabled=False)
    config = load_config(dotenv_path=Path(".env"))

    with TemporaryDirectory(prefix="case-closed-eval-") as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoints.sqlite"
        with create_runtime(
            config,
            checkpoint_path=checkpoint_path,
            store=store,
        ) as runtime:
            final_state = runtime.start(initial_state, f"eval-{uuid4().hex}")

    report = evaluate_state(final_state, store.load_solution(DEFAULT_CASE_ID))
    payload = {"passed": report.passed, **report.model_dump()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
