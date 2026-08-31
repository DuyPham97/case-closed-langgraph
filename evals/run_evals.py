"""Run one live Claude playthrough and print player-flow acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from case_closed.config import load_config
from case_closed.game_runtime import create_player_game_runtime, get_game_interrupt


def main() -> None:
    """Exercise routing, evidence, accusation matching, and the terminal debrief."""
    config = load_config(dotenv_path=Path(".env"))
    thread_id = f"eval-{uuid4().hex}"
    with TemporaryDirectory(prefix="case-closed-eval-") as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoints.sqlite"
        with create_player_game_runtime(config, checkpoint_path=checkpoint_path) as runtime:
            state = runtime.start(thread_id=thread_id)
            open_dossier_phase = get_game_interrupt(state)
            state = runtime.resume(
                thread_id,
                {
                    "request": (
                        "Ask Theo who heard his warning that the original timing could not be "
                        "restored after the migration."
                    )
                },
            )
            second_move_phase = get_game_interrupt(state)
            state = runtime.resume(
                thread_id,
                {
                    "request": (
                        "Reconcile Rowan's equipment-case weight with the entry manifest and "
                        "chain of custody."
                    )
                },
            )
            accusation_phase = get_game_interrupt(state)
            state = runtime.resume(
                thread_id,
                {
                    "suspect_id": "rowan_pike",
                    "motive": (
                        "Rowan wanted to stop the museum from irreversibly erasing Iris Venn's "
                        "original pulse timing and buy time for an estate review."
                    ),
                    "method": (
                        "He used calibration access to bypass the sensor, installed a polarized "
                        "decoy, and hid the sculpture in his camera case before the automatic "
                        "blackout."
                    ),
                },
            )

    result = state["result"]
    evidence_ids = [record["evidence_id"] for record in state["discovered_evidence"]]
    checks = {
        "open_dossier_interrupt": open_dossier_phase is not None
        and open_dossier_phase.get("phase") == "free_form",
        "two_optional_credits": open_dossier_phase is not None
        and open_dossier_phase.get("investigations_remaining") == 2,
        "second_move_interrupt": second_move_phase is not None
        and second_move_phase.get("phase") == "free_form",
        "accusation_interrupt": accusation_phase is not None
        and accusation_phase.get("phase") == "accusation",
        "two_actions_used": state["investigation_count"] == 2,
        "free_form_routed": state["completed_action_ids"]
        == ["ask_theo_who_heard_warning", "crosscheck_rowans_case"],
        "base_and_follow_up_evidence": set(evidence_ids)
        >= {"E01", "E06", "E10", "E11", "E12", "E14", "E18"},
        "culprit_correct": result["culprit_correct"] is True,
        "motive_match": result["motive_match"] is True,
        "method_match": result["method_match"] is True,
        "case_solved": state["status"] == "solved" and result["tier"] == "solved",
        "debrief_written": bool(state["debrief"]),
    }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence_ids": evidence_ids,
        "headline": state["debrief"]["headline"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
