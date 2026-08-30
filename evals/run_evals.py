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
            visual_phase = get_game_interrupt(state)
            state = runtime.resume(
                thread_id,
                {"action_id": "inspect_display_case"},
            )
            free_form_phase = get_game_interrupt(state)
            state = runtime.resume(
                thread_id,
                {
                    "request": (
                        "Open the media locker and inspect Rowan's camera card for close-up "
                        "images or buyer messages."
                    )
                },
            )
            accusation_phase = get_game_interrupt(state)
            state = runtime.resume(
                thread_id,
                {
                    "suspect_id": "rowan_pike",
                    "motive": (
                        "A private collector paid Rowan to steal Aurora Circuit and deliver it "
                        "after the gala."
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
        "visual_interrupt": visual_phase is not None
        and visual_phase.get("phase") == "visual_choice",
        "free_form_interrupt": free_form_phase is not None
        and free_form_phase.get("phase") == "free_form",
        "accusation_interrupt": accusation_phase is not None
        and accusation_phase.get("phase") == "accusation",
        "two_actions_used": state["investigation_count"] == 2,
        "free_form_routed": state["completed_action_ids"]
        == ["inspect_display_case", "inspect_media_locker"],
        "expected_evidence": set(evidence_ids) >= {"E01", "E02", "E03", "E06", "E10"},
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
