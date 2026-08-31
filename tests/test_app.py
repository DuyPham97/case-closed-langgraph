"""Landing-page and player-facing safety tests for the Streamlit game."""

from __future__ import annotations

from pathlib import Path

from langgraph.types import Interrupt
from streamlit.testing.v1 import AppTest

from case_closed.case_store import CaseStore
from case_closed.game_state import create_player_game_state

ROOT = Path(__file__).parents[1]
APP_PATH = ROOT / "src" / "case_closed" / "app.py"


def _active_game_with_one_follow_up() -> AppTest:
    store = CaseStore()
    public_case = store.load_public_case()
    state = create_player_game_state(public_case)
    follow_up = next(record for record in public_case.observations if record.evidence_id == "E14")
    state["investigation_count"] = 1
    state["completed_action_ids"] = ["ask_theo_who_heard_warning"]
    state["discovered_evidence"].append(
        {
            "evidence_id": follow_up.evidence_id,
            "title": follow_up.title,
            "text": follow_up.text,
            "occurred_at": follow_up.occurred_at.isoformat(),
        }
    )
    state["last_observation"] = {
        "summary": "Theo admits someone heard his warning.",
        "evidence_ids": ["E14"],
        "no_new_evidence": False,
    }
    result = dict(state)
    result["__interrupt__"] = [
        Interrupt(
            value={
                "phase": "free_form",
                "investigations_remaining": 1,
                "can_accuse": True,
            }
        )
    ]

    app = AppTest.from_file(str(APP_PATH), default_timeout=10)
    app.session_state["thread_id"] = "active-ui-test"
    app.session_state["game_result"] = result
    app.session_state["ui_error"] = None
    app.session_state["solution_revealed"] = False
    app.session_state["selected_room_id"] = "display_case"
    return app.run()


def test_game_landing_renders_without_api_credentials() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Midnight at the Lumen Museum"
    assert app.sidebar.button[0].label == "Restart case"
    assert app.button[0].label == "Open the case file"
    for suspect_name in ("Mara Vale", "Theo Quinn", "Nia Brooks", "Rowan Pike"):
        assert any(suspect_name in markdown.value for markdown in app.markdown)


def test_player_interface_omits_implementation_and_source_labels() -> None:
    source = APP_PATH.read_text()

    for forbidden in (
        "PORTFOLIO DEMO",
        "LangChain",
        "LangGraph",
        "Model:",
        "Public graph execution trace",
        "source_reference",
        "plinth/W-3/load-audit",
        "case/G3/rear-panel-audit",
    ):
        assert forbidden not in source


def test_player_interface_promises_free_browsing_and_optional_deep_dives() -> None:
    source = APP_PATH.read_text()

    for expected in (
        "Browse every marked room",
        "Optional deep dive",
        "Spend 1 deep-dive credit",
        "Make accusation now",
    ):
        assert expected in source


def test_every_declared_game_asset_exists() -> None:
    expected = {
        "crime-scene.webp",
        "museum-map.svg",
        "mara-vale.webp",
        "theo-quinn.webp",
        "nia-brooks.webp",
        "rowan-pike.webp",
        "security-screening.webp",
        "media-locker.webp",
        "lighting-booth.webp",
        "timeline.webp",
        "evidence-e01.webp",
        "evidence-e02.webp",
        "evidence-e03.webp",
        "evidence-e04.webp",
    }

    assert expected <= {path.name for path in (ROOT / "assets" / "game").iterdir()}


def test_active_case_browses_rooms_and_renders_one_follow_up() -> None:
    app = _active_game_with_one_follow_up()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Scene & map",
        "Evidence · 12 + 1",
        "People",
        "Timeline",
    ]
    button_labels = {button.label for button in app.button}
    assert {
        "01 · Gallery 3",
        "02 · Media locker",
        "03 · Security screening",
        "04 · Lighting booth",
    } <= button_labels
    assert [expander.label for expander in app.expander] == ["Open an optional hint"]
    assert any("Theo's omitted warning" in markdown.value for markdown in app.markdown)

    expected_rooms = (
        ("display_case", None, "### Gallery 3", "Plinth W-3 telemetry"),
        ("media_locker", "02 · Media locker", "### Media locker", "Equipment-chain measurements"),
        (
            "security_screening",
            "03 · Security screening",
            "### Security screening",
            "9:42 synchronization snapshot",
        ),
        (
            "lighting_booth",
            "04 · Lighting booth",
            "### Lighting booth",
            "BLACKOUT_90 console audit",
        ),
    )
    for room_id, button_label, heading, record_title in expected_rooms:
        if button_label is not None:
            next(button for button in app.button if button.label == button_label).click().run()
        assert not app.exception
        assert app.session_state["selected_room_id"] == room_id
        assert any(heading in markdown.value for markdown in app.markdown)
        assert any(record_title in markdown.value for markdown in app.markdown)


def test_active_case_timeline_is_rendered_chronologically() -> None:
    app = _active_game_with_one_follow_up()
    rendered = [markdown.value for markdown in app.markdown]

    case_chain = next(index for index, value in enumerate(rendered) if "hard case moves" in value)
    gallery_session = next(
        index for index, value in enumerate(rendered) if "only recorded occupant" in value
    )
    weight_drop = next(
        index for index, value in enumerate(rendered) if "load falls to zero" in value
    )

    assert case_chain < gallery_session < weight_drop
