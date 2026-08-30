"""Landing-page and player-facing safety tests for the Streamlit game."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parents[1]
APP_PATH = ROOT / "src" / "case_closed" / "app.py"


def test_game_landing_renders_without_api_credentials() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Midnight at the Lumen Museum"
    assert app.sidebar.button[0].label == "Restart case"
    assert app.button[0].label == "Enter Gallery 3"
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
