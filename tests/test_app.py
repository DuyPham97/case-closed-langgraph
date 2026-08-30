"""Landing-page smoke test for the Streamlit portfolio UI."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_caseboard_renders_without_api_credentials() -> None:
    app_path = Path(__file__).parents[1] / "src" / "case_closed" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Case Closed?"
    assert app.subheader[0].value == "Midnight at the Lumen Museum"
    assert app.sidebar.button[0].label == "Start a fresh case"
    assert app.button[0].label == "Begin investigation"
    assert app.expander[0].label == "Read suspect profiles"
    for suspect_name in ("Mara Vale", "Theo Quinn", "Nia Brooks", "Rowan Pike"):
        assert any(suspect_name in markdown.value for markdown in app.markdown)
