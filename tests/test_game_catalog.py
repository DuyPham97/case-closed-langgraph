"""Tests for the bounded player-action catalog and museum map."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest

from case_closed.case_store import CaseStore
from case_closed.game_catalog import (
    all_game_actions,
    get_game_action,
    visual_game_actions,
)

ROOT = Path(__file__).parents[1]
MAP_PATH = ROOT / "assets" / "game" / "museum-map.svg"


def test_catalog_covers_every_scripted_route_once() -> None:
    case = CaseStore().load_public_case()
    action_routes = [
        (action.tool_name, action.target_id, action.topic) for action in all_game_actions()
    ]
    case_routes = [(route.tool_name, route.target_id, route.topic) for route in case.reveal_routes]

    assert len(all_game_actions()) == 14
    assert len(action_routes) == len(set(action_routes))
    assert set(action_routes) == set(case_routes)


def test_catalog_has_four_locations_eight_interviews_and_two_record_checks() -> None:
    actions = all_game_actions()

    assert Counter(action.category for action in actions) == {
        "location": 4,
        "interview": 8,
        "records": 2,
    }
    assert len({action.action_id for action in actions}) == len(actions)


def test_only_the_four_location_choices_are_visual() -> None:
    visual_actions = visual_game_actions()

    assert [action.action_id for action in visual_actions] == [
        "inspect_display_case",
        "inspect_security_screening",
        "inspect_media_locker",
        "inspect_lighting_booth",
    ]
    assert all(action.category == "location" for action in visual_actions)
    assert all(action.is_visual for action in visual_actions)
    assert not any(action.is_visual for action in all_game_actions()[4:])


def test_catalog_lookup_rejects_unknown_action() -> None:
    action = get_game_action("ask_rowan_equipment")

    assert action.target_id == "rowan_pike"
    assert action.topic == "equipment"
    with pytest.raises(KeyError, match="unknown game action"):
        get_game_action("search_the_roof")


def test_player_facing_action_copy_contains_no_implementation_language() -> None:
    copy = " ".join(f"{action.title} {action.description}" for action in all_game_actions()).lower()

    for marker in (
        "demo",
        "portfolio",
        "langchain",
        "langgraph",
        "source_reference",
        "target_id",
        "topic_id",
        "plinth/",
        "case/",
    ):
        assert marker not in copy


def test_two_visual_investigations_can_reveal_complete_proof() -> None:
    store = CaseStore()

    display = store.inspect_location("midnight_museum", "display_case")
    screening = store.inspect_location("midnight_museum", "security_screening")

    assert set(display.evidence_ids + screening.evidence_ids) >= {"E01", "E04", "E07"}


def test_museum_map_exposes_exactly_the_visual_action_ids() -> None:
    root = ElementTree.parse(MAP_PATH).getroot()
    map_action_ids = {
        element.attrib["data-action-id"]
        for element in root.iter()
        if "data-action-id" in element.attrib
    }
    labels = " ".join(text.strip() for text in root.itertext() if text.strip()).lower()

    assert map_action_ids == {action.action_id for action in visual_game_actions()}
    assert "gallery 3" in labels
    assert "security screening" in labels
    assert "media locker" in labels
    assert "lighting booth" in labels
    assert root.attrib["viewBox"] == "0 0 1600 900"
    assert root.attrib["role"] == "img"
