"""Tests for the bounded deep-dive catalog and freely browsable museum map."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest

from case_closed.case_store import CaseStore
from case_closed.game_catalog import all_game_actions, get_game_action

ROOT = Path(__file__).parents[1]
MAP_PATH = ROOT / "assets" / "game" / "museum-map.svg"


def test_catalog_covers_every_scripted_deep_dive_once() -> None:
    case = CaseStore().load_public_case()
    action_routes = [
        (action.tool_name, action.target_id, action.topic) for action in all_game_actions()
    ]
    case_routes = [(route.tool_name, route.target_id, route.topic) for route in case.reveal_routes]

    assert len(all_game_actions()) == 6
    assert len(action_routes) == len(set(action_routes))
    assert set(action_routes) < set(case_routes)


def test_catalog_uses_interviews_location_records_and_cross_person_leads() -> None:
    actions = all_game_actions()

    assert Counter(action.category for action in actions) == {
        "interview": 5,
        "records": 1,
    }
    assert {action.tool_name for action in actions} == {
        "interview_suspect",
        "compare_timeline",
    }
    assert len({action.action_id for action in actions}) == len(actions)
    assert any("Rowan" in action.title and action.target_id != "rowan_pike" for action in actions)


def test_catalog_lookup_rejects_unknown_action() -> None:
    action = get_game_action("crosscheck_rowans_case")

    assert action.target_id == "rowan_case_chain"
    assert action.tool_name == "compare_timeline"
    with pytest.raises(KeyError, match="unknown game action"):
        get_game_action("search_the_roof")


def test_player_facing_action_copy_contains_no_answer_or_implementation_language() -> None:
    copy = " ".join(f"{action.title} {action.description}" for action in all_game_actions()).lower()

    for marker in (
        "demo",
        "portfolio",
        "langchain",
        "langgraph",
        "source_reference",
        "target_id",
        "topic_id",
        "culprit",
        "rowan stole",
        "motive is",
    ):
        assert marker not in copy


def test_every_deep_dive_adds_one_follow_up_without_repeating_base_dossier() -> None:
    case = CaseStore().load_public_case()
    base_ids = set(case.case_file_evidence_ids)
    action_routes = {
        (action.tool_name, action.target_id, action.topic) for action in all_game_actions()
    }
    revealed_ids = [
        evidence_id
        for route in case.reveal_routes
        if (route.tool_name, route.target_id, route.topic) in action_routes
        for evidence_id in route.evidence_ids
    ]

    assert len(revealed_ids) == len(set(revealed_ids)) == 6
    assert base_ids.isdisjoint(revealed_ids)


def test_museum_map_keeps_all_four_locations_freely_browsable() -> None:
    case = CaseStore().load_public_case()
    root = ElementTree.parse(MAP_PATH).getroot()
    labels = " ".join(text.strip() for text in root.itertext() if text.strip()).lower()

    assert {location.name.lower() for location in case.locations} <= {
        "gallery 3",
        "media locker",
        "security screening",
        "lighting booth",
    }
    for label in ("gallery 3", "security screening", "media locker", "lighting booth"):
        assert label in labels
    assert root.attrib["viewBox"] == "0 0 1600 900"
    assert root.attrib["role"] == "img"
