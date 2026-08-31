"""Immersive Streamlit interface for Midnight at the Lumen Museum."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import streamlit as st

from case_closed.case_store import CaseStore
from case_closed.config import ConfigurationError, load_config
from case_closed.game_catalog import get_game_action
from case_closed.game_runtime import (
    PlayerGameRuntime,
    create_player_game_runtime,
    get_game_interrupt,
)
from case_closed.game_schemas import GameDebrief, GameResult
from case_closed.game_state import PlayerGameState
from case_closed.schemas import PublicCase

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets" / "game"

SUSPECT_ASSETS = {
    "mara_vale": "mara-vale.webp",
    "theo_quinn": "theo-quinn.webp",
    "nia_brooks": "nia-brooks.webp",
    "rowan_pike": "rowan-pike.webp",
}
EVIDENCE_ASSETS = {
    "E01": "evidence-e01.webp",
    "E02": "evidence-e02.webp",
    "E03": "evidence-e03.webp",
    "E04": "evidence-e04.webp",
    "E05": "rowan-pike.webp",
    "E06": "media-locker.webp",
    "E07": "security-screening.webp",
    "E08": "lighting-booth.webp",
    "E09": "timeline.webp",
    "E10": "lighting-booth.webp",
    "E11": "media-locker.webp",
    "E12": "timeline.webp",
    "E13": "mara-vale.webp",
    "E14": "theo-quinn.webp",
    "E15": "mara-vale.webp",
    "E16": "security-screening.webp",
    "E17": "rowan-pike.webp",
    "E18": "timeline.webp",
}
ROOM_ASSETS = {
    "display_case": "crime-scene.webp",
    "media_locker": "media-locker.webp",
    "security_screening": "security-screening.webp",
    "lighting_booth": "lighting-booth.webp",
}
ROOM_RECORD_IDS = {
    "display_case": ("E01", "E02", "E03", "E06"),
    "media_locker": ("E07", "E11"),
    "security_screening": ("E04", "E09"),
    "lighting_booth": ("E08", "E10"),
}
ROOM_NOTES = {
    "display_case": (
        "Read the physical scene beside its instrument logs. The display changed before staff "
        "realized anything was missing."
    ),
    "media_locker": (
        "Checked camera gear, removable media, and the hard case stayed here after Rowan's shoot."
    ),
    "security_screening": (
        "Door, occupancy, body-camera, and equipment measurements meet at this desk."
    ),
    "lighting_booth": (
        "The show cue and the sculpture's morning controller work were managed from this room."
    ),
}


def render_app() -> None:
    """Render the complete player-driven museum mystery."""
    st.set_page_config(
        page_title="Midnight at the Lumen Museum",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="auto",
    )
    _inject_styles()
    _initialize_session()

    store = CaseStore()
    public_case = store.load_public_case()
    result = cast(dict[str, object] | None, st.session_state.game_result)

    _render_sidebar(public_case, result)
    if result is None:
        _render_landing(public_case)
        return
    _render_case(public_case, store, result)


@st.cache_resource
def _runtime() -> PlayerGameRuntime:
    config = load_config(dotenv_path=PROJECT_ROOT / ".env")
    return create_player_game_runtime(config)


def _initialize_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = _new_thread_id()
    if "game_result" not in st.session_state:
        st.session_state.game_result = None
    if "ui_error" not in st.session_state:
        st.session_state.ui_error = None
    if "solution_revealed" not in st.session_state:
        st.session_state.solution_revealed = False
    if "selected_room_id" not in st.session_state:
        st.session_state.selected_room_id = "display_case"


def _new_thread_id() -> str:
    return f"case-{uuid4().hex[:12]}"


def _asset(filename: str) -> str:
    return str(ASSET_ROOT / filename)


def _restart_case() -> None:
    st.session_state.thread_id = _new_thread_id()
    st.session_state.game_result = None
    st.session_state.ui_error = None
    st.session_state.solution_revealed = False
    st.session_state.selected_room_id = "display_case"


def _render_sidebar(
    public_case: PublicCase,
    result: dict[str, object] | None,
) -> None:
    used = 0 if result is None else int(result.get("investigation_count", 0))
    remaining = max(0, 2 - used)
    dots = "".join(
        '<span class="credit available"></span>'
        if index < remaining
        else '<span class="credit"></span>'
        for index in range(2)
    )
    with st.sidebar:
        st.markdown(
            """
            <div class="side-mark">LM</div>
            <div class="side-kicker">RESTRICTED ACCESS</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"### {public_case.title}")
        st.caption("Case file 07 · Gallery wing")
        st.markdown("<div class='side-rule'></div>", unsafe_allow_html=True)
        st.markdown("<p class='side-label'>INVESTIGATION CREDITS</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='credit-row'>{dots}<strong>{remaining} / 2</strong></div>",
            unsafe_allow_html=True,
        )
        st.caption("Optional deep dives only. Browsing the entire case file is free.")
        st.markdown("<div class='side-rule'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <p class="side-label">CASE PROTOCOL</p>
            <ol class="side-steps">
              <li>Browse every scene, record, and suspect</li>
              <li>Spend up to two credits testing a lead</li>
              <li>Accuse whenever your theory is ready</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Restart case", width="stretch"):
            _restart_case()
            st.rerun()


def _render_landing(public_case: PublicCase) -> None:
    st.markdown('<p class="case-kicker">CASE FILE 07 · LUMEN MUSEUM</p>', unsafe_allow_html=True)
    st.title(public_case.title)
    st.markdown(
        '<p class="lead">The lights went out for ninety seconds. '
        "The sculpture was already gone.</p>",
        unsafe_allow_html=True,
    )
    st.image(_asset("crime-scene.webp"), width="stretch")

    brief_column, rules_column = st.columns([1.55, 0.9], gap="large")
    with brief_column:
        st.markdown("## The incident")
        st.write(public_case.brief)
    with rules_column:
        st.markdown(
            """
            <div class="dossier-card">
              <p class="card-label">YOUR CONSTRAINT</p>
              <div class="big-number">02</div>
              <p>optional deep dives. The map, evidence, timeline, and suspect files stay open.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Persons of interest")
    st.caption("Each had access, knowledge, or equipment that mattered that night.")
    _render_suspect_roster(public_case)

    action_column, note_column = st.columns([0.72, 1.28], vertical_alignment="center")
    with action_column:
        if st.button("Open the case file", type="primary", width="stretch"):
            _start_game()
    with note_column:
        st.caption(
            "Read everything first. The dossier can be solved as written; credits let you "
            "challenge an account or cross-check one narrow lead."
        )
    _render_ui_error()


def _render_suspect_roster(public_case: PublicCase) -> None:
    columns = st.columns(4, gap="medium")
    for column, suspect in zip(columns, public_case.suspects, strict=True):
        with column:
            st.image(_asset(SUSPECT_ASSETS[suspect.suspect_id]), width="stretch")
            st.markdown(f"### {suspect.name}")
            st.caption(suspect.role.upper())
            st.write(suspect.public_profile)


def _start_game() -> None:
    try:
        runtime = _runtime()
        with st.spinner("Unsealing the case file…"):
            st.session_state.game_result = runtime.start(
                thread_id=st.session_state.thread_id,
            )
    except ConfigurationError as exc:
        st.session_state.ui_error = str(exc)
        return
    except Exception:
        LOGGER.exception("Could not start the player game")
        st.session_state.ui_error = (
            "The case file could not be opened. Check the connection and retry."
        )
        return
    st.session_state.ui_error = None
    st.rerun()


def _render_case(
    public_case: PublicCase,
    store: CaseStore,
    result: dict[str, object],
) -> None:
    state = cast(PlayerGameState, result)
    interrupt_payload = get_game_interrupt(result)
    phase = "complete" if interrupt_payload is None else str(interrupt_payload.get("phase", ""))

    st.markdown('<p class="case-kicker">ACTIVE CASE · GALLERY WING</p>', unsafe_allow_html=True)
    st.title(public_case.title)
    _render_status_strip(state, phase)
    _render_ui_error()

    if phase == "complete":
        _render_outcome(public_case, store, state)
        return

    _render_case_browser(public_case, state)
    st.markdown("<div class='case-divider'></div>", unsafe_allow_html=True)

    investigation_column, context_column = st.columns([1.35, 0.65], gap="large")
    with investigation_column:
        if interrupt_payload is None:
            st.error("The case paused without a player prompt. Restart the case to continue.")
        elif phase in {"free_form", "free_form_clarification"}:
            _render_free_form(interrupt_payload)
        elif phase == "accusation":
            _render_accusation(public_case, interrupt_payload)
        else:
            st.error("The current case step is unavailable. Restart the case to continue.")
    with context_column:
        _render_latest_deep_dive(state)
        with st.expander("Open an optional hint"):
            st.caption("Opening this hint does not spend a deep-dive credit.")
            st.write(
                "Separate the time of removal from the time of discovery. Then connect "
                "opportunity, the changed equipment weight, and what was due to happen at 7 a.m."
            )


def _render_status_strip(state: PlayerGameState, phase: str) -> None:
    remaining = max(0, 2 - state["investigation_count"])
    mode = "VERDICT" if phase == "complete" else "ACCUSATION" if phase == "accusation" else "OPEN"
    st.markdown(
        f"""
        <div class="status-strip">
          <div><span>CASE FILE</span><strong>OPEN</strong></div>
          <div><span>DEEP-DIVE CREDITS</span><strong>{remaining} / 2</strong></div>
          <div><span>CURRENT DESK</span><strong>{mode}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_case_browser(public_case: PublicCase, state: PlayerGameState) -> None:
    deep_dive_count = max(
        0,
        len(state["discovered_evidence"]) - len(public_case.case_file_evidence_ids),
    )
    scene_tab, evidence_tab, suspects_tab, timeline_tab = st.tabs(
        [
            "Scene & map",
            f"Evidence · {len(public_case.case_file_evidence_ids)} + {deep_dive_count}",
            "People",
            "Timeline",
        ]
    )
    with scene_tab:
        scene_column, brief_column = st.columns([1.35, 0.65], gap="large")
        with scene_column:
            st.image(_asset("crime-scene.webp"), width="stretch")
        with brief_column:
            st.markdown("### Incident brief")
            st.write(public_case.brief)
            st.markdown(
                f"**Missing work:** {public_case.artwork.name}  \n"
                f"**Mass:** {public_case.artwork.mass_kg:.2f} kg  \n"
                "**Wing status:** sealed after discovery"
            )
        st.markdown("### Museum floor")
        st.caption("Browse every marked room. Looking costs no credits.")
        st.image(_asset("museum-map.svg"), width="stretch")
        _render_room_browser(public_case)
    with evidence_tab:
        _render_evidence_board(public_case, state)
    with suspects_tab:
        st.caption(
            "Public profiles are not testimony. Compare what each person could do with "
            "where the records place them."
        )
        _render_suspect_roster(public_case)
    with timeline_tab:
        timeline_column, events_column = st.columns([0.8, 1.2], gap="large")
        with timeline_column:
            st.image(_asset("timeline.webp"), width="stretch")
        with events_column:
            st.markdown("### Known sequence")
            for event in sorted(public_case.timeline_events, key=lambda item: item.starts_at):
                st.markdown(
                    f"**{_format_case_time(event.starts_at, public_case.display_timezone)}**  \n"
                    f"{event.summary}"
                )


def _render_room_browser(public_case: PublicCase) -> None:
    location_ids = {location.location_id for location in public_case.locations}
    selected_room_id = str(st.session_state.selected_room_id)
    if selected_room_id not in location_ids:
        selected_room_id = public_case.locations[0].location_id
        st.session_state.selected_room_id = selected_room_id

    st.markdown("#### Select a room")
    room_columns = st.columns(len(public_case.locations), gap="small")
    for index, (column, location) in enumerate(
        zip(room_columns, public_case.locations, strict=True),
        start=1,
    ):
        with column:
            if st.button(
                f"{index:02d} · {location.name}",
                key=f"browse-room-{location.location_id}",
                type="primary" if location.location_id == selected_room_id else "secondary",
                width="stretch",
            ):
                st.session_state.selected_room_id = location.location_id
                st.rerun()

    selected_location = next(
        location for location in public_case.locations if location.location_id == selected_room_id
    )
    room_image, room_file = st.columns([1.15, 0.85], gap="large")
    with room_image:
        asset_name = ROOM_ASSETS.get(selected_room_id)
        if asset_name is not None:
            st.image(_asset(asset_name), width="stretch")
    with room_file:
        st.caption("FREE ROOM BROWSE · NO CREDIT SPENT")
        st.markdown(f"### {selected_location.name}")
        st.write(selected_location.description)
        room_note = ROOM_NOTES.get(selected_room_id)
        if room_note is not None:
            st.write(room_note)
    _render_room_records(public_case, selected_room_id)


def _render_room_records(public_case: PublicCase, location_id: str) -> None:
    record_ids = ROOM_RECORD_IDS.get(location_id, ())
    records_by_id = {record.evidence_id: record for record in public_case.observations}
    st.markdown("#### Records tied to this room")
    for row_start in range(0, len(record_ids), 2):
        row = record_ids[row_start : row_start + 2]
        record_columns = st.columns(2, gap="medium")
        for column, evidence_id in zip(record_columns[: len(row)], row, strict=True):
            record = records_by_id[evidence_id]
            with column, st.container(border=True):
                st.caption(evidence_id)
                st.markdown(f"**{record.title}**")
                st.write(record.text)


def _format_case_time(timestamp: object, timezone_name: str) -> str:
    if not hasattr(timestamp, "astimezone"):
        return "Time unavailable"
    local = timestamp.astimezone(ZoneInfo(timezone_name))
    rendered = local.strftime("%a, %b %d · %I:%M:%S %p")
    return rendered.replace("· 0", "· ")


def _render_free_form(payload: Mapping[str, object]) -> None:
    is_clarification = payload.get("phase") == "free_form_clarification"
    remaining = int(payload.get("investigations_remaining", 0))
    st.markdown("## Optional deep dive")
    if is_clarification:
        st.warning(str(payload.get("prompt", "Narrow the lead.")))
    else:
        st.write(
            "The open dossier is enough to solve the case. Spend a credit only when you want "
            "to challenge an account, ask about someone else, or reconcile one specific record."
        )
    _render_payload_error(payload)

    suggestions = payload.get("suggestions", [])
    if is_clarification and isinstance(suggestions, list) and suggestions:
        st.caption("Possible directions")
        for raw_suggestion in suggestions:
            if not isinstance(raw_suggestion, Mapping):
                continue
            title = str(raw_suggestion.get("title", "Follow this lead"))
            description = str(raw_suggestion.get("description", ""))
            if st.button(title, key=f"suggestion-{raw_suggestion.get('action_id', title)}"):
                _resume_game(
                    {"request": f"{title}. {description}"},
                    "Following the lead…",
                )

    if remaining > 0:
        with st.form("free-form-investigation"):
            request = st.text_area(
                "What do you want to dig into?",
                placeholder=(
                    "Example: Ask Theo who heard his warning about the controller, "
                    "or challenge Rowan with the reflected service panel."
                ),
                height=130,
            )
            submitted = st.form_submit_button(
                "Spend 1 deep-dive credit",
                type="primary",
                width="stretch",
            )
        if submitted:
            _resume_game({"request": request}, "Following the lead…")
    else:
        st.info("Both deep-dive credits are spent. The complete dossier remains open above.")

    if st.button("Make accusation now", key="accuse-now", width="stretch"):
        _resume_game({"next_step": "accuse"}, "Opening the accusation file…")


def _render_accusation(
    public_case: PublicCase,
    payload: Mapping[str, object],
) -> None:
    st.markdown("## Make your accusation")
    st.write(
        "Name the culprit, then state why they wanted Aurora Circuit and how they removed it. "
        "The case closes when the culprit is right and either your motive or method fits the facts."
    )
    _render_payload_error(payload)

    portrait_columns = st.columns(4, gap="small")
    for column, suspect in zip(portrait_columns, public_case.suspects, strict=True):
        with column:
            st.image(_asset(SUSPECT_ASSETS[suspect.suspect_id]), width="stretch")
            st.caption(f"{suspect.name} · {suspect.role}")

    suspect_names = {suspect.suspect_id: suspect.name for suspect in public_case.suspects}
    suspect_ids = list(suspect_names)
    with st.form("final-accusation"):
        suspect_id = st.selectbox(
            "Who took Aurora Circuit?",
            suspect_ids,
            format_func=suspect_names.__getitem__,
            index=None,
            placeholder="Select a suspect",
        )
        motive = st.text_area(
            "Why did they do it?",
            placeholder="State the motive in your own words.",
            height=110,
        )
        method = st.text_area(
            "How did they do it?",
            placeholder="Reconstruct the theft from access to concealment or escape.",
            height=140,
        )
        submitted = st.form_submit_button(
            "Lock the accusation",
            type="primary",
            width="stretch",
        )
    if submitted:
        _resume_game(
            {
                "suspect_id": suspect_id or "",
                "motive": motive,
                "method": method,
            },
            "Testing the accusation against the case…",
        )


def _render_evidence_board(public_case: PublicCase, state: PlayerGameState) -> None:
    evidence = state["discovered_evidence"]
    case_file_ids = set(public_case.case_file_evidence_ids)
    case_file = [record for record in evidence if record["evidence_id"] in case_file_ids]
    deep_dives = [record for record in evidence if record["evidence_id"] not in case_file_ids]

    st.markdown("### Open dossier")
    st.caption(
        "All twelve records were available when the wing was sealed. Titles are descriptive, "
        "not conclusions; the comparisons are yours to make."
    )
    _render_evidence_grid(case_file, "DOSSIER")
    if deep_dives:
        st.markdown("### Deep-dive notes")
        st.caption("These follow-up findings came from the credits you chose to spend.")
        _render_evidence_grid(deep_dives, "FOLLOW-UP")


def _render_evidence_grid(
    evidence: list[Mapping[str, object]],
    label: str,
) -> None:
    for row_start in range(0, len(evidence), 3):
        row = evidence[row_start : row_start + 3]
        columns = st.columns(3, gap="medium")
        for offset, (column, record) in enumerate(
            zip(columns[: len(row)], row, strict=True),
            start=1,
        ):
            with column, st.container(border=True):
                evidence_id = str(record.get("evidence_id", ""))
                filename = EVIDENCE_ASSETS.get(evidence_id)
                if filename is not None:
                    st.image(_asset(filename), width="stretch")
                st.caption(f"{label} {row_start + offset:02d}")
                st.markdown(f"**{record.get('title', 'Recovered evidence')}**")
                st.write(str(record.get("text", "")))


def _render_latest_deep_dive(state: PlayerGameState) -> None:
    if not state["completed_action_ids"]:
        with st.container(border=True):
            st.markdown("### No credit spent")
            st.write(
                "That is a valid strategy. Read the dossier, form a theory, and accuse without "
                "opening any follow-up if the records already convince you."
            )
        return
    action = get_game_action(state["completed_action_ids"][-1])
    observation = state["last_observation"]
    summary = ""
    if isinstance(observation, Mapping):
        summary = str(observation.get("summary", "")).split("\n\n", maxsplit=1)[0]
    with st.container(border=True):
        st.caption("LATEST DEEP DIVE")
        st.markdown(f"### {action.title}")
        st.write(summary or "The follow-up note is pinned in the evidence tab.")


def _render_outcome(
    public_case: PublicCase,
    store: CaseStore,
    state: PlayerGameState,
) -> None:
    result = GameResult.model_validate(state["result"])
    debrief = GameDebrief.model_validate(state["debrief"])
    solved = result.tier == "solved"
    outcome_class = "solved" if solved else "partial" if result.tier == "partial" else "failed"
    outcome_label = "CASE SOLVED" if solved else "THE CASE DOESN'T HOLD"
    st.markdown(
        f"""
        <div class="outcome {outcome_class}">
          <p>{outcome_label}</p>
          <h2>{debrief.headline}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_match_cards(result)
    st.markdown("## Closing analysis")
    st.write(debrief.summary)
    for item in debrief.evidence_analysis:
        st.markdown(f"- {item}")
    st.markdown(f"*{debrief.closing_line}*")
    _render_reconstruction(public_case, store, state["case_id"])
    with st.expander("Review the complete case file"):
        _render_evidence_board(public_case, state)

    if st.button("Investigate again", type="primary"):
        _restart_case()
        st.rerun()


def _render_match_cards(result: GameResult) -> None:
    checks = (
        ("CULPRIT", result.culprit_correct),
        ("MOTIVE", result.motive_match),
        ("METHOD", result.method_match),
    )
    columns = st.columns(3, gap="medium")
    for column, (label, matched) in zip(columns, checks, strict=True):
        with column:
            status = "MATCH" if matched else "MISS"
            symbol = "✓" if matched else "X"
            st.markdown(
                f"""
                <div class="match-card {"hit" if matched else "miss"}">
                  <span>{symbol}</span><p>{label}</p><strong>{status}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_reconstruction(
    public_case: PublicCase,
    store: CaseStore,
    case_id: str,
) -> None:
    st.markdown("### Complete reconstruction")
    if not st.session_state.solution_revealed:
        st.caption("Open the sealed answer only when you are ready to see the entire sequence.")
        if st.button("Open the sealed reconstruction", key="reveal-solution"):
            st.session_state.solution_revealed = True
            st.rerun()
        return
    solution = store.load_solution(case_id)
    suspect_names = {suspect.suspect_id: suspect.name for suspect in public_case.suspects}
    with st.container(border=True):
        st.markdown(f"### {suspect_names[solution.culprit_id]}")
        for index, event in enumerate(solution.canonical_sequence, start=1):
            st.markdown(f"**{index:02d}** &nbsp; {event}", unsafe_allow_html=True)


def _render_payload_error(payload: Mapping[str, object]) -> None:
    error = payload.get("error")
    if isinstance(error, str) and error:
        st.error(error)


def _render_ui_error() -> None:
    error = st.session_state.ui_error
    if isinstance(error, str) and error:
        st.error(error)


def _resume_game(payload: Mapping[str, object], spinner_text: str) -> None:
    try:
        with st.spinner(spinner_text):
            st.session_state.game_result = _runtime().resume(
                st.session_state.thread_id,
                payload,
            )
    except Exception:
        LOGGER.exception("Could not resume the player game")
        st.session_state.ui_error = (
            "That lead could not be processed. Check the connection, then restart the case."
        )
        return
    st.session_state.ui_error = None
    st.rerun()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #0a0f0d;
          --forest: #10251f;
          --forest-light: #1c3a31;
          --cream: #efe7d5;
          --paper: #d8ceb9;
          --muted: #a69d8a;
          --oxblood: #8f302d;
          --gold: #c9aa6b;
        }
        .stApp {
          background:
            radial-gradient(circle at 82% 8%, rgba(50, 83, 68, .22), transparent 32rem),
            linear-gradient(180deg, #0c1310 0%, #080d0b 100%);
          color: var(--cream);
        }
        .block-container { max-width: 1380px; padding-top: 2.5rem; padding-bottom: 6rem; }
        [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { color: var(--cream) !important; }
        h1 {
          font-family: Georgia, 'Times New Roman', serif !important;
          font-size: clamp(2.8rem, 6vw, 5.8rem) !important;
          line-height: .94 !important;
          max-width: 1000px;
          letter-spacing: -.045em;
        }
        h2 { font-family: Georgia, 'Times New Roman', serif !important; letter-spacing: -.02em; }
        p, li, label, [data-testid="stCaptionContainer"] { color: var(--paper); }
        [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
        [data-testid="stImage"] img {
          border-radius: 4px;
          box-shadow: 0 20px 55px rgba(0, 0, 0, .26);
        }
        .case-kicker, .board-kicker, .card-label, .side-kicker, .side-label {
          color: var(--gold) !important;
          font-size: .7rem;
          font-weight: 800;
          letter-spacing: .19em;
          margin: 0 0 .85rem;
        }
        .lead {
          color: var(--paper) !important;
          font-family: Georgia, 'Times New Roman', serif;
          font-size: clamp(1.25rem, 2.2vw, 1.8rem);
          margin: .6rem 0 2rem;
          max-width: 820px;
        }
        .dossier-card {
          border: 1px solid rgba(201, 170, 107, .45);
          background: rgba(20, 41, 34, .72);
          padding: 1.4rem 1.5rem;
          min-height: 220px;
        }
        .dossier-card .big-number {
          color: var(--cream);
          font-family: Georgia, 'Times New Roman', serif;
          font-size: 5rem;
          line-height: 1;
        }
        .status-strip {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          border-top: 1px solid rgba(216, 206, 185, .2);
          border-bottom: 1px solid rgba(216, 206, 185, .2);
          margin: 2rem 0 2.4rem;
        }
        .status-strip div {
          display: flex;
          flex-direction: column;
          gap: .25rem;
          padding: .9rem .25rem;
        }
        .status-strip span {
          color: var(--muted);
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .09em;
          text-transform: uppercase;
        }
        .status-strip strong { color: var(--cream); }
        .case-divider {
          height: 1px;
          background: rgba(216, 206, 185, .2);
          margin: 2.7rem 0;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
          gap: .45rem;
          border-bottom: 1px solid rgba(216, 206, 185, .2);
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
          color: var(--paper);
          padding-left: 1rem;
          padding-right: 1rem;
        }
        [data-testid="stBaseButton-primary"] {
          background: var(--oxblood) !important;
          border: 1px solid #a9403b !important;
          color: white !important;
          min-height: 3.25rem;
          border-radius: 2px !important;
          font-weight: 800 !important;
          letter-spacing: .04em;
        }
        [data-testid="stBaseButton-secondary"] {
          background: rgba(23, 46, 38, .78) !important;
          border: 1px solid rgba(201, 170, 107, .42) !important;
          color: var(--cream) !important;
          border-radius: 2px !important;
          min-height: 2.8rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: rgba(18, 32, 27, .68);
          border-color: rgba(201, 170, 107, .25) !important;
          border-radius: 2px !important;
        }
        .empty-board {
          border: 1px dashed rgba(201, 170, 107, .28);
          padding: 3.2rem 1rem;
          text-align: center;
          color: var(--muted);
        }
        .empty-board span { color: var(--gold); font-size: 2rem; }
        .outcome {
          border: 1px solid rgba(201, 170, 107, .4);
          padding: clamp(1.5rem, 4vw, 3rem);
          margin: 1.2rem 0 1.5rem;
          background: linear-gradient(120deg, rgba(23, 58, 47, .85), rgba(10, 15, 13, .6));
        }
        .outcome.partial, .outcome.failed {
          background: linear-gradient(120deg, rgba(81, 31, 29, .72), rgba(10, 15, 13, .65));
        }
        .outcome p {
          color: var(--gold); font-size: .72rem; font-weight: 900; letter-spacing: .2em;
        }
        .outcome h2 { font-size: clamp(2rem, 4vw, 3.5rem) !important; margin: .4rem 0 0; }
        .match-card {
          border: 1px solid rgba(216, 206, 185, .2);
          padding: 1rem 1.2rem;
          display: grid;
          grid-template-columns: auto 1fr;
          column-gap: .9rem;
          background: rgba(18, 32, 27, .72);
        }
        .match-card span { grid-row: 1 / 3; font-size: 2rem; color: #6fa17f; }
        .match-card.miss span { color: #b65b54; }
        .match-card p { margin: 0; font-size: .65rem; letter-spacing: .14em; color: var(--muted); }
        .match-card strong { color: var(--cream); }
        [data-testid="stSidebar"] { background: #10251f; border-right: 1px solid #2c493e; }
        [data-testid="stSidebar"] .stMarkdown h3 { font-family: Georgia, serif; line-height: 1.1; }
        .side-mark {
          align-items: center;
          border: 1px solid var(--gold);
          color: var(--gold);
          display: flex;
          font-family: Georgia, serif;
          font-size: 1.2rem;
          height: 3.1rem;
          justify-content: center;
          margin-bottom: 1rem;
          width: 3.1rem;
        }
        .side-rule { height: 1px; background: rgba(216, 206, 185, .18); margin: 1.4rem 0; }
        .credit-row { display: flex; align-items: center; gap: .55rem; margin-bottom: .75rem; }
        .credit { width: 2.5rem; height: .36rem; background: #3c4e46; display: inline-block; }
        .credit.available { background: var(--gold); }
        .credit-row strong { color: var(--cream); margin-left: auto; }
        .side-steps { padding-left: 1.25rem; }
        .side-steps li { margin-bottom: .65rem; color: var(--paper); }
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
          background: #111b17 !important;
          border-color: #44594f !important;
          color: var(--cream) !important;
        }
        [data-baseweb="select"] * { color: var(--cream) !important; }
        [data-testid="stExpander"] { border-color: rgba(201, 170, 107, .3) !important; }
        @media (max-width: 900px) {
          .block-container { padding-top: 4rem; }
          h1 { font-size: 3.1rem !important; }
          .status-strip { grid-template-columns: 1fr; }
          [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
          [data-testid="stColumn"] { flex: 1 1 100% !important; width: 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_app()
