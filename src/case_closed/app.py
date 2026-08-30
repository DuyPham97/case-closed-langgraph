"""Immersive Streamlit interface for Midnight at the Lumen Museum."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import uuid4

import streamlit as st

from case_closed.case_store import CaseStore
from case_closed.config import ConfigurationError, load_config
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
    "E10": "media-locker.webp",
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


def _new_thread_id() -> str:
    return f"case-{uuid4().hex[:12]}"


def _asset(filename: str) -> str:
    return str(ASSET_ROOT / filename)


def _restart_case() -> None:
    st.session_state.thread_id = _new_thread_id()
    st.session_state.game_result = None
    st.session_state.ui_error = None
    st.session_state.solution_revealed = False


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
        st.caption("One location. One line of inquiry. Then you accuse.")
        st.markdown("<div class='side-rule'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <p class="side-label">CASE PROTOCOL</p>
            <ol class="side-steps">
              <li>Search one location</li>
              <li>Follow one lead</li>
              <li>Name who, why, and how</li>
            </ol>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Restart case", use_container_width=True):
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
    st.image(_asset("crime-scene.webp"), use_container_width=True)

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
              <p>investigations before the museum reopens and the trail goes cold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Persons of interest")
    st.caption("Each had access, knowledge, or equipment that mattered that night.")
    _render_suspect_roster(public_case)

    action_column, note_column = st.columns([0.72, 1.28], vertical_alignment="center")
    with action_column:
        if st.button("Enter Gallery 3", type="primary", use_container_width=True):
            _start_game()
    with note_column:
        st.caption(
            "Your first move is visual. Your second can be written in your own words. "
            "After that, the accusation is yours."
        )
    _render_ui_error()


def _render_suspect_roster(public_case: PublicCase) -> None:
    columns = st.columns(4, gap="medium")
    for column, suspect in zip(columns, public_case.suspects, strict=True):
        with column:
            st.image(_asset(SUSPECT_ASSETS[suspect.suspect_id]), use_container_width=True)
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
    _render_stage_bar(phase)
    _render_ui_error()

    if phase == "complete":
        _render_outcome(public_case, store, state)
        return

    investigation_column, evidence_column = st.columns([1.45, 0.8], gap="large")
    with investigation_column:
        if interrupt_payload is None:
            st.error("The case paused without a player prompt. Restart the case to continue.")
        elif phase == "visual_choice":
            _render_visual_choice(interrupt_payload)
        elif phase in {"free_form", "free_form_clarification"}:
            _render_free_form(interrupt_payload)
        elif phase == "accusation":
            _render_accusation(public_case, interrupt_payload)
        else:
            st.error("The current case step is unavailable. Restart the case to continue.")
    with evidence_column:
        _render_evidence_board(state)


def _render_stage_bar(phase: str) -> None:
    active_index = {
        "visual_choice": 0,
        "free_form": 1,
        "free_form_clarification": 1,
        "accusation": 2,
        "complete": 3,
    }.get(phase, 0)
    labels = ("Search", "Follow a lead", "Accuse", "Verdict")
    items_list: list[str] = []
    for index, label in enumerate(labels):
        stage_class = "active" if index == active_index else "done" if index < active_index else ""
        items_list.append(
            f'<div class="stage {stage_class}"><span>0{index + 1}</span>{label}</div>'
        )
    items = "".join(items_list)
    st.markdown(f'<div class="stage-bar">{items}</div>', unsafe_allow_html=True)


def _render_visual_choice(payload: Mapping[str, object]) -> None:
    st.markdown("## Choose your first search")
    st.write(
        "You may inspect one marked location. Choose carefully; the museum will grant one search."
    )
    st.image(_asset("museum-map.svg"), use_container_width=True)
    _render_payload_error(payload)

    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        return
    rows = [actions[index : index + 2] for index in range(0, len(actions), 2)]
    for row in rows:
        columns = st.columns(2, gap="medium")
        for column, raw_action in zip(columns, row, strict=True):
            if not isinstance(raw_action, Mapping):
                continue
            with column:
                image_path = raw_action.get("image_path")
                if isinstance(image_path, str):
                    st.image(str(PROJECT_ROOT / image_path), use_container_width=True)
                title = str(raw_action.get("title", "Search location"))
                st.markdown(f"### {title}")
                st.write(str(raw_action.get("description", "")))
                action_id = str(raw_action.get("action_id", ""))
                if st.button(
                    title,
                    key=f"visual-{action_id}",
                    use_container_width=True,
                ):
                    _resume_game({"action_id": action_id}, "Searching the location…")


def _render_free_form(payload: Mapping[str, object]) -> None:
    is_clarification = payload.get("phase") == "free_form_clarification"
    st.markdown("## Your final investigation")
    if is_clarification:
        st.warning(str(payload.get("prompt", "Narrow the lead.")))
    else:
        st.write(
            "Ask a suspect, inspect another location, or compare events. Describe the lead in "
            "your own words; only one investigation will be carried out."
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

    with st.form("free-form-investigation"):
        request = st.text_area(
            "What do you investigate?",
            placeholder=(
                "Example: Compare the security records with Rowan's equipment case, "
                "or ask Nia how staff gear was screened."
            ),
            height=130,
        )
        submitted = st.form_submit_button(
            "Commit final investigation",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        _resume_game({"request": request}, "Following the lead…")


def _render_accusation(
    public_case: PublicCase,
    payload: Mapping[str, object],
) -> None:
    st.markdown("## Make your accusation")
    st.write(
        "Name the culprit, then state why they wanted Aurora Circuit and how they removed it. "
        "The case holds if the culprit is right and either your motive or method fits the evidence."
    )
    _render_payload_error(payload)

    portrait_columns = st.columns(4, gap="small")
    for column, suspect in zip(portrait_columns, public_case.suspects, strict=True):
        with column:
            st.image(_asset(SUSPECT_ASSETS[suspect.suspect_id]), use_container_width=True)
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
            use_container_width=True,
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


def _render_evidence_board(state: PlayerGameState) -> None:
    st.markdown('<p class="board-kicker">PINNED EVIDENCE</p>', unsafe_allow_html=True)
    evidence = state["discovered_evidence"]
    if not evidence:
        st.markdown(
            """
            <div class="empty-board">
              <span>◇</span>
              <p>The board is empty.<br>Your searches will pin what they uncover.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    for index, record in enumerate(evidence, start=1):
        with st.container(border=True):
            evidence_id = str(record.get("evidence_id", ""))
            filename = EVIDENCE_ASSETS.get(evidence_id)
            if filename is not None:
                st.image(_asset(filename), use_container_width=True)
            st.caption(f"CLUE {index:02d}")
            st.markdown(f"**{record.get('title', 'Recovered evidence')}**")
            st.write(str(record.get("text", "")))


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
    analysis_column, evidence_column = st.columns([1.25, 0.75], gap="large")
    with analysis_column:
        st.markdown("## Closing analysis")
        st.write(debrief.summary)
        for item in debrief.evidence_analysis:
            st.markdown(f"- {item}")
        st.markdown(f"*{debrief.closing_line}*")
        _render_reconstruction(public_case, store, state["case_id"])
    with evidence_column:
        _render_evidence_board(state)

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
        .stage-bar {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          border-top: 1px solid rgba(216, 206, 185, .2);
          border-bottom: 1px solid rgba(216, 206, 185, .2);
          margin: 2rem 0 2.4rem;
        }
        .stage {
          color: #5f685f;
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .09em;
          padding: .9rem .25rem;
          text-transform: uppercase;
        }
        .stage span { margin-right: .55rem; color: #59665f; }
        .stage.active { color: var(--cream); border-bottom: 2px solid var(--oxblood); }
        .stage.active span { color: var(--gold); }
        .stage.done, .stage.done span { color: #789385; }
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
          .stage { font-size: .58rem; letter-spacing: .03em; }
          .stage span { display: block; margin-bottom: .15rem; }
          [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
          [data-testid="stColumn"] { flex: 1 1 100% !important; width: 100% !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_app()
