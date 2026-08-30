"""Streamlit caseboard for the Case Closed portfolio demo."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast
from uuid import uuid4

import streamlit as st

from case_closed.case_store import CaseStore
from case_closed.config import ConfigurationError, load_config
from case_closed.runtime import InvestigationRuntime, create_runtime, get_interrupt_payload
from case_closed.schemas import PublicCase, Verdict
from case_closed.state import CaseState, create_initial_state


def render_app() -> None:
    """Render and operate the one-page Streamlit investigation caseboard."""
    st.set_page_config(
        page_title="Case Closed?",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    store = CaseStore()
    public_case = store.load_public_case()
    _initialize_session()

    st.markdown(
        '<p class="eyebrow">LANGCHAIN + LANGGRAPH PORTFOLIO DEMO</p>',
        unsafe_allow_html=True,
    )
    st.title("Case Closed?")
    st.subheader(public_case.title)
    st.caption(
        "A bounded Claude investigator uses deterministic tools, evidence-grounded state, "
        "conditional routing, and human-in-the-loop checkpoint resume."
    )

    with st.sidebar:
        st.markdown("### Investigation controls")
        st.code(st.session_state.thread_id, language=None)
        st.caption("This ID points to the SQLite checkpoint thread.")
        if st.button("Start a fresh case", use_container_width=True):
            st.session_state.thread_id = _new_thread_id()
            st.session_state.result = None
            st.rerun()
        st.divider()
        st.markdown("**Graph guarantees**")
        st.markdown(
            "- Maximum six tool rounds\n"
            "- Two invalid-action retries\n"
            "- One verdict repair\n"
            "- Evidence citations validated\n"
            "- Private answer excluded from state"
        )
        st.caption("Model: Claude Haiku 4.5 (configurable)")

    brief_column, board_column, evidence_column = st.columns([1.0, 1.25, 1.1], gap="large")
    result = cast(dict[str, object] | None, st.session_state.result)

    with brief_column:
        _render_brief(public_case)
    with board_column:
        _render_investigation(public_case, result)
    with evidence_column:
        _render_evidence(result)

    if result is not None:
        _render_trace(result)


@st.cache_resource
def _runtime() -> InvestigationRuntime:
    config = load_config(dotenv_path=".env")
    return create_runtime(config)


def _initialize_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = _new_thread_id()
    if "result" not in st.session_state:
        st.session_state.result = None


def _new_thread_id() -> str:
    return f"case-{uuid4().hex[:12]}"


def _render_brief(public_case: PublicCase) -> None:
    st.markdown("### Case brief")
    st.write(public_case.brief)
    st.markdown("### Suspects")
    for suspect in public_case.suspects:
        with st.container(border=True):
            st.markdown(f"**{suspect.name}** · {suspect.role}")
            st.caption(suspect.public_profile)


def _render_investigation(
    public_case: PublicCase,
    result: dict[str, object] | None,
) -> None:
    st.markdown("### Investigation")
    if result is None:
        st.info(
            "Claude will select public actions, while LangGraph owns the loop, limits, "
            "checkpoint, pause, and terminal state."
        )
        if st.button("Begin investigation", type="primary", use_container_width=True):
            try:
                runtime = _runtime()
            except ConfigurationError as exc:
                st.error(str(exc))
                return
            initial_state = create_initial_state(public_case, human_review_enabled=True)
            with st.spinner("The graph is investigating…"):
                st.session_state.result = runtime.start(
                    initial_state,
                    st.session_state.thread_id,
                )
            st.rerun()
        return

    state = cast(CaseState, result)
    status = state["status"].replace("_", " ").title()
    metric_status, metric_rounds = st.columns(2)
    metric_status.metric("Status", status)
    metric_rounds.metric("Tool rounds", f"{state['round_count']} / {state['max_rounds']}")

    interrupt_payload = get_interrupt_payload(result)
    if interrupt_payload is not None:
        _render_interrupt(interrupt_payload)
        return
    if state["status"] == "resolved" and state["proposed_verdict"] is not None:
        _render_verdict(state, public_case)
        return
    if state["status"] == "inconclusive":
        st.warning(state["final_report"] or "The graph stopped without a supported verdict.")


def _render_interrupt(payload: Mapping[str, object]) -> None:
    st.warning("Human-in-the-loop interrupt reached. The checkpoint is saved.")
    st.write(payload.get("question", "Which lead should come next?"))
    leads = payload.get("suggested_leads", [])
    if isinstance(leads, list):
        for lead in leads:
            st.markdown(f"- {lead}")
    with st.form("direction-form"):
        direction = st.text_input(
            "Direction",
            value="Inspect the strongest remaining physical or timeline lead.",
        )
        submitted = st.form_submit_button("Resume from checkpoint", type="primary")
    if submitted:
        with st.spinner("Resuming the same graph thread…"):
            st.session_state.result = _runtime().resume(
                st.session_state.thread_id,
                direction,
            )
        st.rerun()


def _render_verdict(state: CaseState, public_case: PublicCase) -> None:
    verdict = Verdict.model_validate(state["proposed_verdict"])
    suspect_names = {suspect.suspect_id: suspect.name for suspect in public_case.suspects}
    st.success("The deterministic validator accepted the evidence path and citations.")
    st.markdown(f"## {suspect_names.get(verdict.culprit_id, verdict.culprit_id)}")
    st.caption(f"Confidence {verdict.confidence}% · Citations {', '.join(verdict.citations)}")
    st.write(verdict.summary)
    st.write(verdict.case_theory)


def _render_evidence(result: dict[str, object] | None) -> None:
    st.markdown("### Evidence board")
    if result is None:
        st.caption("Source-backed clue cards appear here as tools discover them.")
        return
    state = cast(CaseState, result)
    if not state["discovered_evidence"]:
        st.caption("No evidence discovered yet.")
        return
    for evidence in state["discovered_evidence"]:
        with st.container(border=True):
            st.markdown(f"**{evidence['evidence_id']} · {evidence.get('title', 'Evidence')}**")
            st.write(evidence.get("text", ""))
            st.caption(evidence.get("source_reference", ""))


def _render_trace(result: dict[str, object]) -> None:
    state = cast(CaseState, result)
    with st.expander("Public graph execution trace", expanded=False):
        st.caption("Node outcomes and public decision summaries only; no hidden chain-of-thought.")
        for event in state["trace"]:
            st.markdown(
                f"`{event['node']}` **{event['event'].replace('_', ' ')}**  \n"
                f"{event.get('message', '')}"
            )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f0e6; color: #17211c; }
        .block-container { padding-top: 2.2rem; max-width: 1480px; }
        .eyebrow { color: #9a3c2f; font-size: .72rem; font-weight: 800;
                   letter-spacing: .18em; margin-bottom: -.5rem; }
        h1, h2, h3 { color: #173f35 !important; }
        [data-testid="stMetric"] { background: #fffaf0; border: 1px solid #d8cfbc;
                                   border-radius: .75rem; padding: .8rem; }
        [data-testid="stSidebar"] { background: #173f35; }
        [data-testid="stSidebar"] * { color: #f6f0df !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_app()
