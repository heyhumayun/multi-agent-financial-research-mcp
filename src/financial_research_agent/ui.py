from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

from financial_research_agent.agents import SupervisorAgent
from financial_research_agent.config import load_settings
from financial_research_agent.serialization import to_jsonable

DEFAULT_QUERY = "Assess NVDA AI infrastructure earnings risk with research papers and document framework"
EXAMPLE_QUERIES = [
    ("NVDA risk", "Assess NVDA downside risk and current news tension"),
    ("Compare", "Compare NVDA and AMD volatility and downside risk"),
    ("Fundamentals", "Assess AAPL company fundamentals and earnings risk"),
    ("Research", "Find papers about graph neural networks for financial market prediction"),
]


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #d9e0ea;
            --blue: #2457a6;
            --green: #16796c;
        }
        .stApp {
            background: #f7f9fc;
            color: var(--ink);
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            background: #f7f9fc;
        }
        [data-testid="stHeader"] {
            background: rgba(247, 249, 252, 0.92);
        }
        [data-testid="stSidebar"] {
            background: #eef2f7;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] label p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] button,
        [data-testid="stDownloadButton"] button {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stSidebar"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            background: #e6eef9 !important;
            border-color: #b8c9e4 !important;
            color: #1d4789 !important;
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .desk-kicker {
            color: var(--blue);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .desk-title {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.65rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.04;
            margin: 0;
        }
        .desk-subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.65rem 0 0;
            max-width: 760px;
        }
        .query-band {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--blue);
            padding: 1.1rem 1.25rem 0.35rem;
            margin: 1.5rem 0 1.15rem;
        }
        .section-label {
            color: var(--ink);
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin: 0.3rem 0 0.75rem;
        }
        .thesis-band {
            background: #eaf4f1;
            border-left: 4px solid var(--green);
            color: #164d46;
            padding: 1.1rem 1.3rem;
            margin: 0.7rem 0 1.35rem;
            font-size: 1.08rem;
            line-height: 1.55;
        }
        .status-pill {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-right: 0.35rem;
        }
        .status-green { background: #d9f2eb; color: #126354; }
        .status-amber { background: #fff0d8; color: #86510e; }
        .status-blue { background: #e1ebfb; color: #1f4e94; }
        .pipeline {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            overflow-x: auto;
            padding: 0.2rem 0 1rem;
        }
        .pipeline-step {
            min-width: 112px;
            background: #ffffff;
            border: 1px solid var(--line);
            padding: 0.7rem 0.75rem;
            text-align: center;
        }
        .pipeline-step strong {
            display: block;
            color: var(--ink);
            font-size: 0.76rem;
        }
        .pipeline-step span {
            color: var(--green);
            font-size: 0.72rem;
            font-weight: 700;
        }
        .pipeline-arrow {
            color: #8c98aa;
            font-size: 1.1rem;
        }
        .table-wrap {
            overflow-x: auto;
            margin: 0.35rem 0 1.3rem;
        }
        .desk-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid var(--line);
            color: var(--ink);
            font-size: 0.86rem;
        }
        .desk-table th {
            background: #edf2f8;
            color: var(--ink);
            font-weight: 800;
            padding: 0.65rem 0.7rem;
            text-align: left;
            white-space: nowrap;
        }
        .desk-table td {
            border-top: 1px solid var(--line);
            color: var(--ink);
            padding: 0.65rem 0.7rem;
            vertical-align: top;
            word-break: break-word;
        }
        .desk-table tr:nth-child(even) td {
            background: #f8fafc;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] div[role="group"] {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
            color: var(--ink) !important;
        }
        [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--line);
            gap: 0.2rem;
        }
        [data-baseweb="tab"],
        button[role="tab"] {
            color: var(--muted) !important;
        }
        [data-baseweb="tab"] p,
        button[role="tab"] p {
            color: var(--muted) !important;
        }
        [data-baseweb="tab"][aria-selected="true"],
        button[role="tab"][aria-selected="true"] {
            color: var(--blue) !important;
            border-bottom-color: var(--blue) !important;
        }
        [data-baseweb="tab"][aria-selected="true"] p,
        button[role="tab"][aria-selected="true"] p {
            color: var(--blue) !important;
        }
        .finding-meta {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 0.4rem;
        }
        .small-note {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.45;
        }
        [data-testid="stTextArea"] textarea,
        [data-baseweb="textarea"],
        [data-baseweb="textarea"] textarea {
            background: #ffffff !important;
            color: var(--ink) !important;
            caret-color: var(--blue) !important;
            border-color: var(--line) !important;
        }
        [data-testid="stTextArea"] textarea::placeholder {
            color: #7a8799 !important;
            opacity: 1 !important;
        }
        [data-baseweb="textarea"]:focus-within,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 1px var(--blue) !important;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            padding: 0.8rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }
        [data-testid="stFormSubmitButton"] button,
        button[kind="primary"] {
            background: var(--blue) !important;
            border-color: var(--blue) !important;
            color: #ffffff !important;
        }
        [data-testid="stFormSubmitButton"] button:hover,
        button[kind="primary"]:hover {
            background: #1d4789 !important;
            border-color: #1d4789 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _set_query(st: Any, query: str) -> None:
    st.session_state.query_input = query
    st.rerun()


def _render_sidebar(st: Any, settings: Any) -> tuple[bool, bool, bool]:
    with st.sidebar:
        st.markdown("## Signal Desk")
        st.caption("Auditable multi-agent financial research")
        st.divider()

        st.markdown("#### Run controls")
        live_mode = st.toggle(
            "Live provider mode",
            value=settings.live_data_enabled,
            help="Attempt Polygon/Finnhub/SEC/arXiv before using the configured fallback.",
        )
        fallback = st.toggle(
            "Allow offline fallback",
            value=settings.offline_fallback_enabled,
            help="Keep the research run available when a provider is unavailable.",
        )
        ollama = st.toggle(
            "Use Ollama reasoning",
            value=False,
            key="ollama_reasoning_v2",
            help="Use the local model for planning and synthesis. This can add substantial latency.",
        )

        st.divider()
        st.markdown("#### Current stack")
        runtime = settings.tool_runtime
        runtime_class = "status-blue" if runtime.startswith("mcp") else "status-amber"
        data_class = "status-green" if live_mode else "status-blue"
        reasoning_class = "status-green" if ollama else "status-blue"
        st.markdown(
            f'<span class="status-pill {runtime_class}">{runtime}</span>'
            f'<span class="status-pill {data_class}">{"live" if live_mode else "offline"}</span>'
            f'<span class="status-pill {reasoning_class}">{"ollama" if ollama else "deterministic"}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="small-note">Every run records provider provenance, evidence IDs, '
            "critic checks, and stage-level latency.</p>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("#### Example investigations")
        for label, query in EXAMPLE_QUERIES:
            if st.button(label, key=f"example_{label}", use_container_width=True):
                _set_query(st, query)

        st.divider()
        st.markdown(
            '<p class="small-note">Research assistance only. Results are not investment advice '
            "or execution instructions.</p>",
            unsafe_allow_html=True,
        )
    return live_mode, fallback, ollama


def _apply_run_settings(live_mode: bool, fallback: bool, ollama: bool) -> None:
    os.environ["FIN_RESEARCH_LIVE"] = "1" if live_mode else "0"
    os.environ["FIN_RESEARCH_OFFLINE_FALLBACK"] = "1" if fallback else "0"
    os.environ["FIN_RESEARCH_LLM"] = "ollama" if ollama else "off"


def _render_table(st: Any, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        st.caption("No records available for this run.")
        return
    headers = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="table-wrap"><table class="desk-table"><thead><tr>{headers}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_header(st: Any) -> None:
    st.markdown('<div class="desk-kicker">Auditable research workflow</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="desk-title">Financial Research Desk</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="desk-subtitle">Ask one question. The supervisor routes it across market, '
        'news, fundamentals, research, documents, risk, and a critic before producing a '
        "traceable brief.</p>",
        unsafe_allow_html=True,
    )


def _render_query_composer(st: Any) -> bool:
    st.markdown('<div class="section-label">Research question</div>', unsafe_allow_html=True)
    with st.form("research_form", clear_on_submit=False):
        st.text_area(
            "",
            key="query_input",
            height=96,
            label_visibility="collapsed",
            placeholder="Example: Compare NVDA and AMD volatility, news risk, and financial ML research",
        )
        submitted = st.form_submit_button("Run full investigation", type="primary", use_container_width=True)
    return submitted


def _render_pipeline(st: Any, summary: dict[str, Any]) -> None:
    planned = summary.get("planned_agents", [])
    stages = [
        ("Plan", bool(planned)),
        ("Specialists", bool(summary.get("selected_agents"))),
        ("Critic", "Critic Agent" in summary.get("selected_agents", [])),
        ("Ground", bool(summary.get("evidence_count"))),
        ("Synthesis", bool(summary.get("stop_reason"))),
        ("Evaluate", bool(summary.get("end_to_end_latency_ms"))),
    ]
    parts = []
    for index, (label, complete) in enumerate(stages):
        parts.append(
            f'<div class="pipeline-step"><strong>{"[x]" if complete else "[ ]"} {label}</strong>'
            f'<span>{"complete" if complete else "pending"}</span></div>'
        )
        if index < len(stages) - 1:
            parts.append('<div class="pipeline-arrow">&gt;</div>')
    st.markdown('<div class="pipeline">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def _render_overview(st: Any, report: Any, summary: dict[str, Any]) -> None:
    evaluation = report.evaluation
    grounding = evaluation.get("grounding", {})
    metric_columns = st.columns(5)
    values = [
        ("Quality", f"{evaluation.get('score', 0):.2f}"),
        ("Agents", str(len(summary.get("planned_agents", [])))),
        ("Evidence", str(len(report.evidence_registry))),
        ("Citations", f"{grounding.get('citation_coverage', 0):.0%}"),
        ("Latency", f"{summary.get('end_to_end_latency_ms', 0):.0f} ms"),
    ]
    for column, (label, value) in zip(metric_columns, values):
        column.metric(label, value)

    st.markdown('<div class="section-label">Executive view</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="thesis-band">{escape(report.thesis)}</div>',
        unsafe_allow_html=True,
    )

    for finding in report.findings:
        confidence = f"{finding.confidence:.0%} confidence"
        with st.expander(f"{finding.agent_name}  ·  {finding.headline}", expanded=finding.agent_name == "Critic Agent"):
            st.markdown(
                f'<div class="finding-meta">{confidence} · citations: '
                f'{", ".join(finding.citations) if finding.citations else "none"}</div>',
                unsafe_allow_html=True,
            )
            for detail in finding.details:
                st.markdown(f"- {detail}")
            if finding.reasoning:
                st.markdown(f"**Reasoning**  \n{finding.reasoning}")
            if finding.critique:
                st.markdown(f"**Self-critique**  \n{finding.critique}")


def _render_evidence(st: Any, report: Any) -> None:
    st.markdown('<div class="section-label">Evidence registry</div>', unsafe_allow_html=True)
    st.caption("Each citation ID points to the source or fixture used by the report.")
    rows = [
        {"Evidence ID": evidence_id, "Source": source}
        for evidence_id, source in report.evidence_registry.items()
    ]
    _render_table(st, rows, ["Evidence ID", "Source"])

    st.markdown('<div class="section-label">Source provenance</div>', unsafe_allow_html=True)
    source_rows = []
    for finding in report.findings:
        for result in finding.tool_results:
            source_rows.append(
                {
                    "Agent": finding.agent_name,
                    "Tool": result.tool_name,
                    "Provider": result.provider,
                    "Fallback": "Yes" if result.fallback_used else "No",
                    "Evidence": len(result.evidence),
                    "Round trip": f"{result.latency_ms:.1f} ms",
                }
            )
    _render_table(
        st,
        source_rows,
        ["Agent", "Tool", "Provider", "Fallback", "Evidence", "Round trip"],
    )


def _render_quality(st: Any, report: Any) -> None:
    evaluation = report.evaluation
    grounding = evaluation.get("grounding", {})
    freshness = evaluation.get("freshness", {})
    columns = st.columns(4)
    quality_metrics = [
        ("Structural score", evaluation.get("score", 0)),
        ("Claim support", grounding.get("claim_support_rate", 0)),
        ("Evidence tools", len(freshness.get("newest_by_tool", {}))),
        ("Unresolved citations", len(grounding.get("unresolved_citations", []))),
    ]
    for column, (label, value) in zip(columns, quality_metrics):
        if isinstance(value, float):
            column.metric(label, f"{value:.0%}")
        else:
            column.metric(label, value)

    st.markdown('<div class="section-label">Quality checks</div>', unsafe_allow_html=True)
    for check in evaluation.get("checks", []):
        st.markdown(f"- {check}")

    contradictions = evaluation.get("contradictions", [])
    if contradictions:
        st.warning("Cross-agent tensions detected")
        for item in contradictions:
            st.markdown(f"- {item}")
    else:
        st.success("No labelled cross-agent tension detected in this run")


def _render_trace(st: Any, summary: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Execution trace</div>', unsafe_allow_html=True)
    stage_rows = summary.get("stage_latencies", [])
    if stage_rows:
        _render_table(st, stage_rows, ["stage", "latency_ms"])

    trace_columns = st.columns(4)
    trace_metrics = [
        ("Provider", f"{summary.get('provider_latency_ms', 0):.1f} ms"),
        ("MCP transport", f"{summary.get('transport_latency_ms', 0):.1f} ms"),
        ("Ollama", f"{summary.get('ollama_latency_ms', 0):.1f} ms"),
        ("MCP failures", str(summary.get("mcp_protocol_failures", 0))),
    ]
    for column, (label, value) in zip(trace_columns, trace_metrics):
        column.metric(label, value)

    if summary.get("decisions"):
        st.markdown('<div class="section-label">Supervisor decisions</div>', unsafe_allow_html=True)
        for decision in summary["decisions"]:
            st.markdown(f"- {decision}")

    with st.expander("Raw trace JSON"):
        st.json(summary)


def _render_run(st: Any, report: Any, trace: Any) -> None:
    summary = trace.to_summary()
    _render_pipeline(st, summary)
    _render_overview(st, report, summary)

    tabs = st.tabs(["Evidence & sources", "Quality checks", "Execution trace"])
    with tabs[0]:
        _render_evidence(st, report)
    with tabs[1]:
        _render_quality(st, report)
    with tabs[2]:
        _render_trace(st, summary)

    report_json = json.dumps(
        {"report": to_jsonable(report), "trace": summary},
        indent=2,
    )
    st.download_button(
        "Download run JSON",
        report_json,
        file_name=f"financial-research-{trace.run_id}.json",
        mime="application/json",
    )
    st.download_button(
        "Download report Markdown",
        report.to_markdown(),
        file_name=f"financial-research-{trace.run_id}.md",
        mime="text/markdown",
    )


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install the ui extra to run the UI."
        ) from exc

    st.set_page_config(
        page_title="Financial Research Desk",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)
    settings = load_settings()
    live_mode, fallback, ollama = _render_sidebar(st, settings)
    _render_header(st)

    if "query_input" not in st.session_state:
        st.session_state.query_input = DEFAULT_QUERY
    submitted = _render_query_composer(st)

    if submitted:
        query = st.session_state.query_input.strip()
        if len(query) < 3:
            st.error("Enter a research question with at least three characters.")
            return
        _apply_run_settings(live_mode, fallback, ollama)
        with st.spinner("Supervisor is coordinating the research run..."):
            try:
                report, trace = SupervisorAgent().run_with_trace(query)
            except (RuntimeError, ValueError, OSError) as exc:
                st.error(f"Research run failed: {exc}")
                st.caption("Disable live mode or enable offline fallback to keep the demo available.")
                return
        st.session_state.last_report = report
        st.session_state.last_trace = trace

    report = st.session_state.get("last_report")
    trace = st.session_state.get("last_trace")
    if report is None or trace is None:
        st.info("Enter a question above, then run the full investigation to see every pipeline stage.")
        return
    _render_run(st, report, trace)


def launch() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install the ui extra to run the UI."
        ) from exc

    ui_file = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(ui_file), "--server.headless=true"]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
