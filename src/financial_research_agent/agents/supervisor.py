from __future__ import annotations

import re
import time

from financial_research_agent.agents.comparison import ComparisonAgent
from financial_research_agent.agents.critic import CriticAgent
from financial_research_agent.agents.fundamentals import FundamentalsAgent
from financial_research_agent.agents.market import MarketDataAgent
from financial_research_agent.agents.news import NewsAgent
from financial_research_agent.agents.research import DocumentAgent, ResearchPapersAgent
from financial_research_agent.agents.risk import RiskAnalysisAgent
from financial_research_agent.agents.synthesis import ReportSynthesisAgent
from financial_research_agent.domain import AgentFinding, ResearchReport
from financial_research_agent.evaluation import evaluate_report
from financial_research_agent.grounding import ground_findings
from financial_research_agent.observability import RunTrace, timed_call
from financial_research_agent.reasoning import (
    ReasoningEngine,
    consume_reasoning_telemetry,
    reset_reasoning_telemetry,
)
from financial_research_agent.tool_gateway import ToolGateway, build_tool_gateway


class SupervisorAgent:
    """Routes the user query to specialist agents and asks synthesis to compose a brief."""

    def __init__(self) -> None:
        self.tools = build_tool_gateway()
        self.market_agent = MarketDataAgent()
        self.news_agent = NewsAgent()
        self.research_agent = ResearchPapersAgent()
        self.document_agent = DocumentAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.fundamentals_agent = FundamentalsAgent()
        self.comparison_agent = ComparisonAgent()
        self.synthesis_agent = ReportSynthesisAgent()
        self.critic_agent = CriticAgent()
        self.reasoning_engine = ReasoningEngine()

    def infer_ticker(self, query: str) -> str:
        return self.infer_tickers(query)[0]

    def infer_tickers(self, query: str) -> list[str]:
        known = {"NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "META", "TSLA"}
        tokens = set(re.findall(r"\b[A-Z]{2,5}\b", query.upper()))
        matches = known & tokens
        if matches:
            return sorted(matches)
        excluded = {"AI", "API", "ETF", "FAISS", "GNN", "LLM", "MCP", "ML", "RAG", "RISK"}
        symbols = [token for token in re.findall(r"\b[A-Z]{1,5}\b", query) if token not in excluded]
        return list(dict.fromkeys(symbols)) or ["NVDA"]

    def plan(self, query: str) -> list[str]:
        lowered = query.lower()
        selected = ["market", "news", "risk"]
        if any(
            term in lowered
            for term in [
                "paper",
                "research",
                "model",
                "ml",
                "agent",
                "rag",
                "compare",
                "volatility",
            ]
        ):
            selected.append("research")
        if any(term in lowered for term in ["note", "document", "framework", "infrastructure"]):
            selected.append("document")
        if any(
            term in lowered
            for term in [
                "company",
                "fundamental",
                "earnings",
                "revenue",
                "valuation",
                "balance sheet",
            ]
        ):
            selected.append("fundamentals")
        if len(self.infer_tickers(query)) >= 2 or any(
            term in lowered for term in ["compare", "versus", " vs "]
        ):
            selected.append("comparison")
        planned = self.reasoning_engine.plan_agents(query, selected)
        if len(self.infer_tickers(query)) < 2:
            planned = [agent for agent in planned if agent != "comparison"]
        return planned

    def _run_agent(
        self, agent_key: str, query: str, ticker: str, trace: RunTrace, tools: ToolGateway
    ) -> AgentFinding:
        agents = {
            "market": self.market_agent,
            "news": self.news_agent,
            "research": self.research_agent,
            "document": self.document_agent,
            "risk": self.risk_agent,
            "fundamentals": self.fundamentals_agent,
            "comparison": self.comparison_agent,
        }
        agent = agents[agent_key]
        trace.record_agent(agent.name)
        try:
            with timed_call() as elapsed_ms:
                finding = agent.run(query, ticker, tools)
        except Exception as exc:
            if tools.runtime.startswith("mcp"):
                trace.record_protocol_failure(f"{agent.name}: {type(exc).__name__}: {exc}")
            raise

        agent_latency = elapsed_ms()
        tool_latency = sum(result.latency_ms for result in finding.tool_results)
        trace.record_stage(f"agent:{agent_key}", agent_latency)
        trace.record_stage(f"agent_reasoning:{agent_key}", max(agent_latency - tool_latency, 0.0))
        for result in finding.tool_results:
            trace.record_tool(
                agent_name=finding.agent_name,
                tool_name=result.tool_name,
                provider=result.provider,
                fallback_used=result.fallback_used,
                latency_ms=result.latency_ms,
                evidence_count=len(result.evidence),
                provider_latency_ms=result.provider_latency_ms,
                transport_latency_ms=result.transport_latency_ms,
            )
        return finding

    def run_with_trace(self, query: str) -> tuple[ResearchReport, RunTrace]:
        run_start = time.perf_counter()
        reset_reasoning_telemetry()
        query = query.strip()
        if not query:
            raise ValueError("Research query must not be empty.")
        trace = RunTrace()
        tickers = self.infer_tickers(query)
        ticker = tickers[0]
        with timed_call() as planning_ms:
            planned_agents = self.plan(query)
        trace.record_stage("planning", planning_ms())
        trace.record_plan(planned_agents)
        tools = self.tools
        trace.tool_runtime = tools.runtime
        findings: list[AgentFinding] = []

        try:
            for agent_key in planned_agents:
                trace.record_iteration()
                agent_ticker = ",".join(tickers) if agent_key == "comparison" else ticker
                findings.append(self._run_agent(agent_key, query, agent_ticker, trace, tools))

            follow_up = self.reasoning_engine.follow_up_agents(query, findings, planned_agents)
            if follow_up:
                trace.record_decision(
                    f"Follow-up pass requested for missing evidence: {', '.join(follow_up)}."
                )
                for agent_key in follow_up:
                    trace.record_iteration()
                    agent_ticker = ",".join(tickers) if agent_key == "comparison" else ticker
                    findings.append(self._run_agent(agent_key, query, agent_ticker, trace, tools))
            else:
                trace.record_decision(
                    "Initial evidence coverage was sufficient; no follow-up pass."
                )

            trace.record_agent(self.critic_agent.name)
            findings.append(self.critic_agent.review(query, findings))
            trace.stop_reason = "bounded_review_complete"
        finally:
            close = getattr(tools, "close", None)
            if callable(close):
                close()

        with timed_call() as grounding_ms:
            findings, evidence_registry = ground_findings(findings)
        trace.record_stage("grounding", grounding_ms())
        with timed_call() as synthesis_ms:
            report = self.synthesis_agent.run(query, findings)
        trace.record_stage("synthesis", synthesis_ms())
        report = ResearchReport(
            query=report.query,
            thesis=report.thesis,
            findings=report.findings,
            risks=report.risks,
            next_steps=report.next_steps,
            run_id=trace.run_id,
            evidence_registry=evidence_registry,
        )
        with timed_call() as evaluation_ms:
            evaluation = evaluate_report(report)
        trace.record_stage("evaluation", evaluation_ms())
        report = ResearchReport(
            query=report.query,
            thesis=report.thesis,
            findings=report.findings,
            risks=report.risks,
            next_steps=report.next_steps,
            run_id=report.run_id,
            evaluation=evaluation,
            evidence_registry=report.evidence_registry,
        )
        for event in consume_reasoning_telemetry():
            trace.record_ollama(**event)
        trace.end_to_end_latency_ms = round((time.perf_counter() - run_start) * 1000, 2)
        return report, trace

    def run(self, query: str) -> ResearchReport:
        report, _ = self.run_with_trace(query)
        return report
