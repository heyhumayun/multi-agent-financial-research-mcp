from __future__ import annotations

import re

from financial_research_agent.domain import AgentFinding, ResearchReport
from financial_research_agent.grounding import ensure_thesis_citations
from financial_research_agent.reasoning import ReasoningEngine


class ReportSynthesisAgent:
    name = "Report Synthesis Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, findings: list[AgentFinding]) -> ResearchReport:
        supportive_count = sum(
            1
            for finding in findings
            if any(
                word in finding.headline.lower() for word in ["positive", "supportive", "supports"]
            )
        )
        risk_count = sum(
            1
            for finding in findings
            if any(word in finding.headline.lower() for word in ["risk", "cautious", "drawdown"])
        )

        if supportive_count > risk_count:
            thesis = (
                "The research stack leans constructively, but the trade should be sized around "
                "volatility and catalyst risk rather than treated as a one-way thematic bet."
            )
        else:
            thesis = (
                "The evidence is not strong enough for a one-way conclusion; data coverage, "
                "volatility, and catalyst risk matter more than headline momentum."
            )

        thesis = self.reasoning_engine.synthesize(
            query=query,
            finding_summaries=[
                f"{finding.agent_name}: {finding.headline} [{', '.join(finding.citations)}]"
                for finding in findings
            ],
            fallback=thesis,
        )
        thesis = ensure_thesis_citations(thesis, findings)

        risks = [
            "Live providers can still fail or return stale data; fallback usage must be monitored.",
            "Agent outputs are research aids, not trading instructions.",
            "Source quality, latency, and stale data must be monitored in production.",
        ]
        lowered = query.lower()
        if "option" in lowered or "implied volatility" in lowered:
            risks.append(
                "Options-chain and implied-volatility-surface analysis is unsupported because no options tool is available."
            )
        if any(term in lowered for term in ["forex", "eurusd", "currency", "carry"]):
            risks.append(
                "Forex and central-bank carry analysis is unsupported by the current equity-data tools."
            )
        if any(term in lowered for term in ["commodity", "crude oil", "futures curve"]):
            risks.append(
                "Commodity futures curves and inventory data are unsupported by the current tool set."
            )
        explicit_symbols = re.findall(r"\b[A-Z]{2,5}\b", query)
        excluded = {"AI", "ML", "MCP", "RAG", "GNN", "LLM", "ETF", "RISK"}
        if not any(symbol not in excluded for symbol in explicit_symbols):
            risks.append(
                "No explicit ticker or symbol was supplied; entity resolution should be confirmed before relying on company-specific evidence."
            )
        next_steps = [
            "Validate the cited source records, publication dates, and provider provenance before relying on the brief.",
            "Review the quality and trace panels for fallback use, stale evidence, contradictions, and MCP failures.",
            "Run the labelled benchmark and deterministic-versus-Ollama ablation before promoting routing or reasoning changes.",
        ]
        return ResearchReport(
            query=query, thesis=thesis, findings=findings, risks=risks, next_steps=next_steps
        )
