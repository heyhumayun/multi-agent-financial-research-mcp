from __future__ import annotations

from financial_research_agent.agents.base import Agent
from financial_research_agent.domain import AgentFinding, FundamentalSnapshot, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import ToolGateway


class FundamentalsAgent(Agent):
    name = "Fundamentals Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        result = tools.get_company_fundamentals(ticker)
        snapshot = result.output
        if not isinstance(snapshot, FundamentalSnapshot):
            raise TypeError("Fundamentals tool returned an invalid snapshot")
        margin = (
            snapshot.net_income / snapshot.revenue
            if snapshot.revenue and snapshot.net_income is not None
            else None
        )
        details = [f"Latest reported period: {snapshot.period}."]
        if snapshot.revenue is not None:
            details.append(f"Revenue: ${snapshot.revenue / 1e9:.1f}B.")
        if snapshot.net_income is not None:
            details.append(f"Net income: ${snapshot.net_income / 1e9:.1f}B.")
        if margin is not None:
            details.append(f"Reported net margin: {margin:.1%}.")
        if snapshot.assets is not None and snapshot.liabilities is not None:
            details.append(
                f"Assets minus liabilities: ${(snapshot.assets - snapshot.liabilities) / 1e9:.1f}B."
            )
        details.append(
            "Fundamental values are reported facts, not a valuation or earnings forecast."
        )
        return AgentFinding(
            agent_name=self.name,
            headline=f"{ticker.upper()} fundamentals provide primary company context",
            details=details,
            confidence=0.62 if result.provider == "sec" else 0.42,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "reported revenue, earnings, assets, and liabilities were summarized with period metadata.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "company facts do not by themselves establish fair value, future growth, or accounting quality.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="get_company_fundamentals",
                    inputs={"ticker": ticker, "query": query},
                    output=snapshot,
                    evidence=[snapshot.source],
                    provider=result.provider,
                    fallback_used=result.fallback_used,
                    latency_ms=result.latency_ms,
                    provider_latency_ms=result.provider_latency_ms,
                    transport_latency_ms=result.transport_latency_ms,
                )
            ],
        )
