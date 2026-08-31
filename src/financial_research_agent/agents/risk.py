from __future__ import annotations

from financial_research_agent.agents.base import Agent
from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import ToolGateway, expect_market_bars


class RiskAnalysisAgent(Agent):
    name = "Risk Analysis Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        market_result = tools.get_market_data(ticker, days=90)
        bars = expect_market_bars(market_result)
        closes = [bar.close for bar in bars]
        returns = tools.calculate_returns(closes).output
        volatility_result = tools.calculate_volatility(closes, window=30)
        drawdown_result = tools.calculate_max_drawdown(closes)
        volatility = float(volatility_result.output)
        max_drawdown = float(drawdown_result.output)
        negative_days = sum(1 for ret in returns[-30:] if ret < 0)

        details = [
            f"Annualized 30-session volatility is {volatility:.2%}.",
            f"Maximum drawdown over the sampled period is {max_drawdown:.2%}.",
            f"{negative_days} of the last 30 sessions had negative log returns.",
            "Primary qualitative risks: policy shocks, crowded positioning, earnings catalyst timing, and valuation compression.",
        ]

        confidence = 0.8 if len(closes) >= 60 else 0.55
        return AgentFinding(
            agent_name=self.name,
            headline="Risk is manageable only with explicit catalyst and drawdown controls",
            details=details,
            confidence=confidence,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "volatility, drawdown, negative-return frequency, and qualitative risk categories were combined.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "risk metrics use historical prices and may understate future event shocks.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="calculate_max_drawdown",
                    inputs={"ticker": ticker, "days": 90},
                    output=max_drawdown,
                    evidence=[f"{market_result.provider}://market-data/{ticker.upper()}"],
                    provider=market_result.provider,
                    fallback_used=market_result.fallback_used,
                    latency_ms=market_result.latency_ms + drawdown_result.latency_ms,
                    provider_latency_ms=market_result.provider_latency_ms,
                    transport_latency_ms=(
                        market_result.transport_latency_ms + drawdown_result.transport_latency_ms
                    ),
                ),
                ToolResult(
                    tool_name="calculate_volatility",
                    inputs={"ticker": ticker, "window": 30},
                    output=volatility,
                    provider=volatility_result.provider,
                    latency_ms=volatility_result.latency_ms,
                    provider_latency_ms=volatility_result.provider_latency_ms,
                    transport_latency_ms=volatility_result.transport_latency_ms,
                ),
            ],
        )
