from __future__ import annotations

from financial_research_agent.agents.base import Agent
from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import ToolGateway, expect_market_bars


class MarketDataAgent(Agent):
    name = "Market Data Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        market_result = tools.get_market_data(ticker, days=60)
        bars = expect_market_bars(market_result)
        closes = [bar.close for bar in bars]
        returns = tools.calculate_returns(closes).output
        volatility_result = tools.calculate_volatility(closes, window=20)
        volatility = float(volatility_result.output)
        cumulative_return = (closes[-1] / closes[0]) - 1.0 if closes else 0.0
        latest = bars[-1]

        trend_label = "positive" if cumulative_return > 0 else "negative"
        details = [
            f"Latest close for {ticker.upper()} is {latest.close:.2f} on {latest.date.isoformat()}.",
            f"60-session cumulative return is {cumulative_return:.2%}, indicating {trend_label} price momentum.",
            f"Annualized 20-session volatility is {volatility:.2%}.",
            f"Recent average daily log return is {(sum(returns[-20:]) / max(len(returns[-20:]), 1)):.4f}.",
        ]

        return AgentFinding(
            agent_name=self.name,
            headline=f"{ticker.upper()} market tape shows {trend_label} medium-term momentum",
            details=details,
            confidence=0.78,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "price trend, log returns, and annualized volatility were compared.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "market tape alone ignores fundamentals, catalysts, and liquidity conditions.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="get_market_data",
                    inputs={"ticker": ticker, "days": 60},
                    output=bars,
                    evidence=[f"{market_result.provider}://market-data/{ticker.upper()}"],
                    provider=market_result.provider,
                    fallback_used=market_result.fallback_used,
                    latency_ms=market_result.latency_ms,
                    provider_latency_ms=market_result.provider_latency_ms,
                    transport_latency_ms=market_result.transport_latency_ms,
                ),
                ToolResult(
                    tool_name="calculate_volatility",
                    inputs={"prices": "latest 21 closes", "window": 20},
                    output=volatility,
                    provider=volatility_result.provider,
                    latency_ms=volatility_result.latency_ms,
                    provider_latency_ms=volatility_result.provider_latency_ms,
                    transport_latency_ms=volatility_result.transport_latency_ms,
                ),
            ],
        )
