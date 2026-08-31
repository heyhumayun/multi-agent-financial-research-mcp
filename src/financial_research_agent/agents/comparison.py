from __future__ import annotations

from financial_research_agent.agents.base import Agent
from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import ToolGateway, expect_market_bars


class ComparisonAgent(Agent):
    name = "Comparison Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        tickers = [part.strip().upper() for part in ticker.split(",") if part.strip()]
        metrics: list[tuple[str, float, float]] = []
        results: list[ToolResult] = []
        for symbol in tickers:
            market = tools.get_market_data(symbol, days=60)
            bars = expect_market_bars(market)
            closes = [bar.close for bar in bars]
            volatility = float(tools.calculate_volatility(closes, window=20).output)
            cumulative_return = (closes[-1] / closes[0]) - 1.0 if closes else 0.0
            metrics.append((symbol, cumulative_return, volatility))
            results.append(
                ToolResult(
                    tool_name="get_market_data",
                    inputs={"ticker": symbol, "days": 60},
                    output=bars,
                    evidence=[f"{market.provider}://market-data/{symbol}"],
                    provider=market.provider,
                    fallback_used=market.fallback_used,
                    latency_ms=market.latency_ms,
                    provider_latency_ms=market.provider_latency_ms,
                    transport_latency_ms=market.transport_latency_ms,
                )
            )
        if len(metrics) < 2:
            details = ["A comparison requires at least two identifiable tickers."]
            headline = "Relative comparison could not be established"
            confidence = 0.25
        else:
            metrics.sort(key=lambda item: item[2])
            low_vol, high_vol = metrics[0], metrics[-1]
            details = [
                f"Lower observed 20-session volatility: {low_vol[0]} at {low_vol[2]:.2%}.",
                f"Higher observed 20-session volatility: {high_vol[0]} at {high_vol[2]:.2%}.",
                *[
                    f"{symbol}: return {ret:.2%}, annualized volatility {vol:.2%}."
                    for symbol, ret, vol in metrics
                ],
                "Comparison uses the same sampled window; it is not a portfolio optimizer or recommendation.",
            ]
            headline = (
                f"Relative risk differs across {', '.join(symbol for symbol, _, _ in metrics)}"
            )
            confidence = 0.7
        return AgentFinding(
            agent_name=self.name,
            headline=headline,
            details=details,
            confidence=confidence,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "aligned price windows were compared using return and volatility metrics.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "pairwise historical statistics do not capture valuation, liquidity, or correlation in a portfolio.",
            ),
            tool_results=results,
        )
