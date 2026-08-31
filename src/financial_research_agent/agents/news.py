from __future__ import annotations

from financial_research_agent.agents.base import Agent
from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import ToolGateway, expect_news_items


class NewsAgent(Agent):
    name = "News Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        news_result = tools.search_news(query=query, ticker=ticker, limit=5)
        items = expect_news_items(news_result)
        if not items:
            return AgentFinding(
                agent_name=self.name,
                headline="No relevant news found",
                details=[
                    f"The {news_result.provider} news provider returned no matching articles."
                ],
                confidence=0.35,
                reasoning=self.reasoning_engine.reason(
                    self.name,
                    "the selected provider returned an empty result, so no sentiment conclusion was drawn.",
                ),
                critique=self.reasoning_engine.critique(
                    self.name,
                    "an empty result can reflect limited provider coverage rather than absence of events.",
                ),
                tool_results=[
                    ToolResult(
                        tool_name="search_news",
                        inputs={"query": query, "ticker": ticker, "limit": 5},
                        output=items,
                        evidence=[],
                        provider=news_result.provider,
                        fallback_used=news_result.fallback_used,
                        latency_ms=news_result.latency_ms,
                        provider_latency_ms=news_result.provider_latency_ms,
                        transport_latency_ms=news_result.transport_latency_ms,
                    )
                ],
            )

        avg_sentiment = sum(item.sentiment for item in items) / len(items)
        sentiment_label = "supportive" if avg_sentiment > 0.15 else "cautious"
        details = [
            f"Average sampled news sentiment is {avg_sentiment:.2f}, which is {sentiment_label}.",
            *[
                f"{item.published.isoformat()} | {item.source}: {item.title} ({item.sentiment:+.2f})"
                for item in items
            ],
        ]

        return AgentFinding(
            agent_name=self.name,
            headline=f"News flow is {sentiment_label} for {ticker.upper()}",
            details=details,
            confidence=0.72,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "article sentiment and event themes were aggregated into a cautious/supportive label.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "headline sentiment is noisy and should be cross-checked with source quality and recency.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="search_news",
                    inputs={"query": query, "ticker": ticker, "limit": 5},
                    output=items,
                    evidence=[item.url for item in items if item.url],
                    provider=news_result.provider,
                    fallback_used=news_result.fallback_used,
                    latency_ms=news_result.latency_ms,
                    provider_latency_ms=news_result.provider_latency_ms,
                    transport_latency_ms=news_result.transport_latency_ms,
                )
            ],
        )
