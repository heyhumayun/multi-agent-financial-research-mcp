import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from financial_research_agent.agents import SupervisorAgent
from financial_research_agent.agents.critic import CriticAgent
from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import _decode_mcp_result
from financial_research_agent.tools.fundamentals import (
    get_company_fundamentals_with_metadata,
)
from financial_research_agent.tools.market_data import get_market_data_with_metadata
from financial_research_agent.tools.news import search_news_with_metadata


class FailureInjectionTests(unittest.TestCase):
    def test_polygon_timeout_falls_back_and_is_visible_to_critic(self):
        environment = {
            "FIN_RESEARCH_LIVE": "1",
            "FIN_RESEARCH_OFFLINE_FALLBACK": "1",
            "FIN_RESEARCH_TOOL_RUNTIME": "local",
            "FIN_RESEARCH_MARKET_PROVIDER": "polygon",
            "POLYGON_API_KEY": "test-key",
            "FIN_RESEARCH_NEWS_PROVIDER": "offline",
            "FIN_RESEARCH_PAPERS_PROVIDER": "offline",
            "FIN_RESEARCH_LLM": "off",
        }
        with (
            patch.dict("os.environ", environment),
            patch(
                "financial_research_agent.tools.market_data.urlopen",
                side_effect=TimeoutError("injected Polygon timeout"),
            ),
        ):
            report = SupervisorAgent().run("Assess NVDA downside risk")
        self.assertIn("get_market_data", report.evaluation["freshness"]["fallback_tools"])
        critic = next(
            finding for finding in report.findings if finding.agent_name == "Critic Agent"
        )
        self.assertIn("fell back", " ".join(critic.details).lower())
        self.assertLess(report.evaluation["score"], 1.0)

    def test_polygon_timeout_raises_in_strict_mode(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "FIN_RESEARCH_OFFLINE_FALLBACK": "0",
                    "POLYGON_API_KEY": "test-key",
                },
            ),
            patch(
                "financial_research_agent.tools.market_data.urlopen",
                side_effect=TimeoutError("injected Polygon timeout"),
            ),
            self.assertRaisesRegex(TimeoutError, "injected Polygon timeout"),
        ):
            get_market_data_with_metadata("NVDA", provider="polygon")

    def test_finnhub_503_retries_then_falls_back(self):
        error = HTTPError("https://finnhub.io", 503, "unavailable", {}, None)
        with (
            patch.dict(
                "os.environ",
                {
                    "FIN_RESEARCH_OFFLINE_FALLBACK": "1",
                    "FINNHUB_API_KEY": "test-key",
                },
            ),
            patch(
                "financial_research_agent.tools.news.urlopen", side_effect=error
            ) as mocked_urlopen,
            patch("financial_research_agent.tools.news.time.sleep"),
        ):
            items, provider, fallback = search_news_with_metadata(
                "NVDA risk", "NVDA", provider="finnhub"
            )
        self.assertEqual(mocked_urlopen.call_count, 3)
        self.assertEqual(provider, "offline")
        self.assertTrue(fallback)
        self.assertTrue(items)

    def test_malformed_mcp_payload_fails_closed(self):
        result = MagicMock()
        result.structured_content = None
        content = MagicMock()
        content.text = "not-json"
        result.content = [content]
        with self.assertRaises(json.JSONDecodeError):
            _decode_mcp_result(result)

    def test_ollama_unavailable_uses_deterministic_reasoning(self):
        with (
            patch.dict("os.environ", {"FIN_RESEARCH_LLM": "ollama"}),
            patch(
                "financial_research_agent.reasoning.urlopen",
                side_effect=OSError("injected Ollama outage"),
            ),
        ):
            reasoning = ReasoningEngine().reason("Risk Agent", "volatility was high")
        self.assertIn("deterministic rules", reasoning)

    def test_empty_sec_response_falls_back_without_fabricating_live_provenance(self):
        response = MagicMock()
        response.read.return_value = json.dumps({"facts": {}}).encode()
        response.__enter__.return_value = response
        with (
            patch.dict("os.environ", {"FIN_RESEARCH_OFFLINE_FALLBACK": "1"}),
            patch(
                "financial_research_agent.tools.fundamentals.urlopen",
                return_value=response,
            ),
        ):
            snapshot, provider, fallback = get_company_fundamentals_with_metadata(
                "NVDA", provider="sec"
            )
        self.assertEqual(provider, "offline")
        self.assertTrue(fallback)
        self.assertTrue(snapshot.source.startswith("offline://"))

    def test_critic_detects_injected_cross_agent_contradiction(self):
        findings = [
            AgentFinding(
                agent_name="Market Data Agent",
                headline="NVDA market tape shows positive momentum",
                details=["Momentum is positive."],
                confidence=0.8,
                tool_results=[ToolResult("get_market_data", {}, [], ["market://NVDA"])],
            ),
            AgentFinding(
                agent_name="News Agent",
                headline="News flow is cautious for NVDA",
                details=["News is cautious."],
                confidence=0.7,
                tool_results=[ToolResult("search_news", {}, [], ["news://NVDA"])],
            ),
            AgentFinding(
                agent_name="Risk Analysis Agent",
                headline="Risk includes quantified drawdown",
                details=["Maximum drawdown is -10%."],
                confidence=0.8,
            ),
        ]
        critic = CriticAgent().review("Assess NVDA risk", findings)
        self.assertIn("contradiction", " ".join(critic.details).lower())


if __name__ == "__main__":
    unittest.main()
