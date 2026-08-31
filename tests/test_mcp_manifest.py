import unittest

from financial_research_agent.mcp import tool_manifest


class McpManifestTests(unittest.TestCase):
    def test_manifest_contains_required_tools(self):
        names = {tool["name"] for tool in tool_manifest()}
        self.assertLessEqual(
            {
                "get_market_data",
                "search_news",
                "search_arxiv",
                "search_documents",
                "search_documents_vector",
                "search_documents_semantic",
                "calculate_returns",
                "calculate_volatility",
                "calculate_max_drawdown",
            },
            names,
        )


if __name__ == "__main__":
    unittest.main()
