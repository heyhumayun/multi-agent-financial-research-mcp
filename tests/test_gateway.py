import os
import unittest
from unittest.mock import patch

from financial_research_agent.tool_gateway import (
    LocalToolGateway,
    McpStdioToolGateway,
    McpToolGateway,
    build_tool_gateway,
)


class GatewayTests(unittest.TestCase):
    def test_local_gateway_default_shape(self):
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            gateway = LocalToolGateway()
            result = gateway.get_market_data("NVDA", days=3)
        self.assertEqual(len(result.output), 3)
        self.assertTrue(result.provider)

    def test_mcp_runtime_selects_gateway_boundary(self):
        old_value = os.environ.get("FIN_RESEARCH_TOOL_RUNTIME")
        os.environ["FIN_RESEARCH_TOOL_RUNTIME"] = "mcp"
        try:
            gateway = build_tool_gateway()
        finally:
            if old_value is None:
                os.environ.pop("FIN_RESEARCH_TOOL_RUNTIME", None)
            else:
                os.environ["FIN_RESEARCH_TOOL_RUNTIME"] = old_value
        self.assertIsInstance(gateway, McpToolGateway)

    def test_mcp_stdio_runtime_selects_real_client_gateway(self):
        old_value = os.environ.get("FIN_RESEARCH_TOOL_RUNTIME")
        os.environ["FIN_RESEARCH_TOOL_RUNTIME"] = "mcp-stdio"
        try:
            gateway = build_tool_gateway()
        finally:
            if old_value is None:
                os.environ.pop("FIN_RESEARCH_TOOL_RUNTIME", None)
            else:
                os.environ["FIN_RESEARCH_TOOL_RUNTIME"] = old_value
        self.assertIsInstance(gateway, McpStdioToolGateway)

    def test_mcp_stdio_preserves_provider_metadata(self):
        with patch.dict(
            "os.environ",
            {
                "FIN_RESEARCH_LIVE": "1",
                "FIN_RESEARCH_MARKET_PROVIDER": "alpha_vantage",
                "ALPHA_VANTAGE_API_KEY": "",
            },
        ):
            gateway = McpStdioToolGateway()
            try:
                result = gateway.get_market_data("NVDA", days=3)
            finally:
                gateway.close()
        self.assertEqual(result.provider, "offline")
        self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
