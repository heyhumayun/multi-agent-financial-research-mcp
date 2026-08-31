from __future__ import annotations

import time

from financial_research_agent.mcp import tool_manifest
from financial_research_agent.serialization import to_jsonable
from financial_research_agent.tools import (
    calculate_max_drawdown,
    calculate_returns,
    calculate_volatility,
    get_company_fundamentals_with_metadata,
    get_market_data_with_metadata,
    search_arxiv_with_metadata,
    search_documents,
    search_documents_semantic,
    search_documents_vector,
    search_news_with_metadata,
)

try:
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("financial-research-tools")
    MCP_SDK_STYLE = "v2"
except ImportError:  # pragma: no cover - fallback for older SDKs
    try:
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("financial-research-tools")
        MCP_SDK_STYLE = "v1"
    except ImportError:  # pragma: no cover - exercised only when MCP is not installed
        mcp = None
        MCP_SDK_STYLE = "missing"


if mcp is not None:

    @mcp.tool(name="get_market_data")
    def get_market_data_tool(ticker: str, days: int = 60, provider: str = "auto") -> dict:
        """Get recent market bars together with provider provenance."""
        start = time.perf_counter()
        bars, used_provider, fallback_used = get_market_data_with_metadata(
            ticker=ticker, days=days, provider=provider
        )
        return {
            "data": to_jsonable(bars),
            "provider": used_provider,
            "fallback_used": fallback_used,
            "provider_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    @mcp.tool(name="get_company_fundamentals")
    def get_company_fundamentals_tool(ticker: str, provider: str = "auto") -> dict:
        """Get reported company facts with provider provenance."""
        start = time.perf_counter()
        snapshot, used_provider, fallback_used = get_company_fundamentals_with_metadata(
            ticker=ticker, provider=provider
        )
        return {
            "data": to_jsonable(snapshot),
            "provider": used_provider,
            "fallback_used": fallback_used,
            "provider_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    @mcp.tool(name="search_news")
    def search_news_tool(
        query: str, ticker: str | None = None, limit: int = 5, provider: str = "auto"
    ) -> dict:
        """Search financial news together with provider provenance."""
        start = time.perf_counter()
        items, used_provider, fallback_used = search_news_with_metadata(
            query=query, ticker=ticker, limit=limit, provider=provider
        )
        return {
            "data": to_jsonable(items),
            "provider": used_provider,
            "fallback_used": fallback_used,
            "provider_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    @mcp.tool(name="search_arxiv")
    def search_arxiv_tool(query: str, limit: int = 5, provider: str = "auto") -> dict:
        """Search research papers together with provider provenance."""
        start = time.perf_counter()
        papers, used_provider, fallback_used = search_arxiv_with_metadata(
            query=query, limit=limit, provider=provider
        )
        return {
            "data": to_jsonable(papers),
            "provider": used_provider,
            "fallback_used": fallback_used,
            "provider_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    @mcp.tool(name="search_documents")
    def search_documents_tool(query: str, limit: int = 5) -> list[dict]:
        """Search local research documents."""
        return to_jsonable(search_documents(query=query, limit=limit))

    @mcp.tool(name="search_documents_vector")
    def search_documents_vector_tool(query: str, limit: int = 5) -> list[dict]:
        """Search local research documents with vector-space ranking."""
        return to_jsonable(search_documents_vector(query=query, limit=limit))

    @mcp.tool(name="search_documents_semantic")
    def search_documents_semantic_tool(query: str, limit: int = 5) -> list[dict]:
        """Search local research documents with FAISS embeddings when available."""
        return to_jsonable(search_documents_semantic(query=query, limit=limit))

    @mcp.tool(name="calculate_returns")
    def calculate_returns_tool(prices: list[float]) -> list[float]:
        """Calculate log returns from a price series."""
        return calculate_returns(prices)

    @mcp.tool(name="calculate_volatility")
    def calculate_volatility_tool(
        prices: list[float], window: int = 20, annualize: bool = True
    ) -> float:
        """Calculate rolling volatility from prices."""
        return calculate_volatility(prices=prices, window=window, annualize=annualize)

    @mcp.tool(name="calculate_max_drawdown")
    def calculate_max_drawdown_tool(prices: list[float]) -> float:
        """Calculate maximum drawdown from prices."""
        return calculate_max_drawdown(prices)


def main() -> None:
    if mcp is None:
        names = ", ".join(item["name"] for item in tool_manifest())
        raise RuntimeError(f"MCP package is not installed. Available tool contracts: {names}")
    if MCP_SDK_STYLE == "v2":
        mcp.run("stdio")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
