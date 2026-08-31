from __future__ import annotations

import json
import os
import queue
import sys
import time
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Protocol

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from typing_extensions import Self

from financial_research_agent.config import load_settings
from financial_research_agent.domain import (
    DocumentHit,
    FundamentalSnapshot,
    MarketBar,
    NewsItem,
    PaperItem,
)
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


@dataclass(frozen=True)
class GatewayResult:
    output: object
    provider: str
    fallback_used: bool = False
    latency_ms: float = 0.0
    provider_latency_ms: float = 0.0
    transport_latency_ms: float = 0.0


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class ToolGateway(Protocol):
    runtime: str

    def get_market_data(self, ticker: str, days: int) -> GatewayResult: ...

    def get_company_fundamentals(self, ticker: str) -> GatewayResult: ...

    def search_news(self, query: str, ticker: str | None, limit: int) -> GatewayResult: ...

    def search_arxiv(self, query: str, limit: int) -> GatewayResult: ...

    def search_documents(self, query: str, limit: int) -> GatewayResult: ...

    def search_documents_vector(self, query: str, limit: int) -> GatewayResult: ...

    def search_documents_semantic(self, query: str, limit: int) -> GatewayResult: ...

    def calculate_returns(self, prices: list[float]) -> GatewayResult: ...

    def calculate_volatility(self, prices: list[float], window: int) -> GatewayResult: ...

    def calculate_max_drawdown(self, prices: list[float]) -> GatewayResult: ...


class LocalToolGateway:
    runtime = "local"

    def get_market_data(self, ticker: str, days: int) -> GatewayResult:
        start = time.perf_counter()
        bars, provider, fallback_used = get_market_data_with_metadata(ticker=ticker, days=days)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=bars,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=latency,
            provider_latency_ms=latency,
        )

    def get_company_fundamentals(self, ticker: str) -> GatewayResult:
        start = time.perf_counter()
        snapshot, provider, fallback_used = get_company_fundamentals_with_metadata(ticker=ticker)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=snapshot,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=latency,
            provider_latency_ms=latency,
        )

    def search_news(self, query: str, ticker: str | None, limit: int) -> GatewayResult:
        start = time.perf_counter()
        items, provider, fallback_used = search_news_with_metadata(
            query=query, ticker=ticker, limit=limit
        )
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=items,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=latency,
            provider_latency_ms=latency,
        )

    def search_arxiv(self, query: str, limit: int) -> GatewayResult:
        start = time.perf_counter()
        papers, provider, fallback_used = search_arxiv_with_metadata(query=query, limit=limit)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=papers,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=latency,
            provider_latency_ms=latency,
        )

    def search_documents(self, query: str, limit: int) -> GatewayResult:
        start = time.perf_counter()
        output = search_documents(query=query, limit=limit)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=output, provider="local", latency_ms=latency, provider_latency_ms=latency
        )

    def search_documents_vector(self, query: str, limit: int) -> GatewayResult:
        start = time.perf_counter()
        output = search_documents_vector(query=query, limit=limit)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=output,
            provider="local-vector",
            latency_ms=latency,
            provider_latency_ms=latency,
        )

    def search_documents_semantic(self, query: str, limit: int) -> GatewayResult:
        start = time.perf_counter()
        output = search_documents_semantic(query=query, limit=limit)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=output,
            provider="semantic-faiss-or-vector-fallback",
            latency_ms=latency,
            provider_latency_ms=latency,
        )

    def calculate_returns(self, prices: list[float]) -> GatewayResult:
        start = time.perf_counter()
        output = calculate_returns(prices)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=output, provider="local", latency_ms=latency, provider_latency_ms=latency
        )

    def calculate_volatility(self, prices: list[float], window: int) -> GatewayResult:
        start = time.perf_counter()
        output = calculate_volatility(prices, window=window)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=output, provider="local", latency_ms=latency, provider_latency_ms=latency
        )

    def calculate_max_drawdown(self, prices: list[float]) -> GatewayResult:
        start = time.perf_counter()
        output = calculate_max_drawdown(prices)
        latency = _elapsed_ms(start)
        return GatewayResult(
            output=output, provider="local", latency_ms=latency, provider_latency_ms=latency
        )


class McpToolGateway(LocalToolGateway):
    """MCP runtime boundary.

    This class gives agents an MCP-client-shaped tool path while keeping the demo
    synchronous and reliable. A production version would replace each inherited
    method with calls to an MCP stdio/session client.
    """

    runtime = "mcp"


class McpStdioToolGateway:
    """Real MCP stdio client gateway.

    The gateway opens one MCP client/server session lazily and reuses it for
    every tool call made during a research run. The supervisor closes the
    session at the end of the run.
    """

    runtime = "mcp-stdio"

    def __init__(self) -> None:
        self._portal_cm = None
        self._portal: BlockingPortal | None = None
        self._requests: queue.Queue[tuple[str | None, dict, queue.Queue]] = queue.Queue()
        self._worker_future: Future | None = None

    def close(self) -> None:
        if self._worker_future is not None:
            response: queue.Queue = queue.Queue(maxsize=1)
            self._requests.put((None, {}, response))
            self._worker_future.result(timeout=10)
        if self._portal_cm is not None:
            self._portal_cm.__exit__(None, None, None)
        self._portal = None
        self._portal_cm = None
        self._worker_future = None

    def __enter__(self) -> Self:
        self._ensure_session()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_session(self) -> None:
        if self._worker_future is not None:
            return
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._worker_future = self._portal.start_task_soon(self._session_worker)

    def _call_tool(self, name: str, arguments: dict) -> tuple[object, float]:
        start = time.perf_counter()
        self._ensure_session()
        response: queue.Queue = queue.Queue(maxsize=1)
        self._requests.put((name, arguments, response))
        deadline = time.monotonic() + 30
        while True:
            try:
                ok, payload = response.get(timeout=0.2)
                break
            except queue.Empty:
                if self._worker_future is not None and self._worker_future.done():
                    self._worker_future.result()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"MCP tool call timed out: {name}")
        if not ok:
            raise payload
        return payload, _elapsed_ms(start)

    async def _session_worker(self) -> None:
        env = dict(os.environ)
        env["FIN_RESEARCH_TOOL_RUNTIME"] = "local"
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "financial_research_agent.mcp.server"],
            env=env,
        )
        async with (
            stdio_client(server) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            while True:
                name, arguments, response = await anyio.to_thread.run_sync(self._requests.get)
                if name is None:
                    return
                try:
                    result = await session.call_tool(name, arguments)
                    if result.is_error:
                        raise RuntimeError(f"MCP tool call failed: {name}")
                    response.put((True, _decode_mcp_result(result)))
                except Exception as exc:  # noqa: BLE001 - relay remote failures to sync caller
                    response.put((False, exc))

    def get_market_data(self, ticker: str, days: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool(
            "get_market_data", {"ticker": ticker, "days": days}
        )
        data, provider, fallback_used, provider_ms = _unwrap_provider_payload(
            payload, default="mcp"
        )
        bars = [
            MarketBar(
                ticker=item["ticker"],
                date=date.fromisoformat(item["date"]),
                close=float(item["close"]),
                volume=int(item["volume"]),
            )
            for item in data  # type: ignore[union-attr]
        ]
        return GatewayResult(
            output=bars,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=round_trip_ms,
            provider_latency_ms=provider_ms,
            transport_latency_ms=max(round_trip_ms - provider_ms, 0.0),
        )

    def get_company_fundamentals(self, ticker: str) -> GatewayResult:
        payload, round_trip_ms = self._call_tool("get_company_fundamentals", {"ticker": ticker})
        data, provider, fallback_used, provider_ms = _unwrap_provider_payload(
            payload, default="mcp"
        )
        return GatewayResult(
            output=FundamentalSnapshot(**data),
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=round_trip_ms,
            provider_latency_ms=provider_ms,
            transport_latency_ms=max(round_trip_ms - provider_ms, 0.0),
        )

    def search_news(self, query: str, ticker: str | None, limit: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool(
            "search_news", {"query": query, "ticker": ticker, "limit": limit}
        )
        data, provider, fallback_used, provider_ms = _unwrap_provider_payload(
            payload, default="mcp"
        )
        items = [
            NewsItem(
                ticker=item["ticker"],
                published=date.fromisoformat(item["published"]),
                title=item["title"],
                source=item["source"],
                sentiment=float(item["sentiment"]),
                url=item.get("url", ""),
            )
            for item in data  # type: ignore[union-attr]
        ]
        return GatewayResult(
            output=items,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=round_trip_ms,
            provider_latency_ms=provider_ms,
            transport_latency_ms=max(round_trip_ms - provider_ms, 0.0),
        )

    def search_arxiv(self, query: str, limit: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool("search_arxiv", {"query": query, "limit": limit})
        data, provider, fallback_used, provider_ms = _unwrap_provider_payload(
            payload, default="mcp"
        )
        papers = [
            PaperItem(
                title=item["title"],
                authors=tuple(item.get("authors", [])),
                summary=item["summary"],
                url=item["url"],
                relevance_score=float(item["relevance_score"]),
            )
            for item in data  # type: ignore[union-attr]
        ]
        return GatewayResult(
            output=papers,
            provider=provider,
            fallback_used=fallback_used,
            latency_ms=round_trip_ms,
            provider_latency_ms=provider_ms,
            transport_latency_ms=max(round_trip_ms - provider_ms, 0.0),
        )

    def search_documents(self, query: str, limit: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool(
            "search_documents", {"query": query, "limit": limit}
        )
        hits = [DocumentHit(**item) for item in payload]  # type: ignore[arg-type]
        return GatewayResult(
            output=hits,
            provider="mcp",
            latency_ms=round_trip_ms,
            transport_latency_ms=round_trip_ms,
        )

    def search_documents_vector(self, query: str, limit: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool(
            "search_documents_vector", {"query": query, "limit": limit}
        )
        hits = [DocumentHit(**item) for item in payload]  # type: ignore[arg-type]
        return GatewayResult(
            output=hits,
            provider="mcp-vector",
            latency_ms=round_trip_ms,
            transport_latency_ms=round_trip_ms,
        )

    def search_documents_semantic(self, query: str, limit: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool(
            "search_documents_semantic", {"query": query, "limit": limit}
        )
        hits = [DocumentHit(**item) for item in payload]  # type: ignore[arg-type]
        return GatewayResult(
            output=hits,
            provider="mcp-semantic",
            latency_ms=round_trip_ms,
            transport_latency_ms=round_trip_ms,
        )

    def calculate_returns(self, prices: list[float]) -> GatewayResult:
        payload, round_trip_ms = self._call_tool("calculate_returns", {"prices": prices})
        return GatewayResult(
            output=payload,
            provider="mcp",
            latency_ms=round_trip_ms,
            transport_latency_ms=round_trip_ms,
        )

    def calculate_volatility(self, prices: list[float], window: int) -> GatewayResult:
        payload, round_trip_ms = self._call_tool(
            "calculate_volatility", {"prices": prices, "window": window}
        )
        return GatewayResult(
            output=payload,
            provider="mcp",
            latency_ms=round_trip_ms,
            transport_latency_ms=round_trip_ms,
        )

    def calculate_max_drawdown(self, prices: list[float]) -> GatewayResult:
        payload, round_trip_ms = self._call_tool("calculate_max_drawdown", {"prices": prices})
        return GatewayResult(
            output=payload,
            provider="mcp",
            latency_ms=round_trip_ms,
            transport_latency_ms=round_trip_ms,
        )


def _decode_mcp_result(result: object) -> object:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured.get("result", structured) if isinstance(structured, dict) else structured

    content = getattr(result, "content", [])
    if not content:
        return None

    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return first
    return json.loads(text)


def _infer_provider_from_payload(payload: object, default: str) -> str:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            url = str(first.get("url", ""))
            source = str(first.get("source", "")).lower()
            if url.startswith("offline://"):
                return "offline"
            if "yahoo" in source:
                return "yahoo_rss"
            if "arxiv.org" in url:
                return "arxiv"
    return default


def _unwrap_provider_payload(payload: object, default: str) -> tuple[object, str, bool, float]:
    """Read the provenance envelope returned by remote data tools.

    The list fallback keeps the client compatible with older MCP server versions.
    """
    if isinstance(payload, dict) and "data" in payload:
        return (
            payload["data"],
            str(payload.get("provider", default)),
            bool(payload.get("fallback_used", False)),
            float(payload.get("provider_latency_ms", 0.0)),
        )
    return payload, _infer_provider_from_payload(payload, default=default), False, 0.0


def build_tool_gateway() -> ToolGateway:
    settings = load_settings()
    if settings.tool_runtime == "mcp-stdio":
        return McpStdioToolGateway()
    if settings.tool_runtime == "mcp":
        return McpToolGateway()
    return LocalToolGateway()


def expect_market_bars(result: GatewayResult) -> list[MarketBar]:
    return list(result.output)  # type: ignore[arg-type]


def expect_news_items(result: GatewayResult) -> list[NewsItem]:
    return list(result.output)  # type: ignore[arg-type]


def expect_paper_items(result: GatewayResult) -> list[PaperItem]:
    return list(result.output)  # type: ignore[arg-type]


def expect_document_hits(result: GatewayResult) -> list[DocumentHit]:
    return list(result.output)  # type: ignore[arg-type]
