# Learning Notes

## RAG vs Agentic Research

RAG answers a question by retrieving context and asking a model to synthesize an answer. Agentic research adds planning, tool selection, intermediate analysis, and multi-step workflows. In this project, retrieval is only one capability. Market data, news, papers, documents, and risk calculations are separate tools.

## Why Multi-Agent?

Multi-agent architecture is useful when different parts of a workflow have different responsibilities:

- Market Data Agent: numerical tape, returns, volatility.
- News Agent: qualitative event flow and sentiment.
- Research/Papers Agent: external technical context.
- Document Agent: local knowledge base retrieval.
- Risk Analysis Agent: downside and uncertainty.
- Synthesis Agent: final report composition.

The supervisor decides which specialists to call. This is cleaner than a single giant prompt that tries to do everything.

## Why MCP?

MCP is a standard way to expose tools to agent clients. The important idea is not just the package; it is the contract boundary:

- Tool inputs and outputs are explicit.
- Tools can be tested without an LLM.
- Agents can be swapped without rewriting domain logic.
- Internal tools can be reused by dashboards, scheduled jobs, or batch workflows.

## Developer Nuances

1. Keep tool logic separate from transport code.
2. Use deterministic fallbacks for demos and tests.
3. Preserve evidence trails for financial research.
4. Do not let the LLM silently invent missing market data.
5. Start with deterministic orchestration, then add LLM reasoning once behavior is testable.
6. Treat generated reports as research briefs, not trade recommendations.

## Live Adapters with Offline Fallback

The project now supports live providers without making them mandatory:

- `get_market_data(..., provider="yfinance")` tries Yahoo Finance through `yfinance`.
- `search_news(..., provider="yahoo_rss")` tries Yahoo Finance RSS.
- `search_arxiv(..., provider="arxiv")` tries the official arXiv Atom API.
- `provider="auto"` uses live data unless `FIN_RESEARCH_LIVE=0`.

The important production lesson is that tool contracts should stay stable even when providers change. The agents still receive `MarketBar`, `NewsItem`, and `PaperItem` objects whether data came from offline samples, Yahoo Finance, arXiv, or a future paid vendor.

In real systems, provider failures are normal: network timeouts, missing keys, rate limits, stale data, changed schemas, or empty responses. That is why the adapter catches provider failures and falls back to deterministic data for demos. In production, we would also log the fallback event and surface data freshness in the final report.

## Product Surfaces

The same core agent system is exposed through multiple surfaces:

- CLI: fastest developer workflow.
- API: integration point for internal tools and dashboards.
- UI: human-friendly demo surface.
- MCP: tool server for agentic clients and automated workflows.

This is how real internal tooling evolves: the core logic stays in reusable modules, and different surfaces wrap that core depending on who or what needs to use it.

## Observability and Evaluation

The system records:

- run ID
- selected agents
- tool calls
- providers used
- fallback count
- tool latency
- evidence count
- simple report-quality checks

This is MLOps thinking applied to agentic systems. A trading or research team needs to know not only what answer was produced, but which tools produced it, whether live data failed, how much evidence was used, and whether core risk sections were present.

## Current vs Fully Autonomous

Fast local runtime:

```text
CLI/API/UI -> Supervisor -> Specialist Agents -> ToolGateway -> Tools
```

Standalone tool-server runtime:

```text
MCP client -> MCP server -> Tools
```

Recommended real MCP runtime:

```text
CLI/API/UI -> Supervisor -> Specialist Agents -> McpStdioToolGateway -> MCP server subprocess -> Tools
```

Optional local-LLM runtime:

```text
CLI/API/UI -> LLM-assisted Supervisor -> Specialists -> MCP client -> MCP server -> Tools -> Critic -> LLM-assisted Report
```

The project uses a real persistent MCP client session in `mcp-stdio` mode. The `mcp` mode remains an in-process compatibility shim and should not be described as network or stdio MCP transport.

## Tool Gateway

The `ToolGateway` is the adapter between agents and tools. Agents ask for capabilities such as market data or arXiv search, but they do not care whether the capability is local Python, MCP stdio, or a future remote service.

Why this matters:

- Agents stay stable when tool infrastructure changes.
- MCP can become the runtime path without rewriting every agent.
- Tests can run through local tools quickly.
- Production can route through a stricter protocol boundary.

## Reasoning Evaluation

The current evaluator checks whether every agent:

- produced reasoning
- produced a self-critique
- used tool output or evidence
- contributed to a complete report with risk and next steps

This is structural reasoning evaluation. The benchmark adds expected routes and tool contracts, and Ollama can act as an optional judge. None of these prove financial correctness; that requires expert labels, point-in-time evidence, claim-level citation checks, and out-of-sample validation.

## Engineering Capabilities Demonstrated

- Multi-agent systems: supervisor plus specialist agents.
- Tool calling: every specialist calls explicit tools.
- MCP: tools exposed through a server and called through a persistent stdio client session.
- MLOps mindset: deterministic tests, observability, source auditability, and saved runs.
- Finance: market data, news sentiment, research papers, and risk diagnostics.
- Internal tooling: CLI, API, UI, and MCP contracts support research workflows.
