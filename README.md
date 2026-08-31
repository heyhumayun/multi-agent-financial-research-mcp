# Multi-Agent Financial Research System with MCP

An auditable financial-research workflow that autonomously plans a bounded investigation, routes a query across specialist agents, invokes typed tools through local or real MCP transport, critiques the evidence, and produces a traceable research brief.

The system combines multi-agent orchestration, typed tool calling, MCP transport, retrieval, live-provider adapters, evaluation, and deterministic fallbacks. It produces research briefs and does not place trades or provide investment advice.

## Architecture

```text
CLI / FastAPI / Streamlit
          |
          v
   Supervisor Agent ---- optional Ollama planner
          |
          +-- Market Agent ------ market bars, returns, volatility
          +-- News Agent -------- headlines and lightweight sentiment
          +-- Fundamentals Agent  SEC company facts and reported metrics
          +-- Comparison Agent --- aligned multi-symbol risk comparison
          +-- Research Agent ---- arXiv papers
          +-- Document Agent ---- semantic/vector RAG over local notes
          +-- Risk Agent -------- volatility and maximum drawdown
          |
          v
      Critic Agent ------ evidence and coverage checks
          |
          v
   Synthesis Agent ------ optional Ollama synthesis
          |
          v
 Report + evaluation + trace + optional saved artifacts

Tool path in the recommended runtime:
Agent -> ToolGateway -> MCP stdio client -> MCP server -> tool adapter -> provider
```

## What Is Agentic Here?

The tools fetch or calculate data. The agents decide how that evidence is used:

- The supervisor selects specialist agents from the query.
- The supervisor records a research plan, executes it, inspects coverage, and may request one evidence-recovery pass before stopping.
- Specialists call tools, transform raw results into findings, and state limitations.
- The critic checks fallback usage and missing evidence categories.
- The synthesis agent combines findings into one risk-aware thesis.
- Optional Ollama reasoning can replace deterministic planning and prose while retaining a free fallback.

This is bounded agent autonomy: the workflow can plan and execute research, but it cannot place trades or modify external systems.

## Quick Start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[api,ui,market]"
```

Run a reproducible offline analysis through the real MCP client/server path:

```bash
FIN_RESEARCH_LIVE=0 \
FIN_RESEARCH_TOOL_RUNTIME=mcp-stdio \
financial-research-agent "Assess NVDA AI infrastructure risk and research" --save
```

Inspect machine-readable output and the execution trace:

```bash
FIN_RESEARCH_LIVE=0 \
FIN_RESEARCH_TOOL_RUNTIME=mcp-stdio \
financial-research-agent "Assess NVDA AI infrastructure risk and research" --json --trace
```

Saved runs contain `report.md`, `report.json`, and `trace.json` under `runs/<run_id>/`. The directory is intentionally ignored by Git.

## Interfaces

CLI:

```bash
financial-research-agent "Assess AMD competition and downside risk"
```

API:

```bash
financial-research-api
```

Then use `GET /health`, `GET /tools`, or `POST /research` with:

```json
{"query": "Assess NVDA AI infrastructure risk"}
```

Streamlit UI:

```bash
set -a
source .env
set +a
financial-research-ui
```

Open the printed local URL, usually `http://localhost:8501`. Enter a question in the
`Research question` field and select `Run full investigation`. The desk shows the thesis and
agent findings first, then `Evidence & sources`, `Quality checks`, and `Execution trace` tabs.
The sidebar controls live versus offline data, offline fallback, and optional Ollama reasoning.

Standalone MCP server:

```bash
financial-research-mcp
```

## Tool Runtime Modes

- `local`: direct Python calls through the gateway; fastest for unit tests.
- `mcp`: an in-process compatibility shim with the same gateway shape. It does not cross an MCP transport.
- `mcp-stdio`: a real MCP client and server subprocess with one persistent session per research run.

Use `mcp-stdio` when demonstrating MCP. The server returns provider and fallback provenance with live-source results, so traces remain truthful across the protocol boundary.

## Data Sources

Default `auto` behavior attempts free live sources and falls back to bundled deterministic fixtures when enabled:

- Market: yfinance; optional Alpha Vantage, Polygon, or Financial Modeling Prep keys.
- News: Yahoo Finance RSS; optional Finnhub or NewsAPI key.
- Papers: arXiv Atom API.
- Documents: bundled research notes searched by lexical, token-vector, or optional semantic FAISS retrieval.
- Risk: locally calculated log returns, annualized volatility, and maximum drawdown.
- Fundamentals: SEC Company Facts API when live mode is enabled; bundled demo facts offline.

```bash
set -a
source .env
set +a
FIN_RESEARCH_MARKET_PROVIDER=alpha_vantage \
financial-research-agent "Assess PLTR earnings risk"
```

To prevent fallback and fail loudly when a live source is unavailable:

```bash
FIN_RESEARCH_OFFLINE_FALLBACK=0 financial-research-agent "Assess NVDA risk"
```

## Optional Local LLM

Deterministic planning and synthesis work without an API or model bill. For free local reasoning, install Ollama separately and run:

```bash
ollama pull llama3.2:3b
FIN_RESEARCH_LLM=ollama \
FIN_RESEARCH_OLLAMA_MODEL=llama3.2:3b \
financial-research-agent "Assess NVDA AI infrastructure risk and research"
```

Ollama can assist supervisor planning, specialist explanations, critique, synthesis, and an optional LLM judge. If it is unavailable, each operation falls back to deterministic logic.

## Semantic RAG

The default document tool uses a dependency-free token-vector cosine retriever. Install the larger optional stack to use sentence-transformer embeddings with FAISS nearest-neighbor search:

```bash
python3 -m pip install -e ".[rag]"
```

The bundled documents and evaluation cases are Python package data, so they remain available after wheel installation rather than only from a source checkout.

## Evaluation and Verification

```bash
python3 -m pip install -e ".[dev,api]"
ruff check src tests
python3 -m unittest discover -s tests -v
FIN_RESEARCH_LIVE=0 financial-research-eval
financial-research-eval --ablation --limit 6
```

The 30-case labelled benchmark reports routing and tool precision/recall, provider success and fallback rates, citation coverage, claim support, contradiction/critic recall, report completeness, MCP failures, and stage-level latency. The ablation command compares deterministic orchestration with the local Ollama model on a stratified sample. See [EVALUATION.md](EVALUATION.md) for methodology, measured results, and limitations.

The benchmark does **not** prove investment correctness or alpha. Those require time-aligned datasets, independent expert labels, source-entailment evaluation, and out-of-sample financial validation.

GitHub Actions runs linting, tests, and the offline benchmark on Python 3.10 and 3.12. A `Dockerfile` packages the FastAPI service.

## Project Layout

```text
src/financial_research_agent/
  agents/          supervisor, specialists, comparison, fundamentals, critic, synthesis
  mcp/             MCP tool contracts and stdio server
  tools/           providers, document retrieval, risk calculations
  data/            packaged offline fixtures and benchmark cases
  tool_gateway.py  local/shim/real-MCP runtime boundary
  reasoning.py     deterministic and optional Ollama reasoning
  evaluation.py    report, freshness, contradiction, and grounding diagnostics
  grounding.py     evidence IDs, registry, and thesis citations
  observability.py provider/transport/LLM/stage latency, evidence, failures, fallbacks
  cli.py api.py ui.py
tests/             unit, routing, benchmark, and MCP integration tests
```

## Current Scope

- This is an autonomous research-assistance prototype, not investment advice or execution infrastructure.
- Free sources are delayed, rate-limited, and not exchange-grade.
- Headline sentiment is rule-based; it is not a validated finance-language model.
- The deterministic supervisor uses keyword routing; optional Ollama planning is not guaranteed to be correct.
- Multi-ticker comparison is supported for basic return and volatility context; portfolio optimization is not implemented.
- SEC Company Facts are summarized as reported fundamentals; full filing retrieval and claim-level excerpts are future work.
- The critic detects configured fallback, coverage, and market/news contradiction classes; open-ended claim resolution remains future work.
- Confidence values are engineering heuristics, not calibrated probabilities.
- Offline market/news/paper records are synthetic fixtures and are visibly marked as such.
- “Autonomous” is deliberately bounded: there is no open-ended self-modification, trade execution, or unreviewed external action.

See [PRODUCT_READINESS.md](PRODUCT_READINESS.md) for the self-audit and [LEARNING_NOTES.md](LEARNING_NOTES.md) for concept explanations.

## License

MIT. See [LICENSE](LICENSE).
