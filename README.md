# Multi-Agent Financial Research with MCP

## Abstract

This project implements a bounded, auditable financial-research workflow. A supervisor routes a question to specialist agents, tools collect evidence through a local gateway or Model Context Protocol (MCP) transport, and a critic reviews coverage before a research brief is produced. A labelled offline benchmark evaluates routing, tool selection, evidence linkage, failure handling, and latency. The system is a research-assistance prototype; it does not execute trades or establish investment alpha.

## Research objectives

- Evaluate whether specialist-agent orchestration selects appropriate tools and evidence for financial-research questions.
- Compare deterministic orchestration with optional local-LLM reasoning on the same labelled cases.
- Make provider failures, offline fallbacks, source links, and execution traces visible in the final report.
- Separate software-workflow quality from the factual and financial validity of a research conclusion.

## Data

Live adapters can request market data from yfinance, news from Yahoo Finance RSS, papers from arXiv, and reported fundamentals from SEC Company Facts. Optional market and news providers require their respective credentials. Local documents can be searched with a token-vector retriever or optional sentence-transformer/FAISS retrieval.

The reproducible offline mode uses packaged deterministic fixtures, including synthetic market, news, and paper records and demonstration fundamentals. The 30-case evaluation set contains project-authored labels for required agents, tools, evidence categories, contradictions, and failure conditions. Neither the fixtures nor the labels are a point-in-time financial dataset or independently adjudicated analyst judgments.

## Methodology

The supervisor selects a bounded set of market, news, fundamentals, comparison, research, document, and risk specialists. Specialists call typed tools and attach evidence identifiers to their findings. A critic checks configured coverage, fallback, and contradiction conditions; the supervisor may make one evidence-recovery pass. Synthesis produces a brief, structured output, and an execution trace. Optional Ollama reasoning can assist planning and synthesis; deterministic behavior remains available without an LLM.

The tool gateway supports direct local calls, an in-process MCP-shaped compatibility mode, and a real MCP client/server subprocess over stdio. Only `mcp-stdio` crosses an MCP transport boundary. Saved runs contain `report.md`, `report.json`, and `trace.json` under `runs/<run_id>/`.

## Experimental setup

The primary evaluation is a 30-case deterministic offline benchmark. It measures routing and tool selection, citation coverage, claim-to-evidence linkage, contradiction and critic detection, fallback behavior, and stage-level latency. A separate six-case, category-stratified ablation compares deterministic orchestration with local `llama3.2:3b` reasoning. Failure-injection tests cover unavailable providers, malformed MCP responses, unavailable Ollama, stale fixtures, and contradictory findings. See [EVALUATION.md](EVALUATION.md) for definitions and protocol details.

## Results

The recorded offline benchmark from 2026-08-31 reported:

| Metric | Result |
| --- | ---: |
| Overall benchmark score | 0.9799 |
| Routing precision / recall | 1.0000 / 1.0000 |
| Tool-selection accuracy | 1.0000 |
| Citation coverage | 0.9189 |
| Claim-to-evidence support | 0.9716 |
| Contradiction recall | 1.0000 |
| Critic detection rate | 1.0000 |

In the six-case local-model ablation, the deterministic and Ollama configurations scored 0.9668 and 0.9407, respectively. Exact-route rate fell from 1.0000 to 0.5000 with Ollama, while mean end-to-end latency rose from 0.98 ms to 50,679 ms in that sample. These are software-workflow measurements, not estimates of trading or investment performance.

## Limitations

- Offline fixtures are synthetic, and the project-authored labels lack independent analyst adjudication.
- Live public sources may be delayed, rate-limited, incomplete, or unavailable; fallback data is explicitly marked.
- Evidence identifiers establish source linkage, not excerpt-level entailment or factual correctness.
- Headline sentiment, confidence values, and deterministic query routing are heuristics rather than calibrated financial models.
- The six-case LLM ablation is too small for a general model-quality conclusion; offline latency is not representative of live providers.
- The system does not implement point-in-time market-data controls, portfolio construction, execution, or financial outcome validation.

## Repository structure

```text
src/financial_research_agent/
  agents/          supervisor, specialists, critic, and synthesis
  mcp/             tool contracts and stdio server
  tools/           provider adapters, retrieval, and risk calculations
  data/            packaged fixtures and evaluation cases
  benchmark.py     labelled workflow evaluation
  cli.py           command-line interface
  api.py           FastAPI interface
  ui.py            Streamlit interface
tests/             unit and integration tests
EVALUATION.md      evaluation protocol and recorded measurements
PRODUCT_READINESS.md  capability and deployment-gap assessment
```

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev,api,ui,market]"
```

The optional `rag` extra installs sentence-transformer and FAISS dependencies. Live providers and Ollama are not required for the offline workflow.

## Usage

Run an offline investigation through the real MCP transport:

```bash
FIN_RESEARCH_LIVE=0 FIN_RESEARCH_TOOL_RUNTIME=mcp-stdio \
  financial-research-agent "Assess NVDA AI infrastructure risk and research" --save
```

Run the recorded evaluation procedure and repository checks:

```bash
FIN_RESEARCH_LIVE=0 financial-research-eval
ruff check src tests
python3 -m unittest discover -s tests -v
```

`financial-research-api`, `financial-research-ui`, and `financial-research-mcp` start the API, Streamlit interface, and standalone MCP server. Live-provider mode may require API credentials; use `FIN_RESEARCH_OFFLINE_FALLBACK=0` to fail when a live source is unavailable instead of accepting fixture fallback.

## References

- [Evaluation methodology and results](EVALUATION.md)
- [Product-readiness assessment](PRODUCT_READINESS.md)
- [MIT license](LICENSE)
