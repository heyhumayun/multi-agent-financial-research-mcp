# Multi-Agent Evaluation Methodology

## Why This Exists

Report fluency is not evidence that an agent system routes correctly, uses the right tools,
grounds its claims, detects failures, or provides a worthwhile latency tradeoff. This project
therefore evaluates observable workflow behavior against human-authored labels rather than
using an LLM judge as the primary correctness target.

## Labelled Dataset

`src/financial_research_agent/data/eval_prompts.json` contains 30 cases across:

- single-company risk and fundamentals;
- two-company comparisons;
- academic and local-document retrieval;
- ambiguous and unlisted symbols;
- missing providers and stale fixtures;
- contradictory market/news evidence;
- unsupported options, FX, and commodity requests;
- multi-source full-stack investigations.

Each case can specify required and forbidden agents/tools, expected evidence categories,
expected contradictions, critic signals, environment overrides, scope limitations, and known
failure conditions. Known failures remain visible in output; they are not converted into passes.

## Metrics

- Routing precision/recall and exact-route rate
- Tool-selection precision/recall and accuracy
- Live-provider success rate and fallback rate
- Citation coverage and claim-to-evidence support rate
- Evidence-category recall
- Contradiction recall and critic detection rate
- Report completeness and structural quality
- Provider, MCP transport, Ollama, and true end-to-end latency
- MCP protocol failure count

Claim support is currently a finding-level proxy: each displayed finding claim is linked to the
evidence IDs attached to that specialist. It does not yet verify immutable source excerpts or
natural-language entailment.

## Reproducible Baseline

```bash
FIN_RESEARCH_LIVE=0 financial-research-eval
```

The 30-case deterministic offline baseline measured on 2026-08-31 produced:

| Metric | Result |
|---|---:|
| Overall benchmark score | 0.9799 |
| Routing precision / recall | 1.0000 / 1.0000 |
| Tool-selection accuracy | 1.0000 |
| Citation coverage | 0.9189 |
| Claim-to-evidence support | 0.9716 |
| Contradiction recall | 1.0000 |
| Critic detection rate | 1.0000 |

Citation coverage is below one because some unlisted or unsupported symbols have no verifiable
source evidence. Offline latency is not representative of live MCP or provider performance.

## Deterministic Versus Ollama Ablation

```bash
financial-research-eval --ablation --limit 6
```

The limit selects one case from each of six distinct categories. On the local
`llama3.2:3b` model:

| Metric | Deterministic | Ollama | Delta |
|---|---:|---:|---:|
| Overall score | 0.9668 | 0.9407 | -0.0261 |
| Routing precision | 1.0000 | 0.8778 | -0.1222 |
| Routing recall | 1.0000 | 1.0000 | 0.0000 |
| Exact-route rate | 1.0000 | 0.5000 | -0.5000 |
| Citation coverage | 0.8667 | 0.8778 | +0.0111 |
| Claim support | 0.9514 | 0.9569 | +0.0055 |
| Mean end-to-end latency | 0.98 ms | 50,679 ms | +50,678 ms |

The model preserved recall but added unnecessary research or fundamentals agents in half the
sample. Small grounding gains did not offset lower routing precision or roughly 50 seconds of
additional latency. The evidence supports deterministic routing for this bounded agent taxonomy,
with Ollama retained as an optional synthesis/review capability and for future experiments.

## Failure Injection

The test suite injects Polygon timeouts, Finnhub 503 responses, malformed MCP payloads, an
unavailable Ollama process, stale fixtures, empty SEC responses, and contradictory findings.
Tests verify retry/fallback provenance, strict-mode failures, quality penalties, deterministic
LLM fallback, critic warnings, and fail-closed MCP decoding.

## Limitations

- Labels were authored for this project and have not been independently adjudicated by analysts.
- The baseline uses deterministic fixtures; it measures workflow behavior, not investment alpha.
- The ablation is a six-case local-model sample, not a statistically powered model comparison.
- Citation resolution proves source linkage, not factual entailment.
- LLM judge scores are reported as diagnostics and are not treated as ground truth.
