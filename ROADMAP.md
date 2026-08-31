# Roadmap

## Completed in v0.1.0

- Typed financial domain models and stable tool contracts.
- Supervisor, five specialists, critic, and synthesis workflow.
- Real persistent MCP stdio session plus local testing runtime.
- Live/free provider adapters with deterministic fallback and provenance.
- Optional Ollama reasoning and optional sentence-transformer/FAISS RAG.
- CLI, API, UI, tracing, saved artifacts, benchmark suite, tests, CI, and Docker.

## Completed in v0.2.0

- Multi-ticker detection with aligned comparative return and volatility analysis.
- Fundamentals specialist with SEC Company Facts adapter and offline fallback.
- Fundamentals and comparison tools exposed through the real MCP server contract.
- Cross-agent tension detection for conflicting market and news signals.

## Completed in v0.3.0

- Thirty-case labelled multi-agent evaluation dataset with known failure conditions.
- Routing/tool precision and recall, grounding, contradiction, critic, provider, and latency metrics.
- Deterministic-versus-Ollama ablation runner with stratified sampling.
- Stable evidence IDs, report registry, finding citations, and cited synthesis thesis.
- Provider, MCP transport, Ollama-operation, stage, and true wall-clock telemetry.
- Failure injection for provider, MCP, Ollama, stale-data, SEC, and contradiction paths.

## Focused v0.2 Candidates

1. Add full SEC filing retrieval and immutable excerpt-level entailment checks.
2. Cache embeddings and live responses with explicit freshness policies.
3. Add contradiction checks across fundamentals, news, market, and research claims.
4. Independently adjudicate evaluation labels and expand the Ollama ablation sample.

## Production Track

- Point-in-time licensed datasets and market-calendar normalization.
- Calibrated models and expert-labeled evaluation sets.
- Durable storage, authentication, secrets management, rate limits, and telemetry.
- Human review gates and immutable evidence lineage.
- Deployment, load tests, service objectives, and incident procedures.

The project intentionally stops before trade execution. Adding brokerage access would increase risk without improving the core research workflow.
