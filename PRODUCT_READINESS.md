# Product Readiness and Self-Audit

## Release Decision

Repository release: ready. Research experimentation: ready with source warnings. Investment or production trading use: not ready.

## Verified Capabilities

- Supervisor routing across market, news, risk, paper, and document specialists.
- Multi-ticker comparison and a fundamentals specialist using SEC Company Facts when live mode is enabled.
- Critic and synthesis stages with deterministic or optional local-LLM reasoning.
- Real MCP stdio client/server transport with one persistent session per run.
- Provider and fallback provenance preserved across the MCP boundary.
- Live adapters with explicit deterministic fallback.
- Semantic FAISS RAG when optional dependencies are installed; token-vector fallback otherwise.
- CLI, FastAPI, Streamlit, and standalone MCP interfaces.
- Run IDs, selected agents, tool latency, provider, fallback, and evidence traces.
- Bounded autonomous review loop with recorded plan, follow-up decision, iteration count, and stop reason.
- Saved Markdown/JSON reports and JSON traces.
- Unit, routing, benchmark, and real MCP integration tests.
- Thirty labelled routing/reliability cases and deterministic-versus-Ollama ablation tooling.
- Stable evidence IDs with source registry, finding citations, and thesis citations.
- Separate provider, MCP transport, Ollama-operation, stage, and wall-clock telemetry.
- GitHub Actions, Dockerfile, environment template, package data, and MIT license.

## Audit Fixes Applied for v0.1.0

1. Fixed loss of live-provider and fallback metadata across real MCP calls.
2. Moved fixtures into Python package data so installed distributions can find them.
3. Added immediate MCP worker-failure detection instead of waiting for a full tool timeout.
4. Added empty-query validation and support for explicit ticker symbols beyond a hard-coded shortlist.
5. Added tests covering MCP provenance, query validation, and unlisted tickers.
6. Corrected documentation to distinguish the in-process MCP-shaped shim from real stdio MCP.
7. Excluded secrets, generated runs, caches, and build artifacts from version control.

## Known Gaps Before Industry Deployment

- Licensed, point-in-time, exchange-grade market and news data.
- Corporate actions, survivorship-bias controls, market calendars, and timezone normalization.
- Full SEC filing retrieval, estimates, ownership, options, and event-calendar tools.
- Portfolio-level risk aggregation and portfolio construction.
- Immutable source excerpts and entailment checks beyond the current source-level evidence IDs.
- Calibrated sentiment and confidence models with labeled validation data.
- Contradiction detection, source reliability ranking, and human approval workflows.
- Persistent database, caching, retries with backoff, rate-limit budgets, and distributed tracing.
- Authentication, authorization, secrets management, API quotas, and threat modeling.
- Load testing, deployment manifests, service-level objectives, and production monitoring.
- Financial outcome evaluation using time-aligned, out-of-sample data.

## Evaluation Interpretation

The included benchmark is a labelled software-behavior and reliability suite. It measures routing/tool precision and recall, evidence linkage, critic/contradiction detection, provider fallback, and latency. It does not mean the thesis is financially correct. Production evaluation still requires independent labels, excerpt-level entailment, expert review, and forward-looking financial validation. See `EVALUATION.md` for the measured baseline and ablation.

## Project Summary

Built an MCP-enabled multi-agent financial research system with supervisor routing, specialist agents, live/offline provider adapters, semantic document retrieval, critic review, traceable tool execution, benchmark evaluation, CI, and reproducible research artifacts.
