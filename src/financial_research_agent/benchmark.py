from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any

from financial_research_agent.agents import SupervisorAgent
from financial_research_agent.serialization import to_jsonable

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_PATH = PACKAGE_ROOT / "data" / "eval_prompts.json"
AGENT_KEYS = {
    "Market Data Agent": "market",
    "News Agent": "news",
    "Risk Analysis Agent": "risk",
    "Fundamentals Agent": "fundamentals",
    "Research/Papers Agent": "research",
    "Document Agent": "document",
    "Comparison Agent": "comparison",
}
DATA_TOOLS = {
    "get_market_data",
    "search_news",
    "search_arxiv",
    "search_documents_semantic",
    "get_company_fundamentals",
}
LIVE_DATA_TOOLS = {
    "get_market_data",
    "search_news",
    "search_arxiv",
    "get_company_fundamentals",
}


@contextmanager
def temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _precision_recall(expected: set[str], actual: set[str]) -> tuple[float, float]:
    true_positive = len(expected & actual)
    return true_positive / max(len(actual), 1), true_positive / max(len(expected), 1)


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return round(mean(defined), 4) if defined else None


def _critic_signal_present(signal: str, text: str) -> bool:
    variants = {
        "fallback": ("fallback", "fell back"),
        "contradiction": ("contradiction", "tension", "conflict"),
        "unsupported": ("unsupported", "no options", "no forex", "no commodity"),
    }
    return any(variant in text for variant in variants.get(signal, (signal,)))


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    with temporary_environment({str(k): str(v) for k, v in case.get("env", {}).items()}):
        live_requested = os.getenv("FIN_RESEARCH_LIVE", "0") != "0"
        report, trace = SupervisorAgent().run_with_trace(case["query"])

    actual_agents = {
        AGENT_KEYS[finding.agent_name]
        for finding in report.findings
        if finding.agent_name in AGENT_KEYS
    }
    actual_tools = {
        tool.tool_name
        for finding in report.findings
        if finding.agent_name != "Critic Agent"
        for tool in finding.tool_results
    }
    expected_agents = set(case.get("expected_agents", []))
    expected_tools = set(case.get("expected_tools", []))
    forbidden_agents = set(case.get("forbidden_agents", []))
    forbidden_tools = set(case.get("forbidden_tools", []))
    routing_precision, routing_recall = _precision_recall(expected_agents, actual_agents)
    tool_precision, tool_recall = _precision_recall(expected_tools, actual_tools)
    forbidden_agent_hits = sorted(actual_agents & forbidden_agents)
    forbidden_tool_hits = sorted(actual_tools & forbidden_tools)

    results = [result for finding in report.findings for result in finding.tool_results]
    data_results = [result for result in results if result.tool_name in DATA_TOOLS]
    non_live_providers = {
        "offline",
        "local",
        "mcp",
        "mcp-vector",
        "mcp-semantic",
        "semantic-faiss-or-vector-fallback",
    }
    live_capable_results = [
        result for result in data_results if result.tool_name in LIVE_DATA_TOOLS
    ]
    live_results = [
        result for result in live_capable_results if result.provider not in non_live_providers
    ]
    live_provider_success_rate = (
        len(live_results) / len(live_capable_results)
        if live_requested and live_capable_results
        else None
    )
    fallback_rate = (
        sum(result.fallback_used for result in data_results) / len(data_results)
        if data_results
        else 0.0
    )

    evidence_categories = {evidence_id.split("-", 1)[0] for evidence_id in report.evidence_registry}
    expected_categories = set(case.get("expected_evidence_categories", []))
    evidence_category_recall = len(evidence_categories & expected_categories) / max(
        len(expected_categories), 1
    )
    expected_contradiction = case.get("expected_contradiction")
    contradiction_detected = bool(report.evaluation.get("contradictions"))
    contradiction_correct = (
        contradiction_detected == bool(expected_contradiction)
        if expected_contradiction is not None
        else None
    )
    critic = next(
        (finding for finding in report.findings if finding.agent_name == "Critic Agent"), None
    )
    critic_text = " ".join(critic.details).lower() if critic else ""
    expected_critic_signals = [
        str(signal).lower() for signal in case.get("expected_critic_signals", [])
    ]
    critic_detection_rate = (
        sum(_critic_signal_present(signal, critic_text) for signal in expected_critic_signals)
        / max(len(expected_critic_signals), 1)
        if expected_critic_signals
        else None
    )
    limitation_terms = [str(term).lower() for term in case.get("expected_limitation_terms", [])]
    analysis_text = " ".join(
        [
            report.thesis,
            *report.risks,
            *report.next_steps,
            *(detail for finding in report.findings for detail in finding.details),
        ]
    ).lower()
    limitation_acknowledged = (
        any(term in analysis_text for term in limitation_terms) if limitation_terms else None
    )

    grounding = report.evaluation.get("grounding", {})
    report_completeness = mean(
        [
            bool(report.thesis),
            bool(report.findings),
            bool(report.risks),
            bool(report.next_steps),
            bool(report.evidence_registry),
            all(finding.reasoning and finding.critique for finding in report.findings),
        ]
    )
    trace_summary = trace.to_summary()
    exact_route = actual_agents == expected_agents and not forbidden_agent_hits
    exact_tools = expected_tools <= actual_tools and not forbidden_tool_hits
    primary_scores = [
        routing_precision,
        routing_recall,
        tool_precision,
        tool_recall,
        evidence_category_recall,
        float(grounding.get("citation_coverage", 0.0)),
        float(grounding.get("claim_support_rate", 0.0)),
        report_completeness,
    ]
    if contradiction_correct is not None:
        primary_scores.append(float(contradiction_correct))
    if critic_detection_rate is not None:
        primary_scores.append(critic_detection_rate)
    if limitation_acknowledged is not None:
        primary_scores.append(float(limitation_acknowledged))

    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "query": case["query"],
        "known_failure_conditions": case.get("known_failure_conditions", []),
        "score": round(mean(primary_scores), 4),
        "routing_precision": round(routing_precision, 4),
        "routing_recall": round(routing_recall, 4),
        "exact_route": exact_route,
        "missing_agents": sorted(expected_agents - actual_agents),
        "unexpected_agents": sorted(actual_agents - expected_agents),
        "forbidden_agent_hits": forbidden_agent_hits,
        "tool_selection_precision": round(tool_precision, 4),
        "tool_selection_recall": round(tool_recall, 4),
        "tool_selection_accuracy": exact_tools,
        "missing_tools": sorted(expected_tools - actual_tools),
        "forbidden_tool_hits": forbidden_tool_hits,
        "live_provider_success_rate": live_provider_success_rate,
        "fallback_rate": round(fallback_rate, 4),
        "citation_coverage": grounding.get("citation_coverage", 0.0),
        "claim_to_evidence_support_rate": grounding.get("claim_support_rate", 0.0),
        "unresolved_citations": grounding.get("unresolved_citations", []),
        "evidence_category_recall": round(evidence_category_recall, 4),
        "contradiction_expected": expected_contradiction,
        "contradiction_detected": contradiction_detected,
        "contradiction_correct": contradiction_correct,
        "critic_detection_rate": critic_detection_rate,
        "limitation_acknowledged": limitation_acknowledged,
        "report_completeness": round(report_completeness, 4),
        "structural_quality_score": report.evaluation.get("score", 0.0),
        "llm_judge_score": report.evaluation.get("llm_judge", {}).get("score"),
        "end_to_end_latency_ms": trace_summary["end_to_end_latency_ms"],
        "provider_latency_ms": trace_summary["provider_latency_ms"],
        "transport_latency_ms": trace_summary["transport_latency_ms"],
        "ollama_latency_ms": trace_summary["ollama_latency_ms"],
        "mcp_protocol_failures": trace_summary["mcp_protocol_failures"],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "score",
        "routing_precision",
        "routing_recall",
        "tool_selection_precision",
        "tool_selection_recall",
        "fallback_rate",
        "citation_coverage",
        "claim_to_evidence_support_rate",
        "evidence_category_recall",
        "report_completeness",
        "structural_quality_score",
        "end_to_end_latency_ms",
        "provider_latency_ms",
        "transport_latency_ms",
        "ollama_latency_ms",
    ]
    aggregate = {
        name: _mean_defined([result.get(name) for result in results]) for name in metric_names
    }
    aggregate.update(
        {
            "exact_route_rate": round(mean(result["exact_route"] for result in results), 4)
            if results
            else 0.0,
            "tool_selection_accuracy": round(
                mean(result["tool_selection_accuracy"] for result in results), 4
            )
            if results
            else 0.0,
            "live_provider_success_rate": _mean_defined(
                [result.get("live_provider_success_rate") for result in results]
            ),
            "contradiction_recall": _mean_defined(
                [
                    float(result["contradiction_detected"])
                    for result in results
                    if result.get("contradiction_expected") is True
                ]
            ),
            "critic_detection_rate": _mean_defined(
                [result.get("critic_detection_rate") for result in results]
            ),
            "mcp_protocol_failures": sum(result["mcp_protocol_failures"] for result in results),
        }
    )
    return aggregate


def run_benchmark(
    path: Path = DEFAULT_BENCHMARK_PATH,
    limit: int | None = None,
    environment: dict[str, str] | None = None,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if case_ids is not None:
        selected = set(case_ids)
        cases = [case for case in cases if case["id"] in selected]
    elif limit is not None:
        cases = cases[:limit]
    base_environment = {
        "FIN_RESEARCH_LIVE": "0",
        "FIN_RESEARCH_TOOL_RUNTIME": "local",
        **(environment or {}),
    }
    with temporary_environment(base_environment):
        results = [evaluate_case(case) for case in cases]
    return {
        "benchmark_path": str(path),
        "case_count": len(results),
        "metrics": _aggregate(results),
        "results": results,
    }


def run_ablation(path: Path = DEFAULT_BENCHMARK_PATH, limit: int | None = None) -> dict[str, Any]:
    all_cases = json.loads(path.read_text(encoding="utf-8"))
    selected_ids: list[str] | None = None
    if limit is not None:
        selected_ids = []
        seen_categories: set[str] = set()
        for case in all_cases:
            category = case.get("category", "unspecified")
            if category not in seen_categories:
                selected_ids.append(case["id"])
                seen_categories.add(category)
            if len(selected_ids) == limit:
                break
        if len(selected_ids) < limit:
            selected_ids.extend(
                case["id"] for case in all_cases if case["id"] not in selected_ids
            )
            selected_ids = selected_ids[:limit]
    deterministic = run_benchmark(
        path, environment={"FIN_RESEARCH_LLM": "off"}, case_ids=selected_ids
    )
    ollama = run_benchmark(
        path, environment={"FIN_RESEARCH_LLM": "ollama"}, case_ids=selected_ids
    )
    comparable = [
        "score",
        "routing_precision",
        "routing_recall",
        "tool_selection_accuracy",
        "citation_coverage",
        "claim_to_evidence_support_rate",
        "contradiction_recall",
        "critic_detection_rate",
        "end_to_end_latency_ms",
    ]
    deltas = {}
    for metric in comparable:
        baseline = deterministic["metrics"].get(metric)
        candidate = ollama["metrics"].get(metric)
        deltas[metric] = (
            round(candidate - baseline, 4)
            if isinstance(candidate, int | float) and isinstance(baseline, int | float)
            else None
        )
    return {
        "experiment": "deterministic_vs_ollama",
        "case_count": deterministic["case_count"],
        "selected_case_ids": selected_ids,
        "deterministic": deterministic,
        "ollama": ollama,
        "delta_ollama_minus_deterministic": deltas,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate routing, tools, grounding, reliability, and latency."
    )
    parser.add_argument("--cases", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Compare deterministic reasoning with the configured local Ollama model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.cases)
    summary = run_ablation(path, args.limit) if args.ablation else run_benchmark(path, args.limit)
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
