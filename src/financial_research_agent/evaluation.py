from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from financial_research_agent.domain import ResearchReport
from financial_research_agent.grounding import extract_citations
from financial_research_agent.reasoning import ReasoningEngine


def evaluate_report(report: ResearchReport) -> dict:
    checks: list[str] = []
    score = 0.0

    agent_names = {finding.agent_name for finding in report.findings}
    required = {"Market Data Agent", "News Agent", "Risk Analysis Agent"}
    missing = sorted(required - agent_names)
    if missing:
        checks.append(f"Missing core agents: {', '.join(missing)}.")
    else:
        checks.append("Core agents present: market, news, and risk.")
        score += 0.2

    evidence_count = sum(
        len(result.evidence)
        for finding in report.findings
        for result in finding.tool_results
        if not result.provider.startswith("internal-")
    )
    if evidence_count >= 2:
        checks.append(f"Evidence trail present with {evidence_count} source references.")
        score += 0.15
    else:
        checks.append("Evidence trail is thin; add more cited sources.")

    if report.risks:
        checks.append("Risk section present.")
        score += 0.1
    else:
        checks.append("Risk section missing.")

    if report.next_steps:
        checks.append("Next steps present for follow-up research.")
        score += 0.05

    avg_confidence = sum(finding.confidence for finding in report.findings) / max(
        len(report.findings), 1
    )
    if avg_confidence >= 0.65:
        checks.append(f"Average agent confidence is acceptable at {avg_confidence:.2f}.")
        score += 0.1
    else:
        checks.append(f"Average agent confidence is low at {avg_confidence:.2f}.")

    findings_with_reasoning = [finding for finding in report.findings if finding.reasoning]
    findings_with_critique = [finding for finding in report.findings if finding.critique]
    if len(findings_with_reasoning) == len(report.findings):
        checks.append("Every agent included reasoning.")
        score += 0.1
    else:
        checks.append("Some agents are missing reasoning explanations.")

    if len(findings_with_critique) == len(report.findings):
        checks.append("Every agent included a self-critique.")
        score += 0.1
    else:
        checks.append("Some agents are missing self-critiques.")

    unsupported_findings = [
        finding.agent_name
        for finding in report.findings
        if not any(result.evidence or result.output is not None for result in finding.tool_results)
    ]
    if unsupported_findings:
        checks.append(f"Unsupported findings need evidence: {', '.join(unsupported_findings)}.")
    else:
        checks.append("Every finding is connected to tool output or evidence.")
        score += 0.1

    freshness = _freshness_summary(report)
    checks.extend(freshness["checks"])
    if not freshness["stale_tools"]:
        score += 0.05
    if not freshness["fallback_tools"]:
        score += 0.05

    contradictions = detect_contradictions(report.findings)
    if contradictions:
        checks.append(f"Cross-agent tensions detected: {len(contradictions)}.")
    else:
        checks.append("No obvious cross-agent tensions detected.")

    llm_judge = ReasoningEngine().judge_report(report.query, report.to_markdown())
    if llm_judge:
        checks.append(f"LLM judge score {llm_judge['score']:.2f}: {llm_judge['rationale']}")

    grounding = grounding_summary(report)
    checks.extend(grounding["checks"])

    return {
        "score": round(score, 2),
        "checks": checks,
        "freshness": freshness,
        "contradictions": contradictions,
        "llm_judge": llm_judge,
        "grounding": grounding,
    }


def detect_contradictions(findings: list[Any]) -> list[str]:
    headlines = {finding.agent_name: finding.headline.lower() for finding in findings}
    tensions: list[str] = []
    market = headlines.get("Market Data Agent", "")
    news = headlines.get("News Agent", "")
    if "positive" in market and any(word in news for word in ("cautious", "negative")):
        tensions.append("Market momentum is positive while news sentiment is cautious or negative.")
    if "negative" in market and "supportive" in news:
        tensions.append("Market momentum is negative while news sentiment is supportive.")
    return tensions


def grounding_summary(report: ResearchReport) -> dict[str, Any]:
    specialist_findings = [
        finding for finding in report.findings if finding.agent_name != "Critic Agent"
    ]
    cited_findings = [finding for finding in specialist_findings if finding.citations]
    total_claims = sum(len(finding.details) for finding in specialist_findings)
    supported_claims = sum(len(finding.details) for finding in cited_findings)
    cited_ids = {citation for finding in report.findings for citation in finding.citations} | set(
        extract_citations(report.thesis)
    )
    unresolved = sorted(cited_ids - set(report.evidence_registry))
    thesis_citations = extract_citations(report.thesis)
    citation_coverage = len(cited_findings) / max(len(specialist_findings), 1)
    claim_support_rate = supported_claims / max(total_claims, 1)
    checks = [
        f"Citation coverage is {citation_coverage:.0%} across specialist findings.",
        f"Claim-to-evidence support rate is {claim_support_rate:.0%}.",
    ]
    if thesis_citations:
        checks.append(f"Thesis cites {len(thesis_citations)} evidence IDs.")
    else:
        checks.append("Thesis has no claim-level evidence citation.")
    if unresolved:
        checks.append(f"Unresolved evidence IDs: {', '.join(unresolved)}.")
    else:
        checks.append("Every cited evidence ID resolves in the registry.")
    return {
        "citation_coverage": round(citation_coverage, 4),
        "claim_support_rate": round(claim_support_rate, 4),
        "thesis_citation_count": len(thesis_citations),
        "unresolved_citations": unresolved,
        "checks": checks,
    }


def _freshness_summary(report: ResearchReport) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    tool_dates: dict[str, list[date]] = {}
    fallback_tools: list[str] = []
    stale_tools: list[str] = []

    for finding in report.findings:
        for result in finding.tool_results:
            if result.fallback_used:
                fallback_tools.append(result.tool_name)
            dates = _extract_dates(result.output)
            if dates:
                tool_dates.setdefault(result.tool_name, []).extend(dates)

    checks: list[str] = []
    newest_by_tool: dict[str, str] = {}
    for tool_name, dates in sorted(tool_dates.items()):
        newest = max(dates)
        newest_by_tool[tool_name] = newest.isoformat()
        age_days = (today - newest).days
        if tool_name == "get_market_data" and age_days > 7:
            checks.append(f"Freshness warning: market data is {age_days} days old.")
            stale_tools.append(tool_name)
        elif tool_name == "search_news" and age_days > 14:
            checks.append(f"Freshness warning: news data is {age_days} days old.")
            stale_tools.append(tool_name)
        elif tool_name == "get_company_fundamentals" and age_days > 550:
            checks.append(f"Freshness warning: fundamentals period is {age_days} days old.")
            stale_tools.append(tool_name)
        else:
            checks.append(f"Freshness check: {tool_name} latest date is {newest.isoformat()}.")

    if fallback_tools:
        checks.append(f"Fallback used by tools: {', '.join(sorted(set(fallback_tools)))}.")
    else:
        checks.append("No live-provider fallback was needed.")

    return {
        "newest_by_tool": newest_by_tool,
        "fallback_tools": sorted(set(fallback_tools)),
        "stale_tools": sorted(set(stale_tools)),
        "checks": checks,
    }


def _extract_dates(value: Any) -> list[date]:
    if isinstance(value, date):
        return [value]
    if isinstance(value, list | tuple):
        dates: list[date] = []
        for item in value:
            dates.extend(_extract_dates(item))
        return dates
    if hasattr(value, "date") and isinstance(value.date, date):
        return [value.date]
    if hasattr(value, "published") and isinstance(value.published, date):
        return [value.published]
    if hasattr(value, "period"):
        period = str(value.period)
        try:
            return [date.fromisoformat(period)]
        except ValueError:
            pass
        match = re.search(r"\b(20\d{2})\b", period)
        if match:
            return [date(int(match.group(1)), 12, 31)]
    if isinstance(value, dict):
        dates = []
        for item in value.values():
            dates.extend(_extract_dates(item))
        return dates
    return []
