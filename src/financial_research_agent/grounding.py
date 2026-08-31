from __future__ import annotations

import re
from dataclasses import replace

from financial_research_agent.domain import AgentFinding

AGENT_PREFIXES = {
    "Market Data Agent": "market",
    "News Agent": "news",
    "Risk Analysis Agent": "risk",
    "Fundamentals Agent": "fundamentals",
    "Research/Papers Agent": "research",
    "Document Agent": "document",
    "Comparison Agent": "comparison",
}
CITATION_PATTERN = re.compile(
    r"\b(?:market|news|risk|fundamentals|research|document|comparison)-\d+\b"
)


def ground_findings(
    findings: list[AgentFinding],
) -> tuple[list[AgentFinding], dict[str, str]]:
    """Assign stable evidence IDs and attach them to the findings they support."""
    registry: dict[str, str] = {}
    counters: dict[str, int] = {}
    grounded: list[AgentFinding] = []

    for finding in findings:
        prefix = AGENT_PREFIXES.get(finding.agent_name)
        citations: list[str] = []
        if prefix:
            for result in finding.tool_results:
                if result.provider.startswith("internal-"):
                    continue
                for source in result.evidence:
                    existing = next(
                        (
                            key
                            for key, value in registry.items()
                            if key.startswith(f"{prefix}-") and value == source
                        ),
                        None,
                    )
                    if existing:
                        citations.append(existing)
                        continue
                    counters[prefix] = counters.get(prefix, 0) + 1
                    evidence_id = f"{prefix}-{counters[prefix]}"
                    registry[evidence_id] = source
                    citations.append(evidence_id)
        grounded.append(replace(finding, citations=list(dict.fromkeys(citations))))

    cited_by_specialists = [
        citation
        for finding in grounded
        if finding.agent_name != "Critic Agent"
        for citation in finding.citations
    ]
    grounded = [
        replace(finding, citations=list(dict.fromkeys(cited_by_specialists)))
        if finding.agent_name == "Critic Agent"
        else finding
        for finding in grounded
    ]
    return grounded, registry


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_PATTERN.findall(text)))


def ensure_thesis_citations(thesis: str, findings: list[AgentFinding]) -> str:
    """Guarantee that a synthesized thesis remains linked to valid evidence IDs."""
    if extract_citations(thesis):
        return thesis
    preferred: list[str] = []
    for finding in findings:
        if finding.agent_name == "Critic Agent":
            continue
        preferred.extend(finding.citations[:1])
    preferred = list(dict.fromkeys(preferred))[:4]
    if not preferred:
        return thesis
    return f"{thesis} [{', '.join(preferred)}]"
