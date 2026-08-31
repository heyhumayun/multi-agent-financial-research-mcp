from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class MarketBar:
    ticker: str
    date: date
    close: float
    volume: int


@dataclass(frozen=True)
class NewsItem:
    ticker: str
    published: date
    title: str
    source: str
    sentiment: float
    url: str = ""


@dataclass(frozen=True)
class PaperItem:
    title: str
    authors: tuple[str, ...]
    summary: str
    url: str
    relevance_score: float


@dataclass(frozen=True)
class DocumentHit:
    path: str
    snippet: str
    score: float


@dataclass(frozen=True)
class FundamentalSnapshot:
    ticker: str
    cik: str
    period: str
    revenue: float | None
    net_income: float | None
    assets: float | None
    liabilities: float | None
    source: str


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    inputs: dict[str, Any]
    output: Any
    evidence: list[str] = field(default_factory=list)
    provider: str = "offline"
    fallback_used: bool = False
    latency_ms: float = 0.0
    provider_latency_ms: float = 0.0
    transport_latency_ms: float = 0.0


@dataclass(frozen=True)
class AgentFinding:
    agent_name: str
    headline: str
    details: list[str]
    confidence: float
    tool_results: list[ToolResult] = field(default_factory=list)
    reasoning: str = ""
    critique: str = ""
    citations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchReport:
    query: str
    thesis: str
    findings: list[AgentFinding]
    risks: list[str]
    next_steps: list[str]
    run_id: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    evidence_registry: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# Financial Research Brief",
            "",
            f"**Query:** {self.query}",
            "",
            f"**Thesis:** {self.thesis}",
            "",
            "## Findings",
        ]
        for finding in self.findings:
            lines.extend(
                [
                    "",
                    f"### {finding.agent_name}: {finding.headline}",
                    f"Confidence: {finding.confidence:.2f}",
                ]
            )
            if finding.reasoning:
                lines.append(f"Reasoning: {finding.reasoning}")
            if finding.critique:
                lines.append(f"Critique: {finding.critique}")
            citation_text = f" [{', '.join(finding.citations)}]" if finding.citations else ""
            lines.extend(f"- {detail}{citation_text}" for detail in finding.details)

        lines.extend(["", "## Data Sources"])
        source_rows = []
        for finding in self.findings:
            for result in finding.tool_results:
                source_rows.append(
                    (
                        result.tool_name,
                        result.provider,
                        "yes" if result.fallback_used else "no",
                        len(result.evidence),
                    )
                )
        if source_rows:
            lines.extend(
                f"- {tool}: provider={provider}, fallback={fallback}, evidence_items={count}"
                for tool, provider, fallback, count in source_rows
            )
        else:
            lines.append("- No tool sources recorded.")

        if self.evidence_registry:
            lines.extend(["", "## Evidence Registry"])
            lines.extend(
                f"- [{evidence_id}] {source}"
                for evidence_id, source in self.evidence_registry.items()
            )

        lines.extend(["", "## Key Risks"])
        lines.extend(f"- {risk}" for risk in self.risks)

        lines.extend(["", "## Next Steps"])
        lines.extend(f"- {step}" for step in self.next_steps)

        if self.evaluation:
            lines.extend(
                [
                    "",
                    "## Quality Checks",
                    f"- Score: {self.evaluation.get('score', 0):.2f}",
                ]
            )
            lines.extend(f"- {item}" for item in self.evaluation.get("checks", []))
            contradictions = self.evaluation.get("contradictions", [])
            if contradictions:
                lines.extend(["", "## Cross-Agent Tensions"])
                lines.extend(f"- {item}" for item in contradictions)

        if self.run_id:
            lines.extend(["", f"Run ID: `{self.run_id}`"])
        return "\n".join(lines)
