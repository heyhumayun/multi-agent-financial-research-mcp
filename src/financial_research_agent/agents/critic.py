from __future__ import annotations

from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.evaluation import detect_contradictions
from financial_research_agent.reasoning import ReasoningEngine


class CriticAgent:
    name = "Critic Agent"
    reasoning_engine = ReasoningEngine()

    def review(self, query: str, findings: list[AgentFinding]) -> AgentFinding:
        weak_points: list[str] = []
        lowered = query.lower()
        if any(result.fallback_used for finding in findings for result in finding.tool_results):
            weak_points.append(
                "Some live sources fell back to offline data, so provider freshness matters."
            )
        if not any("drawdown" in " ".join(finding.details).lower() for finding in findings):
            weak_points.append("Drawdown risk was not explicitly quantified.")
        if not any("news" in finding.agent_name.lower() for finding in findings):
            weak_points.append("News flow was not included.")
        if not any("paper" in finding.agent_name.lower() for finding in findings):
            weak_points.append(
                "Academic context was not included; this is acceptable only for market-only queries."
            )
        tensions = detect_contradictions(findings)
        weak_points.extend(f"Cross-agent contradiction: {item}" for item in tensions)
        if "option" in lowered or "implied volatility" in lowered:
            weak_points.append("Options analysis is unsupported because no options-chain tool ran.")
        if any(term in lowered for term in ["forex", "eurusd", "currency", "carry"]):
            weak_points.append("Forex analysis is unsupported by the current equity tool set.")
        if any(term in lowered for term in ["commodity", "crude oil", "futures curve"]):
            weak_points.append("Commodity futures analysis is unsupported by the current tools.")

        if not weak_points:
            weak_points = [
                "The report is evidence-linked, but conclusions should still be treated as research hypotheses."
            ]

        evidence_summary = "; ".join(
            f"{finding.agent_name}: {finding.headline}" for finding in findings
        )
        return AgentFinding(
            agent_name=self.name,
            headline="Cross-agent review flags evidence and decision-quality limits",
            details=weak_points,
            confidence=0.74,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "agent findings were checked for fallback usage, missing risk metrics, and missing source categories.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "critic checks are rubric-based and do not prove claim-level source entailment; cited excerpts still need review.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="review_agent_findings",
                    inputs={"query": query},
                    output={"reviewed_findings": evidence_summary},
                    evidence=[finding.agent_name for finding in findings],
                    provider="internal-critic",
                )
            ],
        )
