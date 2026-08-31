from __future__ import annotations

from financial_research_agent.agents.base import Agent
from financial_research_agent.domain import AgentFinding, ToolResult
from financial_research_agent.reasoning import ReasoningEngine
from financial_research_agent.tool_gateway import (
    ToolGateway,
    expect_document_hits,
    expect_paper_items,
)


class ResearchPapersAgent(Agent):
    name = "Research/Papers Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        papers_result = tools.search_arxiv(
            f"{query} financial machine learning volatility graph neural networks", limit=3
        )
        papers = expect_paper_items(papers_result)
        details = [
            f"{paper.title}: {paper.summary} Relevance={paper.relevance_score:.2f}"
            for paper in papers
        ]
        return AgentFinding(
            agent_name=self.name,
            headline="Academic context supports graph, retrieval, and volatility framing",
            details=details,
            confidence=0.69,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "papers were ranked for overlap with financial ML, volatility, graph, and RAG themes.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "paper relevance does not prove alpha; implementation and validation still matter.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="search_arxiv",
                    inputs={"query": query, "limit": 3},
                    output=papers,
                    evidence=[paper.url for paper in papers],
                    provider=papers_result.provider,
                    fallback_used=papers_result.fallback_used,
                    latency_ms=papers_result.latency_ms,
                    provider_latency_ms=papers_result.provider_latency_ms,
                    transport_latency_ms=papers_result.transport_latency_ms,
                )
            ],
        )


class DocumentAgent(Agent):
    name = "Document Agent"
    reasoning_engine = ReasoningEngine()

    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        document_result = tools.search_documents_semantic(query, limit=3)
        hits = expect_document_hits(document_result)
        if not hits:
            details = ["No matching local research notes were found."]
            confidence = 0.35
        else:
            details = [f"{hit.path}: {hit.snippet} Score={hit.score:.2f}" for hit in hits]
            confidence = 0.67

        return AgentFinding(
            agent_name=self.name,
            headline="Local knowledge base retrieved relevant research notes",
            details=details,
            confidence=confidence,
            reasoning=self.reasoning_engine.reason(
                self.name,
                "local notes were searched with semantic FAISS embeddings when available, otherwise vector fallback.",
            ),
            critique=self.reasoning_engine.critique(
                self.name,
                "semantic retrieval improves recall, but retrieved notes still need source-quality review.",
            ),
            tool_results=[
                ToolResult(
                    tool_name="search_documents_semantic",
                    inputs={"query": query, "limit": 3},
                    output=hits,
                    evidence=[hit.path for hit in hits],
                    provider=document_result.provider,
                    latency_ms=document_result.latency_ms,
                    provider_latency_ms=document_result.provider_latency_ms,
                    transport_latency_ms=document_result.transport_latency_ms,
                )
            ],
        )
