from __future__ import annotations

from urllib.parse import quote_plus
from urllib.request import urlopen
from xml.etree import ElementTree

from financial_research_agent.config import load_settings
from financial_research_agent.domain import PaperItem
from financial_research_agent.network import trusted_ssl_context

OFFLINE_PAPERS = [
    PaperItem(
        title="Graph Neural Networks for Financial Market Prediction",
        authors=("Sample Researcher", "Example Quant"),
        summary=(
            "Studies how graph structure can encode relationships between assets, news, "
            "and sectors for return prediction."
        ),
        url="offline://papers/gnn-financial-market-prediction",
        relevance_score=0.91,
    ),
    PaperItem(
        title="Retrieval-Augmented Generation for Financial Question Answering",
        authors=("Sample NLP Lab",),
        summary=(
            "Explores retrieval, source attribution, and hallucination control for finance "
            "domain question answering."
        ),
        url="offline://papers/rag-financial-qa",
        relevance_score=0.88,
    ),
    PaperItem(
        title="Volatility Forecasting with News and Price Features",
        authors=("Example ML Group",),
        summary=(
            "Combines textual sentiment and log-return features for volatility and downside "
            "risk estimation."
        ),
        url="offline://papers/news-volatility",
        relevance_score=0.84,
    ),
]


def _search_offline_arxiv(query: str, limit: int) -> list[PaperItem]:
    terms = {term.lower() for term in query.split() if len(term) > 2}
    scored: list[tuple[float, PaperItem]] = []
    for paper in OFFLINE_PAPERS:
        text = f"{paper.title} {paper.summary}".lower()
        lexical_score = sum(1 for term in terms if term in text)
        score = lexical_score + paper.relevance_score
        scored.append((score, paper))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [paper for _, paper in scored[:limit]]


def _search_live_arxiv(query: str, limit: int) -> list[PaperItem]:
    settings = load_settings()
    encoded_query = quote_plus(query)
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded_query}&start=0&max_results={limit}"
        "&sortBy=relevance&sortOrder=descending"
    )
    with urlopen(
        url, timeout=settings.request_timeout_seconds, context=trusted_ssl_context()
    ) as response:
        payload = response.read()

    root = ElementTree.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[PaperItem] = []
    for entry in root.findall("atom:entry", namespace):
        title = " ".join(entry.findtext("atom:title", default="", namespaces=namespace).split())
        summary = " ".join(entry.findtext("atom:summary", default="", namespaces=namespace).split())
        authors = tuple(
            author.findtext("atom:name", default="", namespaces=namespace)
            for author in entry.findall("atom:author", namespace)
        )
        url_text = entry.findtext("atom:id", default="", namespaces=namespace)
        if title:
            papers.append(
                PaperItem(
                    title=title,
                    authors=authors,
                    summary=summary[:500],
                    url=url_text,
                    relevance_score=0.9,
                )
            )

    if not papers:
        raise RuntimeError("arXiv returned no papers")
    return papers


def _select_papers_provider(provider: str) -> str:
    settings = load_settings()
    selected_provider = settings.papers_provider if provider == "auto" else provider
    if selected_provider == "auto":
        selected_provider = "arxiv" if settings.live_data_enabled else "offline"
    return selected_provider


def search_arxiv_with_metadata(
    query: str, limit: int = 5, provider: str = "auto"
) -> tuple[list[PaperItem], str, bool]:
    settings = load_settings()
    selected_provider = _select_papers_provider(provider)
    if selected_provider == "offline":
        return _search_offline_arxiv(query=query, limit=limit), "offline", False

    if selected_provider == "arxiv":
        try:
            return _search_live_arxiv(query=query, limit=limit), "arxiv", False
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _search_offline_arxiv(query=query, limit=limit), "offline", True

    raise ValueError(f"Unsupported papers provider: {selected_provider}")


def search_arxiv(query: str, limit: int = 5, provider: str = "auto") -> list[PaperItem]:
    papers, _, _ = search_arxiv_with_metadata(query=query, limit=limit, provider=provider)
    return papers
