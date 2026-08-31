def tool_manifest() -> list[dict[str, str]]:
    return [
        {"name": "get_market_data", "purpose": "Fetch recent OHLC-like market bars."},
        {"name": "get_company_fundamentals", "purpose": "Fetch SEC-backed company facts."},
        {"name": "search_news", "purpose": "Search finance news with sentiment metadata."},
        {"name": "search_arxiv", "purpose": "Search academic/research context."},
        {"name": "search_documents", "purpose": "Search local financial research notes."},
        {
            "name": "search_documents_vector",
            "purpose": "Search local financial research notes with vector-space ranking.",
        },
        {
            "name": "search_documents_semantic",
            "purpose": "Search local notes with FAISS embeddings when available.",
        },
        {"name": "calculate_returns", "purpose": "Calculate log returns from prices."},
        {"name": "calculate_volatility", "purpose": "Calculate rolling volatility."},
        {"name": "calculate_max_drawdown", "purpose": "Calculate maximum drawdown."},
    ]


__all__ = ["tool_manifest"]
