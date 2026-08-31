from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    live_data_enabled: bool
    offline_fallback_enabled: bool
    tool_runtime: str
    llm_provider: str
    ollama_model: str
    market_provider: str
    news_provider: str
    papers_provider: str
    request_timeout_seconds: float


def load_settings() -> Settings:
    return Settings(
        live_data_enabled=os.getenv("FIN_RESEARCH_LIVE", "1") != "0",
        offline_fallback_enabled=os.getenv("FIN_RESEARCH_OFFLINE_FALLBACK", "1") != "0",
        tool_runtime=os.getenv("FIN_RESEARCH_TOOL_RUNTIME", "local").lower(),
        llm_provider=os.getenv("FIN_RESEARCH_LLM", "off").lower(),
        ollama_model=os.getenv("FIN_RESEARCH_OLLAMA_MODEL", "llama3.2:3b"),
        market_provider=os.getenv("FIN_RESEARCH_MARKET_PROVIDER", "auto").lower(),
        news_provider=os.getenv("FIN_RESEARCH_NEWS_PROVIDER", "auto").lower(),
        papers_provider=os.getenv("FIN_RESEARCH_PAPERS_PROVIDER", "auto").lower(),
        request_timeout_seconds=float(os.getenv("FIN_RESEARCH_TIMEOUT_SECONDS", "8")),
    )
