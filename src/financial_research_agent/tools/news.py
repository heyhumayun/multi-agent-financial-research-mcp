from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from financial_research_agent.config import load_settings
from financial_research_agent.domain import NewsItem
from financial_research_agent.network import trusted_ssl_context

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NEWS_PATH = PACKAGE_ROOT / "data" / "news" / "sample_news.json"


def _load_news() -> list[NewsItem]:
    raw_items = json.loads(NEWS_PATH.read_text(encoding="utf-8"))
    return [
        NewsItem(
            ticker=item["ticker"],
            published=date.fromisoformat(item["published"]),
            title=item["title"],
            source=item["source"],
            sentiment=float(item["sentiment"]),
            url=item.get("url", ""),
        )
        for item in raw_items
    ]


def _search_offline_news(query: str, ticker: str | None, limit: int) -> list[NewsItem]:
    terms = {term.lower() for term in query.split() if len(term) > 2}
    normalized_ticker = ticker.upper() if ticker else None

    scored: list[tuple[float, NewsItem]] = []
    for item in _load_news():
        if normalized_ticker and item.ticker != normalized_ticker:
            continue
        text = f"{item.ticker} {item.title} {item.source}".lower()
        lexical_score = sum(1 for term in terms if term in text)
        sentiment_weight = abs(item.sentiment) * 0.25
        score = lexical_score + sentiment_weight
        if score > 0 or normalized_ticker:
            scored.append((score, item))

    scored.sort(key=lambda pair: (pair[0], pair[1].published), reverse=True)
    return [item for _, item in scored[:limit]]


def _parse_rss_date(value: str) -> date:
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).date()


def _sentiment_from_title(title: str) -> float:
    positive = {"beat", "growth", "surge", "upgrade", "strong", "record", "expands"}
    negative = {"miss", "falls", "risk", "probe", "cuts", "weak", "constraint"}
    tokens = {token.strip(".,:;!?").lower() for token in title.split()}
    score = 0.0
    score += 0.25 * len(tokens & positive)
    score -= 0.25 * len(tokens & negative)
    return max(min(score, 1.0), -1.0)


def _search_yahoo_rss_news(query: str, ticker: str | None, limit: int) -> list[NewsItem]:
    settings = load_settings()
    search_term = ticker.upper() if ticker else query
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(search_term)}&region=US&lang=en-US"
    with urlopen(
        url, timeout=settings.request_timeout_seconds, context=trusted_ssl_context()
    ) as response:
        payload = response.read()

    root = ElementTree.fromstring(payload)
    items: list[NewsItem] = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        published = _parse_rss_date(item.findtext("pubDate", default=""))
        if not title:
            continue
        items.append(
            NewsItem(
                ticker=ticker.upper() if ticker else search_term.upper(),
                published=published,
                title=title,
                source="Yahoo Finance RSS",
                sentiment=_sentiment_from_title(title),
                url=link,
            )
        )
    if not items:
        raise RuntimeError("Yahoo Finance RSS returned no news items")
    return items


def _search_newsapi_news(query: str, ticker: str | None, limit: int) -> list[NewsItem]:
    settings = load_settings()
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY is not set")

    search_term = f"{ticker} {query}" if ticker else query
    params = urlencode(
        {
            "q": search_term,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": api_key,
        }
    )
    with urlopen(
        f"https://newsapi.org/v2/everything?{params}",
        timeout=settings.request_timeout_seconds,
        context=trusted_ssl_context(),
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))

    articles = payload.get("articles", [])
    items: list[NewsItem] = []
    for article in articles[:limit]:
        title = str(article.get("title", "")).strip()
        if not title:
            continue
        published_at = str(article.get("publishedAt", ""))[:10]
        try:
            published = date.fromisoformat(published_at)
        except ValueError:
            published = datetime.now(timezone.utc).date()
        source = article.get("source") or {}
        items.append(
            NewsItem(
                ticker=ticker.upper() if ticker else "",
                published=published,
                title=title,
                source=f"NewsAPI: {source.get('name', 'unknown')}",
                sentiment=_sentiment_from_title(title),
                url=str(article.get("url", "")),
            )
        )
    if not items:
        raise RuntimeError("NewsAPI returned no articles")
    return items


def _search_finnhub_news(ticker: str | None, limit: int) -> list[NewsItem]:
    settings = load_settings()
    api_key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY or FINNHUB_KEY is not set")
    if not ticker:
        raise RuntimeError("Finnhub company news requires a ticker")

    end_date = datetime.now(timezone.utc).date()
    params = urlencode(
        {
            "symbol": ticker.upper(),
            "from": (end_date - timedelta(days=30)).isoformat(),
            "to": end_date.isoformat(),
        }
    )
    request = Request(
        f"https://finnhub.io/api/v1/company-news?{params}",
        headers={"X-Finnhub-Token": api_key.strip()},
    )
    for attempt in range(3):
        try:
            with urlopen(
                request,
                timeout=settings.request_timeout_seconds,
                context=trusted_ssl_context(),
            ) as response:
                articles = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(0.5 * (2**attempt))

    items: list[NewsItem] = []
    for article in articles[:limit]:
        title = str(article.get("headline", "")).strip()
        if not title:
            continue
        timestamp = article.get("datetime")
        try:
            published = datetime.fromtimestamp(float(timestamp), timezone.utc).date()
        except (TypeError, ValueError, OSError):
            published = end_date
        items.append(
            NewsItem(
                ticker=ticker.upper(),
                published=published,
                title=title,
                source=f"Finnhub: {article.get('source', 'unknown')}",
                sentiment=_sentiment_from_title(title),
                url=str(article.get("url", "")),
            )
        )
    if not items:
        raise RuntimeError("Finnhub returned no company news")
    return items


def _select_news_provider(provider: str) -> str:
    settings = load_settings()
    selected_provider = settings.news_provider if provider == "auto" else provider
    if selected_provider == "auto":
        selected_provider = "yahoo_rss" if settings.live_data_enabled else "offline"
    return selected_provider


def search_news_with_metadata(
    query: str, ticker: str | None = None, limit: int = 5, provider: str = "auto"
) -> tuple[list[NewsItem], str, bool]:
    settings = load_settings()
    selected_provider = _select_news_provider(provider)
    if selected_provider == "offline":
        return _search_offline_news(query=query, ticker=ticker, limit=limit), "offline", False

    if selected_provider == "yahoo_rss":
        try:
            return (
                _search_yahoo_rss_news(query=query, ticker=ticker, limit=limit),
                "yahoo_rss",
                False,
            )
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _search_offline_news(query=query, ticker=ticker, limit=limit), "offline", True

    if selected_provider == "newsapi":
        try:
            return _search_newsapi_news(query=query, ticker=ticker, limit=limit), "newsapi", False
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _search_offline_news(query=query, ticker=ticker, limit=limit), "offline", True

    if selected_provider == "finnhub":
        try:
            return _search_finnhub_news(ticker=ticker, limit=limit), "finnhub", False
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _search_offline_news(query=query, ticker=ticker, limit=limit), "offline", True

    if settings.offline_fallback_enabled:
        return _search_offline_news(query=query, ticker=ticker, limit=limit), "offline", True
    raise ValueError(f"Unsupported news provider: {selected_provider}")


def search_news(
    query: str, ticker: str | None = None, limit: int = 5, provider: str = "auto"
) -> list[NewsItem]:
    items, _, _ = search_news_with_metadata(
        query=query, ticker=ticker, limit=limit, provider=provider
    )
    return items
