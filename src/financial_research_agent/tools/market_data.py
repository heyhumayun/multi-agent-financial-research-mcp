from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

from financial_research_agent.config import load_settings
from financial_research_agent.domain import MarketBar
from financial_research_agent.network import trusted_ssl_context


def _synthetic_close(seed: int, day: int) -> float:
    trend = 0.0015 * day
    cycle = 0.035 * math.sin(day / 4.0 + seed)
    shock = 0.012 * math.sin(day / 2.3)
    return 100.0 * math.exp(trend + cycle + shock)


def _get_offline_market_data(ticker: str, days: int) -> list[MarketBar]:
    normalized = ticker.upper().strip()
    seed = sum(ord(ch) for ch in normalized) % 17
    end = date(2026, 8, 21)
    trading_days: list[date] = []

    current = end
    while len(trading_days) < days:
        if current.weekday() < 5:
            trading_days.append(current)
        current -= timedelta(days=1)

    bars: list[MarketBar] = []
    for idx, trading_day in enumerate(reversed(trading_days)):
        close = round(_synthetic_close(seed, idx), 2)
        volume = 1_000_000 + (seed * 137_000) + (idx % 11) * 53_000
        bars.append(MarketBar(ticker=normalized, date=trading_day, close=close, volume=volume))
    return bars


def _get_yfinance_market_data(ticker: str, days: int) -> list[MarketBar]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Install the market extra.") from exc

    period = f"{max(days * 2, 30)}d"
    dataframe = yf.download(ticker.upper(), period=period, progress=False, auto_adjust=False)
    if dataframe.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    if hasattr(dataframe.columns, "nlevels") and dataframe.columns.nlevels > 1:
        dataframe.columns = [column[0] for column in dataframe.columns]

    dataframe = dataframe.tail(days)
    bars: list[MarketBar] = []
    for index, row in dataframe.iterrows():
        close = float(row["Close"])
        if not math.isfinite(close) or close <= 0:
            continue
        raw_volume = float(row.get("Volume", 0))
        volume = int(raw_volume) if math.isfinite(raw_volume) else 0
        bars.append(
            MarketBar(
                ticker=ticker.upper(),
                date=index.date(),
                close=close,
                volume=volume,
            )
        )
    if not bars:
        raise RuntimeError(f"yfinance returned no valid price rows for {ticker}")
    return bars


def _read_json_url(url: str) -> dict:
    settings = load_settings()
    with urlopen(
        url, timeout=settings.request_timeout_seconds, context=trusted_ssl_context()
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_alpha_vantage_market_data(ticker: str, days: int) -> list[MarketBar]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set")

    query = urlencode(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker.upper(),
            "outputsize": "compact",
            "apikey": api_key,
        }
    )
    payload = _read_json_url(f"https://www.alphavantage.co/query?{query}")
    series = payload.get("Time Series (Daily)", {})
    if not series:
        raise RuntimeError("Alpha Vantage returned no daily time series")

    bars = [
        MarketBar(
            ticker=ticker.upper(),
            date=date.fromisoformat(day),
            close=float(values["4. close"]),
            volume=int(float(values.get("5. volume", 0))),
        )
        for day, values in series.items()
    ]
    bars.sort(key=lambda bar: bar.date)
    return bars[-days:]


def _get_polygon_market_data(ticker: str, days: int) -> list[MarketBar]:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is not set")

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(days * 3, 30))
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
        f"{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    )
    payload = _read_json_url(url)
    rows = payload.get("results", [])
    if not rows:
        raise RuntimeError("Polygon returned no aggregate bars")

    bars = [
        MarketBar(
            ticker=ticker.upper(),
            date=datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc).date(),
            close=float(row["c"]),
            volume=int(float(row.get("v", 0))),
        )
        for row in rows
    ]
    return bars[-days:]


def _get_fmp_market_data(ticker: str, days: int) -> list[MarketBar]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set")

    query = urlencode({"timeseries": days, "apikey": api_key})
    payload = _read_json_url(
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker.upper()}?{query}"
    )
    rows = payload.get("historical", [])
    if not rows:
        raise RuntimeError("Financial Modeling Prep returned no historical prices")

    bars = [
        MarketBar(
            ticker=ticker.upper(),
            date=date.fromisoformat(row["date"]),
            close=float(row["close"]),
            volume=int(float(row.get("volume", 0))),
        )
        for row in rows
    ]
    bars.sort(key=lambda bar: bar.date)
    return bars[-days:]


def _select_market_provider(provider: str) -> str:
    settings = load_settings()
    selected_provider = settings.market_provider if provider == "auto" else provider
    if selected_provider == "auto":
        selected_provider = "yfinance" if settings.live_data_enabled else "offline"
    return selected_provider


def get_market_data_with_metadata(
    ticker: str, days: int = 60, provider: str = "auto"
) -> tuple[list[MarketBar], str, bool]:
    settings = load_settings()
    selected_provider = _select_market_provider(provider)
    if selected_provider == "offline":
        return _get_offline_market_data(ticker, days), "offline", False

    if selected_provider == "yfinance":
        try:
            return _get_yfinance_market_data(ticker, days), "yfinance", False
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _get_offline_market_data(ticker, days), "offline", True

    provider_functions = {
        "alpha_vantage": _get_alpha_vantage_market_data,
        "polygon": _get_polygon_market_data,
        "fmp": _get_fmp_market_data,
    }
    if selected_provider in provider_functions:
        try:
            return provider_functions[selected_provider](ticker, days), selected_provider, False
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _get_offline_market_data(ticker, days), "offline", True

    raise ValueError(f"Unsupported market data provider: {selected_provider}")


def get_market_data(ticker: str, days: int = 60, provider: str = "auto") -> list[MarketBar]:
    """Return recent market bars.

    Provider behavior:
    - offline: deterministic synthetic data for tests and demos.
    - yfinance: live Yahoo Finance data, requires optional dependency/network.
    - alpha_vantage: free-key daily bars through ALPHA_VANTAGE_API_KEY.
    - polygon: free-key historical aggregates through POLYGON_API_KEY.
    - fmp: free-key historical prices through FMP_API_KEY.
    - auto: live only when FIN_RESEARCH_LIVE=1, otherwise offline.
    """
    bars, _, _ = get_market_data_with_metadata(ticker=ticker, days=days, provider=provider)
    return bars


def calculate_returns(prices: list[float]) -> list[float]:
    if len(prices) < 2:
        return []
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def calculate_volatility(prices: list[float], window: int = 20, annualize: bool = True) -> float:
    returns = calculate_returns(prices[-window - 1 :])
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((ret - mean) ** 2 for ret in returns) / (len(returns) - 1)
    volatility = math.sqrt(variance)
    return volatility * math.sqrt(252) if annualize else volatility


def calculate_max_drawdown(prices: list[float]) -> float:
    if not prices:
        return 0.0
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        drawdown = (price / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown
