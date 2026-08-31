from __future__ import annotations

import json
from urllib.request import Request, urlopen

from financial_research_agent.config import load_settings
from financial_research_agent.domain import FundamentalSnapshot
from financial_research_agent.network import trusted_ssl_context

KNOWN_CIKS = {
    "AAPL": "0000320193",
    "AMD": "0000002488",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "TSLA": "0001318605",
}

OFFLINE_FACTS = {
    "AAPL": ("2025-FY", 416e9, 112e9, 352e9, 290e9),
    "AMD": ("2025-FY", 28e9, 1.8e9, 16e9, 7e9),
    "GOOGL": ("2025-FY", 350e9, 100e9, 450e9, 130e9),
    "META": ("2025-FY", 190e9, 65e9, 300e9, 90e9),
    "MSFT": ("2025-FY", 280e9, 100e9, 620e9, 380e9),
    "NVDA": ("2025-FY", 130e9, 75e9, 140e9, 50e9),
    "TSLA": ("2025-FY", 100e9, 8e9, 130e9, 45e9),
}


def _snapshot(ticker: str, values: tuple) -> FundamentalSnapshot:
    normalized = ticker.upper()
    return FundamentalSnapshot(
        ticker=normalized,
        cik=KNOWN_CIKS.get(normalized, "unknown"),
        period=values[0],
        revenue=values[1],
        net_income=values[2],
        assets=values[3],
        liabilities=values[4],
        source="offline://fundamentals",
    )


def _offline_fundamentals(ticker: str) -> FundamentalSnapshot:
    return _snapshot(
        ticker, OFFLINE_FACTS.get(ticker.upper(), ("offline-estimate", None, None, None, None))
    )


def _fact_rows(payload: dict, names: tuple[str, ...]) -> list[dict]:
    matching_rows: list[dict] = []
    for name in names:
        rows = (
            payload.get("facts", {})
            .get("us-gaap", {})
            .get(name, {})
            .get("units", {})
            .get("USD", [])
        )
        matching_rows.extend(row for row in rows if row.get("form") in {"10-K", "10-Q"})
    return matching_rows


def _annual_fact(rows: list[dict], target: dict | None = None) -> dict | None:
    annual = [row for row in rows if row.get("form") == "10-K"]
    if target is not None:
        same_filing = [
            row
            for row in annual
            if row.get("accn") == target.get("accn") and row.get("fy") == target.get("fy")
        ]
        if same_filing:
            annual = same_filing
        else:
            same_year = [row for row in annual if row.get("fy") == target.get("fy")]
            if same_year:
                annual = same_year
    if not annual:
        return None
    return max(annual, key=lambda item: (item.get("filed", ""), item.get("end", "")))


def _get_sec_fundamentals(ticker: str) -> FundamentalSnapshot:
    normalized = ticker.upper()
    cik = KNOWN_CIKS.get(normalized)
    if not cik:
        raise RuntimeError(f"SEC CIK is not configured for {normalized}")
    request = Request(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers={"User-Agent": "financial-research-agent/0.2 analyst@example.com"},
    )
    with urlopen(
        request,
        timeout=load_settings().request_timeout_seconds,
        context=trusted_ssl_context(),
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    revenue_rows = _fact_rows(
        payload, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues")
    )
    revenue_row = _annual_fact(revenue_rows)
    if revenue_row is None:
        raise RuntimeError("SEC returned no annual revenue fact")
    net_income_row = _annual_fact(_fact_rows(payload, ("NetIncomeLoss",)), revenue_row)
    assets_row = _annual_fact(_fact_rows(payload, ("Assets",)), revenue_row)
    liabilities_row = _annual_fact(_fact_rows(payload, ("Liabilities",)), revenue_row)
    revenue = float(revenue_row["val"])
    net_income = float(net_income_row["val"]) if net_income_row else None
    assets = float(assets_row["val"]) if assets_row else None
    liabilities = float(liabilities_row["val"]) if liabilities_row else None
    period = str(revenue_row.get("end") or revenue_row.get("fy") or "unknown")
    if net_income is not None and revenue > 0 and abs(net_income) > revenue * 2:
        raise RuntimeError("SEC annual facts failed a revenue/net-income sanity check")
    if all(value is None for value in (revenue, net_income, assets, liabilities)):
        raise RuntimeError("SEC returned no supported fundamental facts")
    return FundamentalSnapshot(
        normalized,
        cik,
        period,
        revenue,
        net_income,
        assets,
        liabilities,
        "https://data.sec.gov/api/xbrl/companyfacts/",
    )


def get_company_fundamentals_with_metadata(
    ticker: str, provider: str = "auto"
) -> tuple[FundamentalSnapshot, str, bool]:
    settings = load_settings()
    selected = (
        provider if provider != "auto" else ("sec" if settings.live_data_enabled else "offline")
    )
    if selected == "offline":
        return _offline_fundamentals(ticker), "offline", False
    if selected == "sec":
        try:
            return _get_sec_fundamentals(ticker), "sec", False
        except Exception:
            if not settings.offline_fallback_enabled:
                raise
            return _offline_fundamentals(ticker), "offline", True
    raise ValueError(f"Unsupported fundamentals provider: {selected}")
