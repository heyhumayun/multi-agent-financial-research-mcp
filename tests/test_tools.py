import json
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from financial_research_agent.tools import (
    calculate_max_drawdown,
    calculate_returns,
    calculate_volatility,
    get_market_data,
    search_arxiv,
    search_documents,
    search_documents_semantic,
    search_documents_vector,
    search_news,
)
from financial_research_agent.tools.fundamentals import (
    _get_sec_fundamentals,
    get_company_fundamentals_with_metadata,
)


class ToolTests(unittest.TestCase):
    def test_sec_fundamentals_uses_newest_supported_revenue_concept(self):
        payload = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "fy": 2022,
                                    "accn": "old",
                                    "filed": "2023-01-01",
                                    "end": "2022-12-31",
                                    "val": 100,
                                }
                            ]
                        }
                    },
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "fy": 2025,
                                    "accn": "new",
                                    "filed": "2026-01-01",
                                    "end": "2025-12-31",
                                    "val": 200,
                                }
                            ]
                        }
                    },
                }
            }
        }
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.__enter__.return_value = response
        with patch("financial_research_agent.tools.fundamentals.urlopen", return_value=response):
            snapshot = _get_sec_fundamentals("NVDA")
        self.assertEqual(snapshot.period, "2025-12-31")
        self.assertEqual(snapshot.revenue, 200)

    def test_sec_fundamentals_align_facts_to_one_annual_filing(self):
        payload = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "fy": 2024,
                                    "accn": "old",
                                    "filed": "2025-01-01",
                                    "end": "2024-12-31",
                                    "val": 100,
                                },
                                {
                                    "form": "10-K",
                                    "fy": 2025,
                                    "accn": "new",
                                    "filed": "2026-01-01",
                                    "end": "2025-12-31",
                                    "val": 200,
                                },
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "fy": 2024,
                                    "accn": "old",
                                    "filed": "2025-01-01",
                                    "end": "2024-12-31",
                                    "val": 20,
                                },
                                {
                                    "form": "10-K",
                                    "fy": 2025,
                                    "accn": "new",
                                    "filed": "2026-01-01",
                                    "end": "2025-12-31",
                                    "val": 40,
                                },
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "fy": 2025,
                                    "accn": "new",
                                    "filed": "2026-01-01",
                                    "end": "2025-12-31",
                                    "val": 500,
                                }
                            ]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "fy": 2025,
                                    "accn": "new",
                                    "filed": "2026-01-01",
                                    "end": "2025-12-31",
                                    "val": 200,
                                }
                            ]
                        }
                    },
                }
            }
        }
        with patch("financial_research_agent.tools.fundamentals.urlopen") as open_url:
            response = MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps(payload).encode()
            open_url.return_value = response
            with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "1"}):
                snapshot = _get_sec_fundamentals("NVDA")
        self.assertEqual(snapshot.period, "2025-12-31")
        self.assertEqual(snapshot.revenue, 200.0)
        self.assertEqual(snapshot.net_income, 40.0)

    def test_yfinance_adapter_discards_non_finite_prices(self):
        import pandas as pd

        frame = pd.DataFrame(
            {"Close": [100.0, float("nan"), 102.0], "Volume": [10, 20, float("nan")]},
            index=pd.to_datetime(["2026-08-19", "2026-08-20", "2026-08-21"]),
        )
        fake_yfinance = MagicMock()
        fake_yfinance.download.return_value = frame
        with patch.dict("sys.modules", {"yfinance": fake_yfinance}):
            from financial_research_agent.tools.market_data import _get_yfinance_market_data

            bars = _get_yfinance_market_data("NVDA", days=3)
        self.assertEqual([bar.close for bar in bars], [100.0, 102.0])
        self.assertEqual(bars[-1].volume, 0)

    def test_offline_fundamentals_have_provenance(self):
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            snapshot, provider, fallback = get_company_fundamentals_with_metadata("NVDA")
        self.assertEqual(provider, "offline")
        self.assertFalse(fallback)
        self.assertEqual(snapshot.ticker, "NVDA")
        self.assertIsNotNone(snapshot.revenue)

    def test_market_data_is_deterministic(self):
        first = get_market_data("NVDA", days=10, provider="offline")
        second = get_market_data("NVDA", days=10, provider="offline")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(first[-1].ticker, "NVDA")
        self.assertEqual(first[-1].date, date(2026, 8, 21))

    def test_risk_metrics_are_well_formed(self):
        prices = [100, 102, 101, 105, 99, 103]
        returns = calculate_returns(prices)
        self.assertEqual(len(returns), 5)
        self.assertGreater(calculate_volatility(prices, window=5), 0)
        self.assertLess(calculate_max_drawdown(prices), 0)

    def test_search_tools_return_evidence(self):
        self.assertTrue(search_news("NVDA AI infrastructure", ticker="NVDA", provider="offline"))
        self.assertTrue(search_arxiv("financial machine learning graph", provider="offline"))
        self.assertTrue(search_documents("AI infrastructure volatility risk"))
        self.assertTrue(search_documents_vector("AI infrastructure volatility risk"))
        self.assertTrue(search_documents_semantic("AI infrastructure volatility risk"))

    def test_live_provider_falls_back_offline(self):
        market = get_market_data("NVDA", days=5, provider="yfinance")
        news = search_news("NVDA AI infrastructure", ticker="NVDA", provider="yahoo_rss")
        papers = search_arxiv("financial machine learning graph volatility", provider="arxiv")
        self.assertEqual(len(market), 5)
        self.assertTrue(news)
        self.assertTrue(papers)

    def test_keyed_free_providers_fall_back_without_keys(self):
        with patch.dict(
            "os.environ",
            {
                "ALPHA_VANTAGE_API_KEY": "",
                "POLYGON_API_KEY": "",
                "FMP_API_KEY": "",
                "NEWSAPI_KEY": "",
                "FINNHUB_API_KEY": "",
                "FINNHUB_KEY": "",
            },
        ):
            self.assertEqual(len(get_market_data("NVDA", days=5, provider="alpha_vantage")), 5)
            self.assertEqual(len(get_market_data("NVDA", days=5, provider="polygon")), 5)
            self.assertEqual(len(get_market_data("NVDA", days=5, provider="fmp")), 5)
            self.assertTrue(
                search_news("NVDA AI infrastructure", ticker="NVDA", provider="newsapi")
            )
            self.assertTrue(
                search_news("NVDA AI infrastructure", ticker="NVDA", provider="finnhub")
            )

    def test_unsupported_news_provider_falls_back_offline(self):
        self.assertTrue(
            search_news("NVDA AI infrastructure", ticker="NVDA", provider="unsupported")
        )


if __name__ == "__main__":
    unittest.main()
