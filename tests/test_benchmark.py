import json
import unittest
from pathlib import Path
from unittest.mock import patch

from financial_research_agent.benchmark import DEFAULT_BENCHMARK_PATH, evaluate_case


class BenchmarkTests(unittest.TestCase):
    def test_labelled_dataset_has_thirty_diverse_cases(self):
        cases = json.loads(Path(DEFAULT_BENCHMARK_PATH).read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 30)
        categories = {case["category"] for case in cases}
        self.assertGreaterEqual(len(categories), 10)

    def test_evaluate_case_scores_expected_route(self):
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            result = evaluate_case(
                {
                    "id": "smoke",
                    "query": "Assess NVDA AI infrastructure risk and research",
                    "expected_agents": [
                        "market",
                        "news",
                        "risk",
                        "research",
                        "document",
                    ],
                    "expected_tools": [
                        "get_market_data",
                        "search_news",
                        "calculate_volatility",
                        "calculate_max_drawdown",
                        "search_arxiv",
                        "search_documents_semantic",
                    ],
                    "expected_evidence_categories": [
                        "market",
                        "news",
                        "risk",
                        "research",
                        "document",
                    ],
                }
            )
        self.assertGreaterEqual(result["score"], 0.8)
        self.assertEqual(result["routing_recall"], 1.0)
        self.assertEqual(result["citation_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
