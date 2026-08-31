import unittest
from unittest.mock import patch

from financial_research_agent.agents import SupervisorAgent


class SupervisorTests(unittest.TestCase):
    def test_supervisor_generates_report(self):
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            report = SupervisorAgent().run("Assess NVDA for an AI infrastructure trade with risk")
        rendered = report.to_markdown()
        self.assertIn("Financial Research Brief", rendered)
        self.assertIn("Market Data Agent", rendered)
        self.assertIn("News Agent", rendered)
        self.assertIn("Risk Analysis Agent", rendered)
        self.assertIn("Reasoning:", rendered)
        self.assertIn("Critique:", rendered)
        self.assertIn("Data Sources", rendered)
        self.assertIn("Evidence Registry", rendered)
        self.assertTrue(report.evidence_registry)
        self.assertTrue(report.evaluation["grounding"]["thesis_citation_count"])
        self.assertFalse(report.evaluation["grounding"]["unresolved_citations"])
        self.assertIn("Every agent included reasoning", rendered)

    def test_supervisor_routes_research_and_documents(self):
        supervisor = SupervisorAgent()
        plan = supervisor.plan("NVDA research paper and infrastructure document")
        self.assertIn("research", plan)
        self.assertIn("document", plan)

    def test_supervisor_accepts_explicit_unlisted_ticker(self):
        self.assertEqual(SupervisorAgent().infer_ticker("Assess PLTR earnings risk"), "PLTR")

    def test_supervisor_detects_multiple_tickers_and_comparison(self):
        supervisor = SupervisorAgent()
        self.assertEqual(
            supervisor.infer_tickers("Compare NVDA and AMD volatility"), ["AMD", "NVDA"]
        )
        self.assertIn("comparison", supervisor.plan("Compare NVDA and AMD volatility"))
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            report = supervisor.run("Compare NVDA and AMD volatility")
        comparison = next(
            finding for finding in report.findings if finding.agent_name == "Comparison Agent"
        )
        self.assertIn("volatility", " ".join(comparison.details).lower())

    def test_supervisor_rejects_llm_comparison_for_one_ticker(self):
        supervisor = SupervisorAgent()
        with patch.object(
            supervisor.reasoning_engine,
            "plan_agents",
            return_value=["market", "news", "risk", "fundamentals", "comparison"],
        ):
            plan = supervisor.plan("Assess NVDA company fundamentals and earnings risk")
        self.assertIn("fundamentals", plan)
        self.assertNotIn("comparison", plan)

    def test_supervisor_rejects_empty_query(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            SupervisorAgent().run("   ")

    def test_quality_score_penalizes_stale_offline_fixture(self):
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            report = SupervisorAgent().run("Assess NVDA risk")
        self.assertLess(report.evaluation["score"], 1.0)
        self.assertIn("get_market_data", report.evaluation["freshness"]["stale_tools"])

    def test_trace_records_autonomous_plan_and_bounded_stop(self):
        with patch.dict("os.environ", {"FIN_RESEARCH_LIVE": "0"}):
            _, trace = SupervisorAgent().run_with_trace("Assess NVDA AI infrastructure risk")
        summary = trace.to_summary()
        self.assertIn("market", summary["planned_agents"])
        self.assertIn("news", summary["planned_agents"])
        self.assertGreaterEqual(summary["iterations"], len(summary["planned_agents"]))
        self.assertEqual(summary["stop_reason"], "bounded_review_complete")
        self.assertTrue(summary["decisions"])
        self.assertGreater(summary["end_to_end_latency_ms"], 0)
        self.assertEqual(summary["total_latency_ms"], summary["end_to_end_latency_ms"])
        self.assertIn("provider_latency_ms", summary)
        self.assertIn("transport_latency_ms", summary)


if __name__ == "__main__":
    unittest.main()
