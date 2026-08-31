import unittest

from financial_research_agent.api import app


class ApiTests(unittest.TestCase):
    def test_app_imports_or_gracefully_missing_dependency(self):
        self.assertTrue(app is None or getattr(app, "title", ""))


if __name__ == "__main__":
    unittest.main()
