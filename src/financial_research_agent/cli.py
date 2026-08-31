from __future__ import annotations

import argparse
import json

from financial_research_agent.agents import SupervisorAgent
from financial_research_agent.persistence import save_run_artifacts
from financial_research_agent.serialization import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-agent financial research workflow.")
    parser.add_argument("query", help="Research query, ideally including a ticker such as NVDA.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of Markdown.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Include run trace when emitting JSON.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report.md, report.json, and trace.json under runs/<run_id>/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, trace = SupervisorAgent().run_with_trace(args.query)
    run_dir = save_run_artifacts(report, trace) if args.save else None
    if args.json:
        payload = {"report": to_jsonable(report)}
        if args.trace:
            payload["trace"] = trace.to_summary()
        if run_dir:
            payload["saved_to"] = str(run_dir)
        print(json.dumps(payload, indent=2))
    else:
        print(report.to_markdown())
        if run_dir:
            print(f"\nSaved artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
