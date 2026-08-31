from __future__ import annotations

import json
from pathlib import Path

from financial_research_agent.domain import ResearchReport
from financial_research_agent.observability import RunTrace
from financial_research_agent.serialization import to_jsonable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"


def save_run_artifacts(report: ResearchReport, trace: RunTrace) -> Path:
    run_id = report.run_id or trace.run_id
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    (run_dir / "report.json").write_text(
        json.dumps(to_jsonable(report), indent=2),
        encoding="utf-8",
    )
    (run_dir / "trace.json").write_text(
        json.dumps(to_jsonable(trace.to_summary()), indent=2),
        encoding="utf-8",
    )
    return run_dir
