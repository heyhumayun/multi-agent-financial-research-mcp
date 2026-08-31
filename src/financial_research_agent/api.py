from __future__ import annotations

from financial_research_agent.agents import SupervisorAgent
from financial_research_agent.mcp import tool_manifest
from financial_research_agent.serialization import to_jsonable

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - import path for machines without api extra
    FastAPI = None
    BaseModel = object


if FastAPI is not None:

    class ResearchRequest(BaseModel):
        query: str = Field(min_length=3, max_length=2000)

    app = FastAPI(
        title="Multi-Agent Financial Research API",
        version="0.3.0",
        description="Supervisor-driven financial research API with MCP-compatible tools.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/tools")
    def tools() -> list[dict]:
        return tool_manifest()

    @app.post("/research")
    def research(request: ResearchRequest) -> dict:
        report, trace = SupervisorAgent().run_with_trace(request.query)
        return {
            "report": to_jsonable(report),
            "trace": trace.to_summary(),
        }

else:
    app = None


def main() -> None:
    if app is None:
        raise RuntimeError("FastAPI is not installed. Install the api extra to run this service.")
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
