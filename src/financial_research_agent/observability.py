from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ToolTrace:
    agent_name: str
    tool_name: str
    provider: str
    fallback_used: bool
    latency_ms: float
    evidence_count: int
    provider_latency_ms: float = 0.0
    transport_latency_ms: float = 0.0


@dataclass
class StageTrace:
    stage: str
    latency_ms: float


@dataclass
class OllamaTrace:
    operation: str
    latency_ms: float
    succeeded: bool


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_runtime: str = "local"
    selected_agents: list[str] = field(default_factory=list)
    tool_traces: list[ToolTrace] = field(default_factory=list)
    planned_agents: list[str] = field(default_factory=list)
    iterations: int = 0
    decisions: list[str] = field(default_factory=list)
    stop_reason: str = ""
    stage_traces: list[StageTrace] = field(default_factory=list)
    ollama_traces: list[OllamaTrace] = field(default_factory=list)
    protocol_failures: list[str] = field(default_factory=list)
    end_to_end_latency_ms: float = 0.0

    def record_plan(self, agents: list[str]) -> None:
        self.planned_agents = list(agents)

    def record_iteration(self) -> None:
        self.iterations += 1

    def record_decision(self, decision: str) -> None:
        self.decisions.append(decision)

    def record_agent(self, agent_name: str) -> None:
        self.selected_agents.append(agent_name)

    def record_tool(
        self,
        agent_name: str,
        tool_name: str,
        provider: str,
        fallback_used: bool,
        latency_ms: float,
        evidence_count: int,
        provider_latency_ms: float = 0.0,
        transport_latency_ms: float = 0.0,
    ) -> None:
        self.tool_traces.append(
            ToolTrace(
                agent_name=agent_name,
                tool_name=tool_name,
                provider=provider,
                fallback_used=fallback_used,
                latency_ms=latency_ms,
                evidence_count=evidence_count,
                provider_latency_ms=provider_latency_ms,
                transport_latency_ms=transport_latency_ms,
            )
        )

    def record_stage(self, stage: str, latency_ms: float) -> None:
        self.stage_traces.append(StageTrace(stage=stage, latency_ms=round(latency_ms, 2)))

    def record_ollama(self, operation: str, latency_ms: float, succeeded: bool) -> None:
        self.ollama_traces.append(
            OllamaTrace(operation=operation, latency_ms=latency_ms, succeeded=succeeded)
        )

    def record_protocol_failure(self, message: str) -> None:
        self.protocol_failures.append(message)

    def to_summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "tool_runtime": self.tool_runtime,
            "selected_agents": self.selected_agents,
            "planned_agents": self.planned_agents,
            "iterations": self.iterations,
            "decisions": self.decisions,
            "stop_reason": self.stop_reason,
            "tool_calls": [trace.__dict__ for trace in self.tool_traces],
            "stage_latencies": [trace.__dict__ for trace in self.stage_traces],
            "ollama_calls": [trace.__dict__ for trace in self.ollama_traces],
            "fallback_count": sum(1 for trace in self.tool_traces if trace.fallback_used),
            "provider_latency_ms": round(
                sum(trace.provider_latency_ms for trace in self.tool_traces), 2
            ),
            "transport_latency_ms": round(
                sum(trace.transport_latency_ms for trace in self.tool_traces), 2
            ),
            "tool_round_trip_latency_ms": round(
                sum(trace.latency_ms for trace in self.tool_traces), 2
            ),
            "ollama_latency_ms": round(sum(trace.latency_ms for trace in self.ollama_traces), 2),
            "end_to_end_latency_ms": round(self.end_to_end_latency_ms, 2),
            "total_latency_ms": round(self.end_to_end_latency_ms, 2),
            "mcp_protocol_failures": len(self.protocol_failures),
            "protocol_failure_details": self.protocol_failures,
            "evidence_count": sum(trace.evidence_count for trace in self.tool_traces),
        }


@contextmanager
def timed_call() -> Iterator[Callable[[], float]]:
    start = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    yield elapsed_ms
