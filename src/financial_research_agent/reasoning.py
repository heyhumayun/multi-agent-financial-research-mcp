from __future__ import annotations

import json
import re
import time
from contextvars import ContextVar
from urllib.error import URLError
from urllib.request import Request, urlopen

from financial_research_agent.config import load_settings

_OLLAMA_EVENTS: ContextVar[list[dict] | None] = ContextVar("ollama_events", default=None)


def reset_reasoning_telemetry() -> None:
    _OLLAMA_EVENTS.set([])


def consume_reasoning_telemetry() -> list[dict]:
    events = list(_OLLAMA_EVENTS.get() or [])
    _OLLAMA_EVENTS.set([])
    return events


class ReasoningEngine:
    """Small reasoning abstraction.

    By default this returns deterministic explanations. If FIN_RESEARCH_LLM=ollama,
    it calls a local Ollama model and falls back to deterministic reasoning when
    Ollama is not running.
    """

    def reason(self, agent_name: str, evidence_summary: str) -> str:
        settings = load_settings()
        if settings.llm_provider == "ollama":
            prompt = (
                "You are a concise financial research agent. Explain how this agent should "
                "interpret its tool evidence. Avoid investment advice and keep the answer to "
                f"two sentences.\n\nAgent: {agent_name}\nEvidence summary: {evidence_summary}"
            )
            response = _call_ollama(prompt, model=settings.ollama_model, operation="reasoning")
            if response:
                return response
        return (
            f"{agent_name} interpreted tool evidence using deterministic rules: {evidence_summary}"
        )

    def critique(self, agent_name: str, weakness: str) -> str:
        settings = load_settings()
        if settings.llm_provider == "ollama":
            prompt = (
                "You are a skeptical financial model reviewer. Write a short self-critique "
                "for this agent's limitation. Keep it to one sentence.\n\n"
                f"Agent: {agent_name}\nKnown weakness: {weakness}"
            )
            response = _call_ollama(prompt, model=settings.ollama_model, operation="critique")
            if response:
                return response
        return f"{agent_name} self-check: {weakness}"

    def plan_agents(self, query: str, fallback: list[str]) -> list[str]:
        settings = load_settings()
        if settings.llm_provider != "ollama":
            return fallback

        prompt = (
            "You are a financial research supervisor. Choose which specialist agents should run "
            "for the user query. Available agent keys: market, news, risk, research, document, "
            "fundamentals, comparison. "
            "Return only a JSON array of keys. Always include market, news, and risk for equity "
            f"risk questions.\n\nQuery: {query}"
        )
        response = _call_ollama(prompt, model=settings.ollama_model, operation="planning")
        selected = _parse_agent_plan(response)
        if selected:
            selected = list(dict.fromkeys([*fallback, *selected]))
        return selected or fallback

    def follow_up_agents(self, query: str, findings: list[object], planned: list[str]) -> list[str]:
        """Choose at most one evidence-recovery pass after the first review."""
        del query
        markers = {
            "market": "Market Data Agent",
            "news": "News Agent",
            "risk": "Risk Analysis Agent",
            "research": "Research/Papers Agent",
            "document": "Document Agent",
            "fundamentals": "Fundamentals Agent",
            "comparison": "Comparison Agent",
        }
        empty_agents = {
            key
            for key, marker in markers.items()
            if any(
                getattr(finding, "agent_name", "") == marker
                and any(
                    result.output == [] and not result.evidence
                    for result in getattr(finding, "tool_results", [])
                )
                for finding in findings
            )
        }
        return [key for key in planned if key in empty_agents][:1]

    def synthesize(self, query: str, finding_summaries: list[str], fallback: str) -> str:
        settings = load_settings()
        if settings.llm_provider != "ollama":
            return fallback

        prompt = (
            "You are a professional financial research analyst. Write one concise, neutral "
            "thesis from the supplied agent findings. State observations, uncertainty, evidence "
            "quality, and risks only. Do not give or imply investment advice. Never use the words "
            "recommend, recommendation, buy, sell, invest, long, short, position, or trade. "
            "Return only the thesis paragraph.\n\n"
            f"Query: {query}\nFindings:\n- " + "\n- ".join(finding_summaries)
        )
        response = _call_ollama(prompt, model=settings.ollama_model, operation="synthesis")
        if not response or _contains_advice_language(response):
            return fallback
        return response

    def judge_report(self, query: str, report_markdown: str) -> dict:
        settings = load_settings()
        if settings.llm_provider != "ollama":
            return {}

        prompt = (
            "You are evaluating an agentic financial research brief. Score the brief from 0 to 1 "
            "on groundedness, risk coverage, contradiction handling, and actionability as a "
            "research aid. Return only JSON like "
            '{"score": 0.8, "rationale": "short reason"}.\n\n'
            f"Query: {query}\nBrief:\n{report_markdown[:5000]}"
        )
        response = _call_ollama(prompt, model=settings.ollama_model, operation="judging")
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return {}
        score = payload.get("score")
        rationale = payload.get("rationale")
        if not isinstance(score, int | float) or not isinstance(rationale, str):
            return {}
        return {"score": max(0.0, min(float(score), 1.0)), "rationale": rationale}


def _call_ollama(prompt: str, model: str, operation: str = "generation") -> str:
    start = time.perf_counter()
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        data = {}
    generated = str(data.get("response", "")).strip()
    events = list(_OLLAMA_EVENTS.get() or [])
    events.append(
        {
            "operation": operation,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "succeeded": bool(generated),
        }
    )
    _OLLAMA_EVENTS.set(events)
    return generated


def _contains_advice_language(text: str) -> bool:
    lowered = text.lower()
    blocked = ("recommend", "buy", "sell", "invest", "long", "short", "position", "trade")
    return any(re.search(rf"\b{re.escape(word)}\w*\b", lowered) for word in blocked)


def _parse_agent_plan(response: str) -> list[str]:
    allowed = {"market", "news", "risk", "research", "document", "fundamentals", "comparison"}
    if not response:
        return []
    try:
        raw = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("[")
        end = response.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            raw = json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    selected = [str(item).lower() for item in raw if str(item).lower() in allowed]
    ordered = [
        item
        for item in ["market", "news", "risk", "research", "document", "fundamentals", "comparison"]
        if item in selected
    ]
    return ordered
