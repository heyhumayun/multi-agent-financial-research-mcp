from __future__ import annotations

from abc import ABC, abstractmethod

from financial_research_agent.domain import AgentFinding
from financial_research_agent.tool_gateway import ToolGateway


class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, query: str, ticker: str, tools: ToolGateway) -> AgentFinding:
        raise NotImplementedError
