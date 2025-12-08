from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state for the LangGraph ESG agent."""

    query: str
    mode: str
    tool_choice: str
    tool_result: str
    final_answer: str
