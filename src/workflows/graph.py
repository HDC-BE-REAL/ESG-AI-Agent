from __future__ import annotations

from typing import Callable, Dict

from langgraph.graph import END, StateGraph

from src.tools import (
    policy_guideline_tool,
    regulation_monitor_tool,
    report_draft_tool,
    risk_assessment_tool,
)
from .schema import AgentState


TOOLS: Dict[str, Callable] = {
    policy_guideline_tool.name: policy_guideline_tool,
    risk_assessment_tool.name: risk_assessment_tool,
    report_draft_tool.name: report_draft_tool,
    regulation_monitor_tool.name: regulation_monitor_tool,
}

KEYWORDS = {
    policy_guideline_tool.name: ["정책", "지침", "평가", "ISO", "기준"],
    risk_assessment_tool.name: ["리스크", "안전", "위험", "체크리스트", "협력"],
    report_draft_tool.name: ["보고", "SDGs", "ISO 26000", "지속가능"],
    regulation_monitor_tool.name: ["규제", "법", "업데이트", "모니터링", "국토", "환경", "고용"],
}


def detect_mode(state: AgentState) -> AgentState:
    query = state["query"]
    lowered = query.lower()
    selected = policy_guideline_tool.name
    for tool_name, words in KEYWORDS.items():
        if any(word.lower() in lowered for word in words):
            selected = tool_name
            break
    return {"tool_choice": selected}


def execute_tool(state: AgentState) -> AgentState:
    tool_name = state.get("tool_choice")
    if tool_name is None:
        raise ValueError("Tool choice missing. Run detect_mode first.")
    tool = TOOLS[tool_name]
    payload = {"query": state["query"]}
    result = tool.invoke(payload)
    return {"tool_result": result}


def generate_final_answer(state: AgentState) -> AgentState:
    result = state.get("tool_result", "")
    tool = state.get("tool_choice", "지정되지 않음")
    answer = (
        "요청 내용을 분석해 연관된 모듈을 실행했습니다.\n"
        f"선택된 모듈: {tool}\n"
        f"결과:\n{result}"
    )
    return {"final_answer": answer}


def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("detect_mode", detect_mode)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("generate_final_answer", generate_final_answer)

    graph.set_entry_point("detect_mode")
    graph.add_edge("detect_mode", "execute_tool")
    graph.add_edge("execute_tool", "generate_final_answer")
    graph.add_edge("generate_final_answer", END)
    return graph.compile()
