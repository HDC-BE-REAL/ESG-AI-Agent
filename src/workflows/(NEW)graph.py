from __future__ import annotations
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

# 모든 tool을 registry에서 자동 로드
from src.tools import TOOL_REGISTRY


# -----------------------------
# Agent State (협업자 구조 유지하며 확장만 수행)
# -----------------------------
class AgentState(TypedDict, total=False):
    query: str
    tool_choice: str
    tool_result: Any
    final_answer: str


# -----------------------------
# Node 1: Tool 자동 선택
# -----------------------------
def detect_mode(state: AgentState) -> AgentState:
    query = state["query"]

    selected_tool = None

    # 🔥 각 tool에게 matches(query)를 물어보는 방식
    for tool_name, tool in TOOL_REGISTRY.items():
        try:
            if hasattr(tool, "matches") and tool.matches(query):
                selected_tool = tool_name
                break
        except Exception:
            continue

    # Fallback → policy_tool로 지정
    if selected_tool is None:
        selected_tool = "policy_tool"

    state["tool_choice"] = selected_tool
    return state


# -----------------------------
# Node 2: 선택된 Tool 실행
# -----------------------------
def execute_tool(state: AgentState) -> AgentState:
    tool_name = state["tool_choice"]
    tool = TOOL_REGISTRY[tool_name]

    # 🔥 tool은 run(state)를 가지고 있어야 함 (정책툴 내부 모드 포함)
    if hasattr(tool, "run"):
        result = tool.run(state)
    else:
        # 기존 타입(LangChain Tool 같은)의 경우 invoke 사용
        result = tool.invoke({"query": state["query"]})

    state["tool_result"] = result
    return state


# -----------------------------
# Node 3: 최종 결과 정리
# -----------------------------
def generate_final_answer(state: AgentState) -> AgentState:
    tool_name = state.get("tool_choice", "선택되지 않음")
    result = state.get("tool_result", "")

    state["final_answer"] = (
        f"[🔍 실행된 모듈: {tool_name}]\n\n"
        f"{result}"
    )
    return state


# -----------------------------
# Graph Builder
# -----------------------------
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
