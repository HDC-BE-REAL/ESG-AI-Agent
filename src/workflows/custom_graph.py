from __future__ import annotations

"""LangGraph 기반 ESG 멀티 에이전트 파이프라인"""

from typing import Optional, TypedDict
import time

from langgraph.graph import StateGraph, END

from src.tools.policy_tool import policy_guideline_tool
from src.tools.regulation_tool import _monitor_instance as regulation_monitor
from src.tools.risk import RiskToolOrchestrator
from src.tools.report_tool import draft_report


class PipelineState(TypedDict, total=False):
    """LangGraph 상태: 질문과 각 모듈 결과"""

    query: str
    focus_area: Optional[str]
    audience: Optional[str]
    policy: str
    regulation: str
    risk: str
    report: str


_risk_orchestrator = RiskToolOrchestrator()
_graph_builder = StateGraph(PipelineState)

_REGULATION_CACHE = {"timestamp": 0.0, "result": "", "ttl": 300.0}
_REGULATION_KEYWORDS = ["규제", "법", "법령", "compliance", "legal", "업데이트", "정책"]


def _should_run_regulation(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in _REGULATION_KEYWORDS)


def _policy_node(state: PipelineState) -> PipelineState:
    state["policy"] = policy_guideline_tool(state["query"])
    return state


def _regulation_node(state: PipelineState) -> PipelineState:
    query = state["query"]
    now = time.time()

    if not _should_run_regulation(query):
        cached = _REGULATION_CACHE.get("result")
        state["regulation"] = cached or "규제 관련 요청이 없어 직전 정보를 유지합니다."
        return state

    if (
        _REGULATION_CACHE.get("result")
        and now - _REGULATION_CACHE.get("timestamp", 0) < _REGULATION_CACHE.get("ttl", 300)
    ):
        state["regulation"] = _REGULATION_CACHE["result"]
        return state

    result = regulation_monitor.generate_report(query)
    _REGULATION_CACHE["result"] = result
    _REGULATION_CACHE["timestamp"] = now
    state["regulation"] = result
    return state


def _risk_node(state: PipelineState) -> PipelineState:
    state["risk"] = _risk_orchestrator.run(state["query"], state.get("focus_area"))
    return state


def _report_node(state: PipelineState) -> PipelineState:
    state["report"] = draft_report(state["query"], state.get("audience"))
    return state


_graph_builder.add_node("policy", _policy_node)
_graph_builder.add_node("regulation", _regulation_node)
_graph_builder.add_node("risk", _risk_node)
_graph_builder.add_node("report", _report_node)

_graph_builder.set_entry_point("policy")
_graph_builder.add_edge("policy", "regulation")
_graph_builder.add_edge("regulation", "risk")
_graph_builder.add_edge("risk", "report")
_graph_builder.add_edge("report", END)

_pipeline = _graph_builder.compile()


def run_langgraph_pipeline(query: str, focus_area: Optional[str] = None, audience: Optional[str] = None) -> PipelineState:
    """LangGraph 파이프라인 실행"""

    state: PipelineState = {"query": query}
    if focus_area:
        state["focus_area"] = focus_area
    if audience:
        state["audience"] = audience
    return _pipeline.invoke(state)
