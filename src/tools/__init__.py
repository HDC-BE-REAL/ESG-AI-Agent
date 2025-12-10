from .policy_tool import policy_guideline_tool

#from .regulation_tool import regulation_monitor_tool
from .regulation_tool import fetch_regulation_updates as regulation_monitor_tool

from .report_tool import report_draft_tool
from .risk_tool import risk_assessment_tool

__all__ = [
    "policy_guideline_tool",
    "regulation_monitor_tool",
    "report_draft_tool",
    "risk_assessment_tool",
]

# 중복 삽입 방지: ensure_tool 없을 때만 추가

# --- auto-wrap plain functions into LangChain tools (for .name attr) ---
from langchain_core.tools import tool as _lc_tool

def ensure_tool(obj):
    return obj if hasattr(obj, "name") else _lc_tool(obj)

# exports (may be plain functions depending on implementation)
policy_guideline_tool = ensure_tool(policy_guideline_tool)
policy_guideline_tool = ensure_tool(policy_guideline_tool)
regulation_monitor_tool = ensure_tool(regulation_monitor_tool)
report_draft_tool = ensure_tool(report_draft_tool)
risk_assessment_tool = ensure_tool(risk_assessment_tool)
