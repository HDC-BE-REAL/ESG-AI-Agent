from __future__ import annotations

from datetime import datetime

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class RegulationMonitorInput(BaseModel):
    """Schema for monitoring regulatory updates."""

    query: str = Field(..., description="관심 있는 법령/기관/주제에 대한 사용자 요청")
    agencies: list[str] | None = Field(
        default=None, description="감시할 기관 (예: 환경부, 국토부, 고용노동부)"
    )


def _monitor_regulations(query: str, agencies: list[str] | None = None) -> str:
    targets = ", ".join(agencies) if agencies else "환경부/국토부/고용노동부"
    today = datetime.now().strftime("%Y-%m-%d")
    workflow = (
        "1) 관보/API/보도자료 수집 → 2) ESG 영향 영역 태깅 → 3) 적용 가능성 스코어 계산"
    )
    return (
        f"[규제 모니터링]\n요청: {query}\n모니터링 기관: {targets}\n"
        f"업데이트 기준일: {today}\n요약 프로세스: {workflow}"
    )


regulation_monitor_tool = StructuredTool.from_function(
    name="regulation_monitor_tool",
    description="환경부/국토부/고용노동부 등 건설사 관련 ESG 규제 업데이트를 요약한다.",
    func=_monitor_regulations,
    args_schema=RegulationMonitorInput,
)
