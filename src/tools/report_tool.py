from __future__ import annotations

#from langchain.tools import StructuredTool
from langchain_core.tools import StructuredTool

from pydantic import BaseModel, Field


class ReportDraftInput(BaseModel):
    """Schema for ESG/지속가능경영 보고서 초안 생성."""

    query: str = Field(..., description="요청에 포함된 보고 범위, 기간, 핵심 성과")
    audience: str | None = Field(default=None, description="옵션: 경영진, 이사회, 외부 공시 등")


def _draft_report(query: str, audience: str | None = None) -> str:
    target = audience or "일반 이해관계자"
    sections = (
        "① 경영진 메시지\n② 핵심 ESG 성과/지표\n③ UN SDGs 및 ISO 26000 매핑\n"
        "④ 리스크·기회\n⑤ 향후 계획 및 KPI"
    )
    mapping = "SDG 7/9/11/12/13, ISO 26000 핵심 주제(조직 거버넌스, 인권, 노동관행 등)를 자동 연결"
    return (
        f"[보고서 자동 생성]\n요청: {query}\n대상 독자: {target}\n"
        f"권장 섹션:\n{sections}\n매핑 로직: {mapping}"
    )


report_draft_tool = StructuredTool.from_function(
    name="report_draft_tool",
    description="ESG 경영·지속가능경영 보고서 초안을 생성하고 UN SDGs/ISO 26000 매핑을 제공한다.",
    func=_draft_report,
    args_schema=ReportDraftInput,
)


def draft_report(query: str, audience: str | None = None) -> str:
    """외부 모듈에서 텍스트 보고서를 직접 얻을 수 있도록 돕는 헬퍼"""
    return _draft_report(query, audience)
