from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .report_tool import generate_report_from_query


class ReportDraftInput(BaseModel):
    """ESG 보고서 초안 생성을 위한 입력 스키마"""

    query: str = Field(..., description="요청에 포함된 보고 범위, 기간, 핵심 성과")
    audience: str | None = Field(default=None, description="옵션: 경영진, 이사회, 외부 공시 등")


def _draft_report(query: str, audience: str | None = None) -> str:
    """폴더 버전 ReportTool을 호출하여 HTML 보고서를 생성"""

    return generate_report_from_query(query, audience)


report_draft_tool = StructuredTool.from_function(
    name="report_draft_tool",
    description="ESG 경영·지속가능경영 보고서 초안을 생성하고 UN SDGs/ISO 26000 매핑을 제공한다.",
    func=_draft_report,
    args_schema=ReportDraftInput,
)


def draft_report(query: str, audience: str | None = None) -> str:
    return generate_report_from_query(query, audience)