from __future__ import annotations

from typing import List

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class PolicyGuidelineInput(BaseModel):
    """Schema for policy/guideline consolidation requests."""

    query: str = Field(..., description="Original user question about policies or standards")
    target_bodies: List[str] | None = Field(
        default=None, description="Optional list of institutions or standards to prioritize"
    )


def _summarize_policies(query: str, target_bodies: List[str] | None = None) -> str:
    targets = ", ".join(target_bodies) if target_bodies else "핵심 외부 기준"
    synopsis = (
        "- 내부 기준과 외부 규격을 비교하여 공통 요구사항을 식별\n"
        "- 차이가 큰 항목에는 개선 권고안을 포함\n"
        "- 자동 추천안에는 ISO·공공평가 지표 대비 준수 상태 포함"
    )
    return (
        f"[정책·지침 요약]\n요청: {query}\n우선 기준: {targets}\n"
        f"요약 로직:\n{synopsis}\n기준안 추천: 데이터 거버넌스 강화, 공급망 감시, 지속 개선 루프"
    )


policy_guideline_tool = StructuredTool.from_function(
    name="policy_guideline_tool",
    description="내부 정책과 공공 ESG/ISO 규격을 요약·비교하고 추천 기준안을 제안한다.",
    func=_summarize_policies,
    args_schema=PolicyGuidelineInput,
)
