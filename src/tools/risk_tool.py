from __future__ import annotations

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class RiskAssessmentInput(BaseModel):
    """Schema for ESG risk diagnostics."""

    query: str = Field(..., description="User scenario covering 현장, 협력사, 또는 위험요소")
    focus_area: str | None = Field(
        default=None, description="Optional dimension such as safety, environment, labor, or governance"
    )


def _diagnose_risk(query: str, focus_area: str | None = None) -> str:
    focus = focus_area or "종합"
    checklist = (
        "1. 안전: 작업 허가, 실시간 위험성 평가, 비상 대응 훈련\n"
        "2. 환경: 폐기물 관리, 배출 모니터링, 생태 복원 계획\n"
        "3. 노동·인권: 근로시간, 공정한 보상, 신고 채널\n"
        "4. 거버넌스: 윤리경영 서약, 제3자 감사, 제보 보호"
    )
    analysis = (
        "환경오염·안전·노동·거버넌스 각 항목에 위험도(낮음/중간/높음)를 부여하고,"
        " 협력사 평가 항목과 연동해 개선 우선순위를 추천"
    )
    return (
        f"[ESG 리스크 진단]\n요청: {query}\n중점 영역: {focus}\n"
        f"체크리스트:\n{checklist}\n위험도 분석: {analysis}"
    )


risk_assessment_tool = StructuredTool.from_function(
    name="risk_assessment_tool",
    description="현장/협력사 ESG 리스크 체크리스트와 위험도 분석을 생성한다.",
    func=_diagnose_risk,
    args_schema=RiskAssessmentInput,
)
