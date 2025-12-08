from __future__ import annotations

#from langchain.tools import StructuredTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .risk import RiskToolOrchestrator



class RiskAssessmentInput(BaseModel):
    """Schema for ESG risk diagnostics."""

    query: str = Field(..., description="User scenario covering 현장, 협력사, 또는 위험요소")
    focus_area: str | None = Field(
        default=None, description="Optional dimension such as safety, environment, labor, or governance"
    )


def _diagnose_risk(query: str, focus_area: str | None = None) -> str:
    orchestrator = RiskToolOrchestrator()
    return orchestrator.run(query=query, focus_area=focus_area)



risk_assessment_tool = StructuredTool.from_function(
    name="risk_assessment_tool",
    description="현장/협력사 ESG 리스크 체크리스트, ISO31000 분석, Materiality 분석을 생성한다.",

    func=_diagnose_risk,
    args_schema=RiskAssessmentInput,
)
