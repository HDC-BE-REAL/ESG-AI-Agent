import sys
import os
import logging
from typing import Dict, Any, Optional

# Add project root to sys.path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.regulation_tool import _monitor_instance as regulation_monitor
from src.tools.risk import RiskToolOrchestrator
from backend.kv_store import kv_store

LOGGER = logging.getLogger(__name__)
class AgentManager:
    def __init__(self):
        # ① 업로드된 파일·규제 업데이트·정책 분석 등 모든 컨텍스트를 저장
        default_context: Dict[str, Any] = {
            "uploaded_files": [],
            "regulation_updates": None,
            "policy_analysis": None,
            "risk_assessment": None,
            "report_draft": None,
            "chat_history": []
        }
        persisted = kv_store.load_context() or {}
        # ④ Redis에 저장된 값이 있다면 기본 컨텍스트 위에 덮어써 복원
        default_context.update(persisted)
        self.shared_context = default_context
        self._risk_orchestrator = RiskToolOrchestrator()

    def get_context(self) -> Dict[str, Any]:
        return self.shared_context

    def update_context(self, key: str, value: Any):
        self.shared_context[key] = value
        self._persist_context()

    def _persist_context(self):
        # ⑤ Redis 사용 가능 시 전체 컨텍스트를 JSON으로 동기화
        if not kv_store.save_context(self.shared_context):
            LOGGER.warning("Redis 컨텍스트 저장 실패 - 메모리 모드로 지속")

    async def run_regulation_agent(self, query: str = "ESG 규제 동향") -> str:
        """
        Runs the Regulation Monitor tool and updates the shared context.
        """
        print(f"🚀 [AgentManager] Starting Regulation Agent with query: {query}")
        
        # Run the existing monitor logic
        # Note: monitor_all is synchronous, might block if not careful, 
        # but for now we run it directly. In production, use a thread pool or background task.
        try:
            # report = regulation_monitor.monitor_all(query)
            # Use generate_report for instant response (browsing happens in background)
            # ② regulation/policy/risk/report agent 실행
            report = regulation_monitor.generate_report(query)
            self.update_context("regulation_updates", report)
            return report
        except Exception as e:
            error_msg = f"Error running regulation agent: {str(e)}"
            print(error_msg)
            return error_msg

    async def run_policy_agent(self, query: str):
        # Placeholder
        return "Policy Agent not implemented yet."

    async def run_risk_agent(self, query: str, focus_area: Optional[str] = None) -> str:
        """리스크 오케스트레이터를 호출해 ISO31000/Materiality 결과를 생성하고 컨텍스트에 저장"""
        try:
            result = self._risk_orchestrator.run(query=query, focus_area=focus_area)
            # ③ 최신 리스크 분석 리포트를 공유 컨텍스트에 넣어 챗봇·리포트 에이전트에서 활용
            self.update_context("risk_assessment", result)
            return result
        except Exception as exc:
            error_msg = f"Risk agent 실행 오류: {exc}"
            print(error_msg)
            return error_msg

    async def run_report_agent(self, query: str):
        # Placeholder
        return "Report Agent not implemented yet."

    async def run_custom_agent(self, query: str):
        # Placeholder
        return "Custom Agent not implemented yet."

# Singleton instance
agent_manager = AgentManager()
