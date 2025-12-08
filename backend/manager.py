import sys
import os
from typing import Dict, Any, Optional

# Add project root to sys.path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.regulation_tool import _monitor_instance as regulation_monitor

class AgentManager:
    def __init__(self):
        self.shared_context: Dict[str, Any] = {
            "uploaded_files": [],
            "regulation_updates": None,
            "policy_analysis": None,
            "risk_assessment": None,
            "report_draft": None,
            "chat_history": []
        }

    def get_context(self) -> Dict[str, Any]:
        return self.shared_context

    def update_context(self, key: str, value: Any):
        self.shared_context[key] = value

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

    async def run_risk_agent(self, query: str):
        # Placeholder
        return "Risk Agent not implemented yet."

    async def run_report_agent(self, query: str):
        # Placeholder
        return "Report Agent not implemented yet."

    async def run_custom_agent(self, query: str):
        # Placeholder
        return "Custom Agent not implemented yet."

# Singleton instance
agent_manager = AgentManager()
