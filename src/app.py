from __future__ import annotations

from src.workflows import build_workflow

from dotenv import load_dotenv
load_dotenv()


def run_agent(query: str) -> str:
    app = build_workflow()
    state = app.invoke({"query": query})
    return state["final_answer"]


if __name__ == "__main__":
    sample_query = "현장 안전과 협력사 리스크를 포함한 ESG 체크리스트 만들어줘"
    print(run_agent(sample_query))
