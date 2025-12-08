from __future__ import annotations
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .summarizers.policy_summarizer import PolicySummarizer
from .comparators.policy_comparator import PolicyComparator
from .evaluators.policy_evaluator import PolicyEvaluator
from .recommenders.policy_recommender import PolicyRecommender


# -----------------------------
# 0) RAG 구성
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectordb = Chroma(
    persist_directory="vector_db/esg_all",
    embedding_function=embedding_model,
    collection_name="esg_all"
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# PolicyTool 본체
# -----------------------------
class PolicyTool:
    """
    정책 관련 요청을 처리하는 메인 Tool.
    graph.py → execute_tool()에서 단 한 번 실행됨.
    """

    name = "policy_tool"

    keywords = [
        "정책", "지침", "kesg", "k-esg", "gri", "sasb",
        "요약", "비교", "평가", "추천", "개선", "정합성", "gap"
    ]

    # -----------------------------------------
    # 1) 이 tool이 query를 처리할 것인지 판단
    # -----------------------------------------
    def matches(self, query: str) -> bool:
        text = query.lower()
        return any(k in text for k in self.keywords)

    # -----------------------------------------
    # 2) summarize / compare / evaluate / recommend 판단
    # -----------------------------------------
    def detect_mode(self, text: str) -> str:
        t = text.lower()

        if "비교" in t or "compare" in t:
            return "compare"
        if "평가" in t or "evaluate" in t:
            return "evaluate"
        if "추천" in t or "개선" in t or "recommend" in t:
            return "recommend"
        return "summarize"

    # -----------------------------------------
    # 3) 기준 자동 감지 (K-ESG / GRI / SASB / ISSB)
    # -----------------------------------------
    STANDARD_KEYWORDS = {
        "K-ESG": ["kesg", "k-esg", "k esg", "한국 esg", "정부 esg"],
        "GRI": ["gri", "global reporting", "gri standard"],
        "SASB": ["sasb", "industry standard", "sustainability accounting"],
        "ISSB": ["issb", "ifrs s1", "ifrs s2"],
    }

    def detect_standard(self, text: str) -> str:
        t = text.lower()
        for std, keys in self.STANDARD_KEYWORDS.items():
            if any(k in t for k in keys):
                return std
        return "UNKNOWN"

    # -----------------------------------------
    # 4) mode-runner
    # -----------------------------------------
    def run_mode(self, mode: str, text: str) -> Any:

        if mode == "summarize":
            return PolicySummarizer().summarize(text)

        elif mode == "compare":
            if "|" not in text:
                return "비교하려면 '문서A | 문서B' 형식으로 입력하세요."
            a, b = [t.strip() for t in text.split("|", 1)]
            return PolicyComparator().compare(a, b)

        elif mode == "evaluate":
            return PolicyEvaluator().evaluate(text)

        elif mode == "recommend":
            return PolicyRecommender().recommend(text)

        else:
            return f"[ERROR] Unknown mode: {mode}"

    # -----------------------------------------
    # 5) graph.py에서 실행되는 Main 메서드
    # -----------------------------------------
    def run(self, state):
        query = state["query"]

        # 기준 자동 감지
        standard = self.detect_standard(query)
        base_info = f"[감지된 기준: {standard}]"

        # mode 감지
        mode = self.detect_mode(query)

        # 실행
        result = self.run_mode(mode, query)

        # 최종 응답 반환
        return base_info + "\n\n" + result


# graph.py에서 사용될 실제 인스턴스
policy_tool = PolicyTool()
