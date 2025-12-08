from __future__ import annotations
from typing import Any

# ---------------------------------
# 0) RAG 구성
# ---------------------------------
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

#from langchain.tools import StructuredTool
from langchain_core.tools import StructuredTool

from pydantic import BaseModel, Field

# -----------------------------
# 임베딩 및 벡터 DB
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectordb = Chroma(
    persist_directory="vector_db/esg_all",
    embedding_function=embedding_model,
    collection_name="esg_all"
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})


# ============================================================
# 1) PROMPTS (임시용 — 실서비스에서는 별도 파일이었음)
# ============================================================

# Summarizer Prompt
SUMMARIZE_PROMPT = """
다음 정책·지침 내용을 요약해줘.

요약 요구 대상:
{text}

요약 시 요구사항:
1) 핵심 목적
2) 주요 요구사항
3) 준수/평가 기준
4) 적용 대상 및 범위
5) 표준 기반 근거 요약

가능한 한 명확하고 구조적으로 정리해줘.
"""

# Comparator Prompt
COMPARE_PROMPT = """
다음 두 정책을 비교해줘.

A 정책: {policy_a}
B 정책: {policy_b}

비교 요구사항:
1) 목적 비교
2) 구조/항목 비교
3) 요구 수준 차이
4) 위험요인/관리기준 차이
5) 표준기반 근거 기반 주요 차이 요약

가장 중요한 차이를 중심으로 서술해줘.
"""

# Evaluator Prompt
EVALUATE_PROMPT = """
아래 내용을 ESG 기준에 따라 평가해줘.

대상:
{text}

평가 기준:
1) 명확성
2) 표준적합성(K-ESG, GRI 등)
3) 리스크 대응 적절성
4) 개선 필요 요소
"""

# Recommender Prompt
RECOMMEND_PROMPT = """
아래 정책·지침에 대해 개선 또는 추천 조치를 제안해줘.

대상:
{text}

제안 기준:
1) 표준 정합성 향상
2) 리스크 관리 강화
3) 운영 절차 개선
4) 모니터링 체계 강화
"""


# ============================================================
# 2) Summarizer
# ============================================================
class PolicySummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def summarize(self, text: str):
        related_docs = retriever.get_relevant_documents(text)
        context = "\n\n".join([d.page_content for d in related_docs])

        prompt = SUMMARIZE_PROMPT.format(text=text + "\n\n[관련 표준 근거]\n" + context)

        return self.llm.invoke(prompt)


# ============================================================
# 3) Comparator
# ============================================================
class PolicyComparator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def compare(self, a: str, b: str):
        context_a = retriever.get_relevant_documents(a)
        context_b = retriever.get_relevant_documents(b)

        context = "\n\n".join([d.page_content for d in context_a + context_b])

        prompt = COMPARE_PROMPT.format(policy_a=a, policy_b=b)
        prompt += "\n\n[표준 기반 근거]\n" + context

        return self.llm.invoke(prompt)


# ============================================================
# 4) Evaluator
# ============================================================
class PolicyEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def evaluate(self, text: str):
        return self.llm.invoke(EVALUATE_PROMPT.format(text=text))


# ============================================================
# 5) Recommender
# ============================================================
class PolicyRecommender:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def recommend(self, text: str):
        return self.llm.invoke(RECOMMEND_PROMPT.format(text=text))


# ============================================================
# 6) PolicyTool 본체 (summarize / compare / evaluate / recommend)
# ============================================================
class PolicyTool:
    """
    정책 관련 요청을 처리하는 메인 Tool.
    - 원래는 여러 파일에 나뉘어 있던 기능을 모두 단일 파일로 통합한 버전.
    """

    name = "policy_tool"

    keywords = [
        "정책", "지침", "kesg", "k-esg", "gri", "sasb",
        "요약", "비교", "평가", "추천", "개선", "정합성", "gap"
    ]

    STANDARD_KEYWORDS = {
        "K-ESG": ["kesg", "k-esg", "k esg", "한국 esg", "정부 esg"],
        "GRI": ["gri", "global reporting", "gri standard"],
        "SASB": ["sasb", "industry standard", "sustainability accounting"],
        "ISSB": ["issb", "ifrs s1", "ifrs s2"],
    }

    def matches(self, query: str) -> bool:
        text = query.lower()
        return any(k in text for k in self.keywords)

    def detect_standard(self, text: str) -> str:
        t = text.lower()
        for std, keys in self.STANDARD_KEYWORDS.items():
            if any(k in t for k in keys):
                return std
        return "UNKNOWN"

    def detect_mode(self, text: str) -> str:
        t = text.lower()

        if "비교" in t or "compare" in t:
            return "compare"
        if "평가" in t or "evaluate" in t:
            return "evaluate"
        if "추천" in t or "개선" in t or "recommend" in t:
            return "recommend"
        return "summarize"

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
        return f"[ERROR] Unknown mode: {mode}"

    def run(self, state):
        query = state["query"]

        standard = self.detect_standard(query)
        base_info = f"[감지된 기준: {standard}]"

        mode = self.detect_mode(query)

        result = self.run_mode(mode, query)

        return base_info + "\n\n" + result


# Graph/LangGraph에서 import할 실제 인스턴스
policy_tool = PolicyTool()
