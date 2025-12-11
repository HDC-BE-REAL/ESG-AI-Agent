from __future__ import annotations
from typing import Any

# ---------------------------------
# 0) RAG 구성
# ---------------------------------
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# ============================================================
# 1) PROMPTS (prompts 폴더에서 import)
# ============================================================
from src.tools.policy.prompts.summarizer_prompts import SUMMARIZE_PROMPT
from src.tools.policy.prompts.comparator_prompts import COMPARE_PROMPT
from src.tools.policy.prompts.evaluator_prompts import EVALUATE_PROMPT
from src.tools.policy.prompts.recommender_prompts import RECOMMEND_PROMPT




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
# 2) Summarizer
# ============================================================
class PolicySummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def summarize(self, text: str):
        related_docs = retriever.invoke(text)
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
        context_a = retriever.invoke(a)
        context_b = retriever.invoke(b)

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
        
        # [Fix] AIMessage 객체가 반환될 경우 content만 추출
        if hasattr(result, "content"):
            result = result.content

        return base_info + "\n\n" + result

    def _normalize_state(self, data: Any) -> dict:
        """허용된 입력(str 또는 dict)을 LangGraph 상태 형태로 변환"""
        if isinstance(data, str):
            return {"query": data}
        if isinstance(data, dict):
            if "query" not in data:
                raise ValueError("PolicyTool 상태에는 'query' 키가 필요합니다.")
            return data
        raise TypeError("PolicyTool 입력은 문자열 또는 {'query': str} 형태여야 합니다.")

    def __call__(self, data: Any) -> str:
        state = self._normalize_state(data)
        return self.run(state)

    def invoke(self, data: Any, *, config: Any | None = None) -> str:
        # LangChain 호환을 위해 config 인자를 허용하지만 현재는 미사용
        return self.__call__(data)


# Graph/LangGraph에서 import할 실제 인스턴스
policy_guideline_tool = PolicyTool()

# 하위 호환성을 위해 이전 이름도 유지
policy_tool = policy_guideline_tool
