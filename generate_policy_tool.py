# generate_policy_tool.py
import os

BASE = "src/tools/policy_tool"

FILES = {
    "__init__.py": "",
    "policy_tool.py": """
from langchain.tools import tool
from .summarizers.policy_summarizer import PolicySummarizer
from .comparators.policy_comparator import PolicyComparator
from .recommenders.policy_recommender import PolicyRecommender
from .evaluators.policy_evaluator import PolicyEvaluator


class PolicyTool:
    @tool
    def summarize_policy(self, text: str) -> str:
        return PolicySummarizer().summarize(text)

    @tool
    def compare_policies(self, policy_a: str, policy_b: str):
        return PolicyComparator().compare(policy_a, policy_b)

    @tool
    def recommend_policy(self, text: str):
        return PolicyRecommender().recommend(text)

    @tool
    def evaluate_policy(self, text: str):
        return PolicyEvaluator().evaluate(text)
""",
}

DIRS = {
    "parsers": {
        "__init__.py": "",
        "base_parser.py": """
class BasePolicyParser:
    def parse(self, text: str) -> dict:
        raise NotImplementedError
""",
        "policy_parser.py": """
from .base_parser import BasePolicyParser

class PolicyParser(BasePolicyParser):
    def parse(self, text: str) -> dict:
        # TODO: Add real parsing logic
        return {"sections": [], "requirements": []}
""",
        "requirement_extractor.py": """
class RequirementExtractor:
    def extract(self, parsed_doc: dict):
        # TODO: Real extraction logic
        return []
""",
    },
    "summarizers": {
        "__init__.py": "",
        "policy_summarizer.py": """
from ..prompts.summarizer_prompts import SUMMARIZE_PROMPT
from langchain_openai import ChatOpenAI

class PolicySummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def summarize(self, text: str) -> str:
        return self.llm.invoke(SUMMARIZE_PROMPT.format(text=text))
""",
    },
    "comparators": {
        "__init__.py": "",
        "policy_comparator.py": """
from ..prompts.comparator_prompts import COMPARE_PROMPT
from langchain_openai import ChatOpenAI

class PolicyComparator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def compare(self, a: str, b: str):
        return self.llm.invoke(COMPARE_PROMPT.format(policy_a=a, policy_b=b))
""",
    },
    "recommenders": {
        "__init__.py": "",
        "policy_recommender.py": """
from ..prompts.recommender_prompts import RECOMMEND_PROMPT
from langchain_openai import ChatOpenAI

class PolicyRecommender:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def recommend(self, text: str):
        return self.llm.invoke(RECOMMEND_PROMPT.format(text=text))
""",
    },
    "evaluators": {
        "__init__.py": "",
        "policy_evaluator.py": """
from ..prompts.evaluator_prompts import EVALUATE_PROMPT
from langchain_openai import ChatOpenAI

class PolicyEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def evaluate(self, text: str):
        return self.llm.invoke(EVALUATE_PROMPT.format(text=text))
""",
    },
    "prompts": {
        "__init__.py": "",
        "summarizer_prompts.py": """
SUMMARIZE_PROMPT = \"\"\"
당신은 ESG 정책 요약 전문가입니다.
다음 정책 문서를 핵심 항목 중심으로 요약하세요:

{text}
\"\"\"
""",
        "comparator_prompts.py": """
COMPARE_PROMPT = \"\"\"
두 정책 문서를 비교하여 다음을 도출하세요:

1. 공통점
2. 차이점
3. 누락 요소(Gap)
4. 개선 권고사항

[정책 A]
{policy_a}

[정책 B]
{policy_b}
\"\"\"
""",
        "evaluator_prompts.py": """
EVALUATE_PROMPT = \"\"\"
정책 문서를 아래 기준에 따라 평가하세요:

- 명확성
- 측정 가능성
- 책임성
- 투명성
- 글로벌 기준 정합성

출력은 JSON 형식으로 제공하세요.

{text}
\"\"\"
""",
        "recommender_prompts.py": """
RECOMMEND_PROMPT = \"\"\"
문서를 분석하고 ESG 정책 개선안을 제안하세요.

1. 부족한 항목
2. 개선 필요 이유
3. 글로벌 기준 기반 템플릿 제안

{text}
\"\"\"
""",
    },
    "utils": {
        "__init__.py": "",
        "schema.py": """
from pydantic import BaseModel
from typing import List, Optional

class PolicySection(BaseModel):
    title: str
    content: str

class PolicyDocument(BaseModel):
    sections: List[PolicySection]
    requirements: List[str] = []
""",
        "scoring.py": """
def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))
""",
    },
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main():
    print(f"📁 생성 시작: {BASE}")
    ensure_dir(BASE)

    # top-level files
    for filename, content in FILES.items():
        write_file(os.path.join(BASE, filename), content)
        print("  ✔", filename)

    # subdirectories
    for folder, files in DIRS.items():
        folder_path = os.path.join(BASE, folder)
        ensure_dir(folder_path)
        print("📁", folder)
        for filename, content in files.items():
            write_file(os.path.join(folder_path, filename), content)
            print("   ✔", filename)

    print("\n🚀 policy_tool 전체 구조 생성 완료!")


if __name__ == "__main__":
    main()
