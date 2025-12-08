from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..prompts.summarizer_prompts import SUMMARIZE_PROMPT
from src.tools.policy_tool.policy_tool import retriever


class PolicySummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def summarize(self, text: str):
        # 1) 관련 ESG 표준 문서 검색
        related_docs = retriever.get_relevant_documents(text)
        context = "\n\n".join([d.page_content for d in related_docs])

        # 2) LLM 프롬프트 구성
        prompt = SUMMARIZE_PROMPT.format(text=text + "\n\n[관련 표준 근거]\n" + context)

        return self.llm.invoke(prompt)
