from langchain_openai import ChatOpenAI
from ..prompts.comparator_prompts import COMPARE_PROMPT

class PolicyComparator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def compare(self, a: str, b: str):
        context_a = retriever.get_relevant_documents(a)
        context_b = retriever.get_relevant_documents(b)

        context = "\n\n".join(
            [d.page_content for d in context_a + context_b]
        )

        prompt = COMPARE_PROMPT.format(policy_a=a, policy_b=b)
        prompt += "\n\n[표준 기반 근거]\n" + context

        return self.llm.invoke(prompt)

