from ..prompts.evaluator_prompts import EVALUATE_PROMPT
from langchain_openai import ChatOpenAI

class PolicyEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def evaluate(self, text: str):
        return self.llm.invoke(EVALUATE_PROMPT.format(text=text))
