from ..prompts.recommender_prompts import RECOMMEND_PROMPT
from langchain_openai import ChatOpenAI

class PolicyRecommender:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def recommend(self, text: str):
        return self.llm.invoke(RECOMMEND_PROMPT.format(text=text))
