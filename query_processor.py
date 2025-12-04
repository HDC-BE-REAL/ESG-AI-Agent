
import json
from typing import Dict, Any, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import PipelineConfig

class QueryProcessor:
    """
    Step 2: Query Processing
    - Query Rewriting
    - Entity Extraction
    - Intent Detection
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = None
        
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                api_key=config.openai_api_key,
                model_name="gpt-4o", # or config.llm_model
                temperature=0
            )
            
    def process(self, query: str) -> Dict[str, Any]:
        """
        쿼리를 분석하여 구조화된 정보를 반환
        """
        if not self.llm:
            print("Warning: LLM not initialized. Skipping advanced query processing.")
            return {
                "original_query": query,
                "rewritten_query": query,
                "entities": {},
                "intent": "general"
            }
            
        # 프롬프트 정의
        prompt = PromptTemplate.from_template(
            """당신은 ESG 보고서 검색 시스템의 쿼리 분석가입니다.
사용자의 질문을 분석하여 다음 정보를 JSON 형식으로 추출해주세요.

1. rewritten_query: 검색 엔진에 최적화된 쿼리 (키워드 중심, 명확하게)
2. entities: 질문에 포함된 주요 엔티티 (없으면 빈 딕셔너리)
   - company: 회사명 (예: 삼성전자, 현대건설)
   - year: 연도 (예: 2023, 2024)
   - metric: 지표 (예: 영업이익, 탄소배출량)
3. intent: 질문의 의도 (fact, summary, comparison 중 하나)

사용자 질문: {query}

JSON 출력:"""
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({"query": query})
            result["original_query"] = query
            return result
        except Exception as e:
            print(f"Query processing failed: {e}")
            return {
                "original_query": query,
                "rewritten_query": query,
                "entities": {},
                "intent": "general"
            }

if __name__ == "__main__":
    # Test
    config = PipelineConfig()
    # config.openai_api_key = "sk-..." 
    processor = QueryProcessor(config)
    
    q = "현대건설 2023년 탄소배출량은 얼마야?"
    print(f"Input: {q}")
    res = processor.process(q)
    print(json.dumps(res, indent=2, ensure_ascii=False))
