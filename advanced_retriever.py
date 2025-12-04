
import os
from typing import List, Dict, Optional, Union, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from config import PipelineConfig
from query_processor import QueryProcessor
from reranker import Reranker

class AdvancedRetriever:
    """
    고급 검색 기능을 제공하는 Retriever
    - Similarity Search (Top-k)
    - MMR (Maximal Marginal Relevance)
    - Metadata Filtering
    - Query Re-writing
    """
    
    def __init__(self, config: PipelineConfig, vectorstore: Optional[Chroma] = None):
        self.config = config
        
        # 임베딩 모델 초기화 (vectorstore가 없을 경우 필요)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.embedding_device},
            encode_kwargs={'normalize_embeddings': config.normalize_embeddings}
        )
        
        # VectorStore 로드
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            if os.path.exists(config.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=config.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError(f"VectorStore not found at {config.persist_directory}")
                
        # LLM 초기화 (Query Rewriting용)
        self.llm = None
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                api_key=config.openai_api_key,
                model_name="gpt-4o", # or config.llm_model
                temperature=0
            )
            
        # 8-Step Workflow Components
        self.query_processor = QueryProcessor(config)
        self.reranker = Reranker(config)
            
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """기본 유사도 검색"""
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
        
    def mmr_search(
        self, 
        query: str, 
        k: int = 5, 
        fetch_k: int = 20, 
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        MMR (Maximal Marginal Relevance) 검색
        - lambda_mult: 0~1. 1에 가까울수록 유사도 우선, 0에 가까울수록 다양성 우선
        """
        return self.vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=fetch_k, 
            lambda_mult=lambda_mult, 
            filter=filter
        )
        
    def rewrite_query(self, query: str) -> str:
        """
        LLM을 사용하여 검색 쿼리 개선
        """
        if not self.llm:
            print("Warning: LLM not initialized. Skipping query rewriting.")
            return query
            
        prompt = PromptTemplate.from_template(
            """당신은 ESG 보고서 검색을 돕는 AI 어시스턴트입니다.
사용자의 질문을 검색 엔진이 더 잘 이해할 수 있도록 구체적이고 키워드 중심으로 재작성해주세요.
원래 의도를 유지하되, ESG 관련 전문 용어를 포함하면 좋습니다.

사용자 질문: {query}

재작성된 쿼리(설명 없이 쿼리만 출력):"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"query": query})
        return rewritten_query.strip()
        
    def retrieve(
        self,
        query: str,
        k: int = 5,
        search_type: str = "similarity", # "similarity" or "mmr"
        use_rewrite: bool = False,
        filter_company: Optional[str] = None,
        filter_page_min: Optional[int] = None,
        filter_page_max: Optional[int] = None,
        filter_type: Optional[str] = None
    ) -> List[Document]:
        """
        통합 검색 메소드
        """
        # 1. Query Rewriting
        search_query = query
        if use_rewrite:
            print(f"Original Query: {query}")
            search_query = self.rewrite_query(query)
            print(f"Rewritten Query: {search_query}")
            
        # 2. Build Metadata Filter
        filter_dict = {}
        conditions = []
        
        if filter_company:
            conditions.append({"company": filter_company})
        if filter_type:
            conditions.append({"type": filter_type})
            
        # Page range filter (Chroma syntax might vary, assuming simple match or $gte/$lte)
        # Chroma requires specific syntax for operators.
        # If multiple conditions, use $and
        if filter_page_min is not None:
            conditions.append({"page": {"$gte": filter_page_min}})
        if filter_page_max is not None:
            conditions.append({"page": {"$lte": filter_page_max}})
            
        if len(conditions) == 1:
            filter_dict = conditions[0]
        elif len(conditions) > 1:
            filter_dict = {"$and": conditions}
        else:
            filter_dict = None
            
        print(f"Search Type: {search_type}, Filter: {filter_dict}")
            
        # 3. Search
        if search_type == "mmr":
            results = self.mmr_search(search_query, k=k, filter=filter_dict)
        else:
            results = self.similarity_search(search_query, k=k, filter=filter_dict)
            
        return results

    def retrieve_8step(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        use_reranker: bool = True
    ) -> Dict[str, Any]:
        """
        8단계 RAG 워크플로우 실행
        """
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 1단계: 쿼리 입력                     │")
        print(f"├{'─'*40}┤")
        print(f"│  사용자: {query[:30]:<30}  │")
        print(f"└{'─'*40}┘")
        
        # 2단계: Query Processing
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 2단계: Query Processing              │")
        print(f"├{'─'*40}┤")
        processed_query = self.query_processor.process(query)
        rewritten_query = processed_query.get("rewritten_query", query)
        entities = processed_query.get("entities", {})
        intent = processed_query.get("intent", "general")
        print(f"│  A. Rewritten: {rewritten_query[:25]:<25} │")
        print(f"│  B. Entities: {str(entities)[:25]:<26} │")
        print(f"│  C. Intent: {intent:<28} │")
        print(f"└{'─'*40}┘")
        
        # 3단계: Pre-filtering
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 3단계: Pre-filtering                 │")
        print(f"├{'─'*40}┤")
        
        filter_dict = {}
        conditions = []
        
        # Entity 기반 필터링 자동 적용
        if entities.get("company"):
            conditions.append({"company": entities["company"]})
            print(f"│  - Company Filter: {entities['company']:<20} │")
            
        # Page range or other filters could be added here
        
        if len(conditions) == 1:
            filter_dict = conditions[0]
        elif len(conditions) > 1:
            filter_dict = {"$and": conditions}
        else:
            filter_dict = None
            
        print(f"│  - Filter: {str(filter_dict)[:28]:<28} │")
        print(f"└{'─'*40}┘")
        
        # 4단계: Vector Search
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 4단계: Vector Search                 │")
        print(f"├{'─'*40}┤")
        print(f"│  - Fetching {fetch_k} docs...                 │")
        
        # 1차 검색은 넉넉하게 fetch_k개 가져옴
        initial_docs = self.similarity_search(rewritten_query, k=fetch_k, filter=filter_dict)
        print(f"│  - Found: {len(initial_docs)} docs                      │")
        print(f"└{'─'*40}┘")
        
        # 5단계: Re-ranking
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 5단계: Re-ranking                   │")
        print(f"├{'─'*40}┤")
        
        if use_reranker and self.reranker.model:
            print(f"│  - Using Cross-Encoder...                │")
            reranked_docs = self.reranker.rerank(rewritten_query, initial_docs, top_k=fetch_k) # 점수 계산을 위해 전체 rerank
        else:
            print(f"│  - Skipping (Similarity only)            │")
            reranked_docs = initial_docs
            
        print(f"└{'─'*40}┘")
        
        # 6단계: Post-filtering
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 6단계: Post-filtering               │")
        print(f"├{'─'*40}┤")
        
        final_docs = []
        threshold = 0.0 # 필요시 설정 (예: -5.0 for CrossEncoder logits)
        
        for doc in reranked_docs:
            score = doc.metadata.get("relevance_score", 0)
            # CrossEncoder 점수는 Logits일 수 있으므로 임계값 주의
            final_docs.append(doc)
            
        print(f"│  - Filtered to: {len(final_docs)} docs                │")
        print(f"└{'─'*40}┘")
        
        # 7단계: Top-k Selection
        print(f"\n┌{'─'*40}┐")
        print(f"│  🟢 7단계: Top-k Selection               │")
        print(f"├{'─'*40}┤")
        
        top_k_docs = final_docs[:k]
        print(f"│  - Selecting Top-{k}                     │")
        print(f"└{'─'*40}┘")
        
        return {
            "query_info": processed_query,
            "documents": top_k_docs
        }

# Test Code
if __name__ == "__main__":
    # Config
    config = PipelineConfig()
    # config.openai_api_key = "sk-..." # Set your key here or in config.py
    
    try:
        retriever = AdvancedRetriever(config)
        
        query = "탄소 중립 목표"
        
        print("\n--- Similarity Search ---")
        docs = retriever.retrieve(query, k=3)
        for d in docs:
            print(f"- {d.page_content[:50]}... (Meta: {d.metadata})")
            
        print("\n--- MMR Search ---")
        docs = retriever.retrieve(query, k=3, search_type="mmr")
        for d in docs:
            print(f"- {d.page_content[:50]}... (Meta: {d.metadata})")
            
    except Exception as e:
        print(f"Error: {e}")
