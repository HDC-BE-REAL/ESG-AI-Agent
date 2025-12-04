
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import torch
from config import PipelineConfig

class Reranker:
    """
    Step 5: Re-ranking
    - Cross-Encoder를 사용하여 문서와 쿼리의 연관성 재평가
    """
    
    def __init__(self, config: PipelineConfig, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.embedding_device == "cuda" else "cpu"
        print(f"Loading Reranker model: {model_name} on {self.device}")
        
        try:
            self.model = CrossEncoder(model_name, device=self.device)
        except Exception as e:
            print(f"Failed to load Reranker model: {e}")
            self.model = None
            
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        문서 재순위화
        """
        if not self.model or not documents:
            return documents[:top_k]
            
        # (Query, Document) 쌍 생성
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 점수 계산
        scores = self.model.predict(pairs)
        
        # 점수와 함께 문서 정렬
        scored_docs = []
        for i, doc in enumerate(documents):
            doc.metadata["relevance_score"] = float(scores[i])
            scored_docs.append((doc, scores[i]))
            
        # 점수 내림차순 정렬
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Top-k 반환
        return [doc for doc, score in scored_docs[:top_k]]

if __name__ == "__main__":
    # Test
    config = PipelineConfig()
    reranker = Reranker(config)
    
    docs = [
        Document(page_content="현대건설은 2023년 탄소중립을 선언했다."),
        Document(page_content="삼성전자는 반도체 매출이 증가했다."),
        Document(page_content="현대건설의 영업이익은 전년 대비 상승했다.")
    ]
    
    q = "현대건설 환경 정책"
    reranked = reranker.rerank(q, docs, top_k=2)
    
    for d in reranked:
        print(f"Score: {d.metadata.get('relevance_score'):.4f} | Content: {d.page_content}")
