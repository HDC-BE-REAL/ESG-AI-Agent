
import os
import sys
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import PipelineConfig
from advanced_retriever import AdvancedRetriever

def main():
    # Load .env
    load_dotenv()
    
    # 1. Configuration
    config = PipelineConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY")
    )
    
    # API Key Check
    if not config.openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Query processing will be limited.")
        
    try:
        retriever = AdvancedRetriever(config)
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    # Test Queries
    queries = [
        "현대건설 2023년 탄소배출량은?",
        "삼성전자 영업이익 알려줘", # (데이터 없음 테스트)
        "ESG 전략에 대해 요약해줘"
    ]
    
    for query in queries:
        print("\n" + "="*60)
        print(f"TEST QUERY: {query}")
        print("="*60)
        
        try:
            result = retriever.retrieve_8step(
                query, 
                k=3, 
                fetch_k=10, 
                use_reranker=True
            )
            
            print("\n┌─────────────────────────────────────────┐")
            print("│  🟢 8단계: LLM에게 전달 (Simulation)    │")
            print("├─────────────────────────────────────────┤")
            
            docs = result["documents"]
            context = ""
            for i, doc in enumerate(docs):
                score = doc.metadata.get("relevance_score", "N/A")
                if isinstance(score, float):
                    score = f"{score:.4f}"
                
                doc_info = f"[{i+1}] Score: {score} | {doc.page_content[:40]}..."
                print(f"│  {doc_info:<39} │")
                context += f"Document {i+1}:\n{doc.page_content}\n\n"
                
            print("└─────────────────────────────────────────┘")
            
            # 8단계: 결과 생성
            if config.openai_api_key:
                print("\n┌─────────────────────────────────────────┐")
                print("│  🟢 9단계: 최종 답변 생성 (LLM)         │")
                print("├─────────────────────────────────────────┤")
                
                llm = ChatOpenAI(
                    api_key=config.openai_api_key,
                    model_name="gpt-4o",
                    temperature=0
                )
                
                prompt = PromptTemplate.from_template(
                    """당신은 ESG 전문가입니다. 아래 문서들을 바탕으로 사용자의 질문에 답변해주세요.
문서에 정보가 없으면 "문서에서 정보를 찾을 수 없습니다."라고 답하세요.

질문: {query}

관련 문서:
{context}

답변:"""
                )
                
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"query": query, "context": context})
                
                print(f"│  {answer[:50]}... (생략)                  │")
                print("└─────────────────────────────────────────┘")
                print(f"\n[전체 답변]\n{answer}")
            else:
                print("\n⚠️ OPENAI_API_KEY가 없어서 답변 생성을 건너뜁니다.")
            
        except Exception as e:
            print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()
