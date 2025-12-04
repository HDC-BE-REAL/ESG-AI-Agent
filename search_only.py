
import os
import sys
from config import PipelineConfig
from advanced_retriever import AdvancedRetriever

def main():
    # 1. 설정 로드
    config = PipelineConfig()
    
    # 2. Retriever 초기화 (이미 적재된 Vector DB 사용)
    try:
        retriever = AdvancedRetriever(config)
    except Exception as e:
        print(f"Retriever 초기화 실패: {e}")
        print("먼저 'python demo_retriever.py --reingest'를 실행하여 데이터를 적재해주세요.")
        return

    # 3. 검색 실행
    while True:
        query = input("\n검색할 내용을 입력하세요 (종료: q): ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
            
        if not query:
            continue
            
        print(f"\n🔍 '{query}' 검색 결과:")
        print("-" * 50)
        
        # 기본 유사도 검색 (Top-3)
        results = retriever.retrieve(query, k=3)
        
        if not results:
            print("검색 결과가 없습니다.")
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('score', 'N/A') # Chroma returns distance usually, but wrapper might not expose it directly in doc
            
            print(f"[{i}] (Length: {len(doc.page_content)} chars)")
            print(f"{doc.page_content}")
            print(f"    - 출처: {os.path.basename(source)} (p.{page})")
            print("-" * 50)

if __name__ == "__main__":
    main()
