
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def analyze_headers():
    load_dotenv()
    
    print("Loading Vector DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    print("Fetching all documents...")
    # ChromaDB의 get 메서드로 메타데이터만 가져오기
    data = vectorstore.get(include=['metadatas'])
    
    headers = {}
    companies = set()
    
    for meta in data['metadatas']:
        if not meta:
            continue
            
        company = meta.get('company', 'Unknown')
        companies.add(company)
        
        h1 = meta.get('Header1')
        if h1:
            if h1 not in headers:
                headers[h1] = set()
            headers[h1].add(company)
            
    print(f"\n🔍 분석 대상 기업 ({len(companies)}개): {', '.join(companies)}")
    print(f"🔍 발견된 고유 헤더 (Header1): {len(headers)}개")
    
    print("\n[공통 헤더 (2개 이상 기업에서 등장)]")
    common_headers = []
    for h, comps in headers.items():
        if len(comps) >= 1: # 지금은 기업이 적으므로 1개 이상도 다 출력해보자 (정렬해서)
            common_headers.append((h, len(comps), list(comps)))
            
    # 많이 등장한 순서대로 정렬
    common_headers.sort(key=lambda x: x[1], reverse=True)
    
    for h, count, comps in common_headers:
        print(f"- {h} ({count}개 기업): {', '.join(comps)}")

if __name__ == "__main__":
    analyze_headers()
