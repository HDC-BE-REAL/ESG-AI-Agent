"""
복잡한 PDF 기반 RAG 파이프라인
ESG 보고서와 같은 표/레이아웃이 복잡한 한국어 PDF 문서를 위한 고품질 RAG 시스템

Pipeline:
1. LlamaParse로 구조 보존 파싱 (Markdown)
2. Regex 기반 노이즈 제거 전처리
3. Semantic Chunking (헤더 기반 + 토큰 제한)
4. 한국어 특화 임베딩 (BAAI/bge-m3)
5. ChromaDB 벡터 스토어 저장
"""

import re
import os
from typing import List, Dict, Optional
from pathlib import Path

# Core dependencies
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json

class ESGRAGPipeline:
    """
    복잡한 PDF를 위한 고품질 RAG 파이프라인
    """
    
    def __init__(
        self,
        llama_cloud_api_key: str,
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        persist_directory: str = "./chroma_db",
        parsed_cache_dir: str = "./parsed_cache"
    ):
        """
        Args:
            llama_cloud_api_key: LlamaParse API 키
            embedding_model: 임베딩 모델 (한국어 특화 추천: BAAI/bge-m3)
            chunk_size: 청크 크기 (토큰)
            chunk_overlap: 청크 오버랩
            persist_directory: ChromaDB 저장 경로
            parsed_cache_dir: 파싱된 데이터 캐시 디렉토리
        """
        self.llama_cloud_api_key = llama_cloud_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.parsed_cache_dir = parsed_cache_dir

        # 캐시 디렉토리 생성
        Path(parsed_cache_dir).mkdir(parents=True, exist_ok=True)

        # 임베딩 모델 초기화 (한국어 특화)
        print(f"임베딩 모델 로딩 중: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectorstore = None
        
    # ==================== Step 1: Parsing ====================
    def parse_pdf(self, pdf_path: str, use_cache: bool = True) -> List[Document]:
        """
        LlamaParse를 사용하여 PDF를 Markdown으로 변환
        - 표(Table) 구조 완벽 보존
        - 레이아웃 정보 유지
        - 페이지별 메타데이터 보존
        
        Args:
            pdf_path: PDF 파일 경로
            use_cache: True일 경우 캐시된 파싱 결과 사용
            
        Returns:
            LangChain Document 객체 리스트 (페이지 단위)
        """
        print(f"\n[Step 1] PDF 파싱 시작: {pdf_path}")
        pdf_path = str(pdf_path)  # Ensure string for JSON serialization

        # 캐시 파일 경로 생성
        pdf_filename = Path(pdf_path).stem
        cache_file = Path(self.parsed_cache_dir) / f"{pdf_filename}_parsed_v2.json"

        # 캐시 확인
        if use_cache and cache_file.exists():
            print(f"✓ 캐시된 파싱 결과 사용: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                documents = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in cached_data]
                print(f"✓ 캐시 로드 완료: {len(documents)} 페이지")
                return documents

        # LlamaParse 실행
        print("⚡ LlamaParse API 호출 중... (일일 한도 사용)")
        parser = LlamaParse(
            api_key=self.llama_cloud_api_key,
            result_type="markdown",
            language="ko",
            verbose=True
        )

        llama_docs = parser.load_data(pdf_path)
        
        # LlamaIndex Document -> LangChain Document 변환
        documents = []
        for i, doc in enumerate(llama_docs):
            # 메타데이터 구성
            metadata = {
                "source": pdf_path,
                "page": i + 1,  # 1-based page number
                "total_pages": len(llama_docs)
            }
            # LlamaParse 메타데이터 병합 (선택사항)
            # if hasattr(doc, 'metadata'):
            #     metadata.update(doc.metadata)
            
            documents.append(Document(page_content=doc.text, metadata=metadata))

        # 캐시 저장
        cache_data = [
            {'page_content': doc.page_content, 'metadata': doc.metadata}
            for doc in documents
        ]
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 파싱 완료: {len(documents)} 페이지")
        return documents
    
    # ==================== Step 2: Preprocessing ====================
    def preprocess_text(self, documents: List[Document]) -> List[Document]:
        """
        Regex 기반 노이즈 제거 및 텍스트 정제 (페이지별 처리)
        """
        print("\n[Step 2] 전처리 시작")
        
        patterns = [
            r'\n\s*\d{1,3}\s*\n',
            r'\n\s*-\s*\d{1,3}\s*-\s*\n',
            r'\n\s*현대건설\s+ESG\s+보고서\s*\n',
            r'\n\s*HYUNDAI\s+E&C\s*\n',
            r'\n\s*www\.hdec\.kr\s*\n',
            r'\n\s*(?:DL\s+Construction\s+)?Sustainability\s+Report\s+\d{4}\s*\n',
            r'Contents\s*\n',
            r'\n{3,}',
        ]
        
        cleaned_docs = []
        total_reduction = 0
        
        for doc in documents:
            text = doc.page_content
            cleaned_text = text
            
            for pattern in patterns:
                cleaned_text = re.sub(pattern, '\n\n', cleaned_text, flags=re.IGNORECASE)
            
            cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
            cleaned_text = re.sub(r'\n\s+\n', '\n\n', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            total_reduction += len(text) - len(cleaned_text)
            
            # 내용이 너무 짧으면(빈 페이지 등) 스킵할 수도 있음
            if len(cleaned_text) > 10:
                cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
        
        print(f"✓ 전처리 완료: {total_reduction} 글자 제거")
        return cleaned_docs
    
    # ==================== Step 3: Chunking ====================
    def semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """
        의미 단위 청킹 (페이지별 처리 및 메타데이터 보존)
        """
        print("\n[Step 3] 청킹 시작")
        
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        final_chunks = []
        
        for doc in documents:
            # 1. 헤더 기반 분할
            header_splits = markdown_splitter.split_text(doc.page_content)
            
            # 2. 메타데이터 병합 (페이지 정보 + 헤더 정보)
            for split in header_splits:
                split.metadata.update(doc.metadata)
            
            # 3. 토큰 제한 기반 분할
            chunks = text_splitter.split_documents(header_splits)
            
            # 4. 너무 짧은 청크 제거 (헤더 잔여물 등)
            chunks = [c for c in chunks if len(c.page_content) > 50]
            
            final_chunks.extend(chunks)
            
        print(f"✓ 청킹 완료: {len(final_chunks)}개 청크 생성")
        return final_chunks
    
    # ==================== Step 4 & 5: Embedding & Indexing ====================
    def build_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        벡터 스토어 구축
        - 한국어 특화 임베딩 생성
        - ChromaDB에 저장
        
        Args:
            chunks: Document 청크 리스트
            
        Returns:
            ChromaDB 벡터 스토어
        """
        print("\n[Step 4 & 5] 임베딩 생성 및 벡터 스토어 구축 시작")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✓ 벡터 스토어 구축 완료: {len(chunks)}개 문서 저장됨")
        print(f"  - 저장 경로: {self.persist_directory}")
        
        return self.vectorstore
    
    # ==================== Full Pipeline ====================
    def process_pdf(self, pdf_path: str, use_cache: bool = True) -> Chroma:
        """
        전체 파이프라인 실행

        Args:
            pdf_path: PDF 파일 경로
            use_cache: True일 경우 캐시된 파싱 결과 사용 (LlamaParse 한도 절약)

        Returns:
            ChromaDB 벡터 스토어
        """
        print("="*60)
        print("ESG RAG 파이프라인 시작")
        print("="*60)

        # Step 1: Parsing (캐시 옵션 전달)
        documents = self.parse_pdf(pdf_path, use_cache=use_cache)

        # Step 2: Preprocessing
        cleaned_docs = self.preprocess_text(documents)
        
        # 회사명 메타데이터 추가
        company_name = Path(pdf_path).stem.split('_')[0] # 파일명 규칙 가정: "현대건설_ESG..." -> "현대건설"
        for doc in cleaned_docs:
            doc.metadata['company'] = company_name
            doc.metadata['type'] = 'ESG_Report' # 기본값

        # Step 3: Chunking
        chunks = self.semantic_chunking(cleaned_docs)
        
        if not chunks:
            print("❌ Error: No chunks created. Check parsing or preprocessing.")
            return None

        # Step 4 & 5: Embedding & Indexing
        vectorstore = self.build_vectorstore(chunks)

        print("\n" + "="*60)
        print("✓ 파이프라인 완료!")
        print("="*60)

        return vectorstore
    
    # ==================== Retrieval ====================
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        유사도 기반 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            
        Returns:
            관련 문서 리스트
        """
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다. process_pdf()를 먼저 실행하세요.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        유사도 점수와 함께 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            
        Returns:
            (Document, score) 튜플 리스트
        """
        if self.vectorstore is None:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results


# ==================== 사용 예시 ====================
if __name__ == "__main__":
    # 설정

    llama_cloud_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    upstage_api_key: Optional[str] = os.getenv("UPSTAGE_API_KEY")


    LLAMA_CLOUD_API_KEY = llama_cloud_api_key
    PDF_PATH = "pdf_folder/현대건설_ESG보고서.pdf"
    
    # 파이프라인 초기화
    pipeline = ESGRAGPipeline(
        llama_cloud_api_key=LLAMA_CLOUD_API_KEY,
        embedding_model="BAAI/bge-m3",  # 한국어 특화
        chunk_size=800,
        chunk_overlap=200,
        persist_directory="./chroma_db_esg"
    )
    
    # 전체 파이프라인 실행
    vectorstore = pipeline.process_pdf(PDF_PATH)
    
    # 검색 테스트
    print("\n\n" + "="*60)
    print("검색 테스트")
    print("="*60)
    
    test_queries = [
        "현대건설의 탄소배출 감축 목표는?",
        "환경 관련 투자 현황은?",
        "안전사고 발생 건수는?"
    ]
    
    for query in test_queries:
        print(f"\n질의: {query}")
        print("-" * 60)
        
        results = pipeline.search_with_score(query, k=3)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[결과 {i}] 유사도: {score:.4f}")
            print(f"내용 미리보기: {doc.page_content[:200]}...")
            if doc.metadata:
                print(f"메타데이터: {doc.metadata}")
