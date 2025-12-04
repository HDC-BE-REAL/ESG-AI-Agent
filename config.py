"""
설정 파일 - 프로젝트 전체 설정 관리
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    
    # ===== API Keys =====
    llama_cloud_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY", "llx-0vWnOp0um3JoJgrk4akCe4ThGrwwotFpoILfzcCS0WIJTO8K")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    upstage_api_key: Optional[str] = os.getenv("UPSTAGE_API_KEY")
    
    # ===== 임베딩 설정 =====
    # 추천: "BAAI/bge-m3" (한국어 특화, 무료)
    # 대안: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"  # "cpu" or "cuda"
    normalize_embeddings: bool = True
    
    # ===== 청킹 설정 =====
    chunk_size: int = 800  # 토큰 수
    chunk_overlap: int = 200  # 오버랩 토큰 수
    
    # 권장 설정:
    # - 짧은 Q&A: chunk_size=500, overlap=100
    # - 긴 분석: chunk_size=1200, overlap=300
    # - 표 위주: chunk_size=600, overlap=150
    
    # ===== 벡터 DB 설정 =====
    persist_directory: str = "./chroma_db"
    collection_name: str = "esg_documents"
    
    # ===== LlamaParse 설정 =====
    llamaparse_result_type: str = "markdown"  # "markdown" or "text"
    llamaparse_language: str = "ko"  # 한국어
    llamaparse_verbose: bool = True
    
    # ===== 검색 설정 =====
    default_k: int = 5  # 반환할 문서 개수
    search_type: str = "similarity"  # "similarity" or "mmr"
    
    # MMR (Maximal Marginal Relevance) 설정
    # - 다양성을 높이고 중복을 줄임
    mmr_diversity_score: float = 0.3  # 0~1 (1에 가까울수록 다양성 우선)
    
    # ===== LLM 설정 (답변 생성용) =====
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4o"  # or "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.0  # 0=결정적, 1=창의적
    llm_max_tokens: Optional[int] = None  # None=자동
    
    # ===== 전처리 설정 =====
    remove_headers_footers: bool = True
    remove_page_numbers: bool = True
    remove_urls: bool = True
    
    # 커스텀 노이즈 패턴 (Regex)
    custom_noise_patterns: list = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.custom_noise_patterns is None:
            self.custom_noise_patterns = []


# ===== 사전 정의된 설정 프로필 =====

def get_config_fast():
    """빠른 처리 우선 (작은 청크, 적은 검색)"""
    return PipelineConfig(
        chunk_size=500,
        chunk_overlap=100,
        default_k=3,
        embedding_device="cpu"
    )


def get_config_accurate():
    """정확도 우선 (큰 청크, 많은 검색)"""
    return PipelineConfig(
        chunk_size=1200,
        chunk_overlap=300,
        default_k=10,
        search_type="mmr",
        mmr_diversity_score=0.5
    )


def get_config_balanced():
    """균형 잡힌 설정 (기본값)"""
    return PipelineConfig(
        chunk_size=800,
        chunk_overlap=200,
        default_k=5
    )


def get_config_gpu():
    """GPU 가속 설정"""
    return PipelineConfig(
        chunk_size=800,
        chunk_overlap=200,
        embedding_device="cuda",
        default_k=5
    )


# ===== 도메인별 특화 설정 =====

def get_config_for_tables():
    """표 위주 문서 (ESG 보고서 등)"""
    return PipelineConfig(
        chunk_size=600,  # 표가 잘리지 않도록 적당한 크기
        chunk_overlap=150,
        default_k=5,
        custom_noise_patterns=[
            r'\n\s*\d{1,3}\s*\n',  # 페이지 번호
            r'\|\s*-+\s*\|',       # Markdown 표 구분선 중복 제거
        ]
    )


def get_config_for_financial():
    """금융 보고서"""
    return PipelineConfig(
        chunk_size=1000,  # 재무제표는 컨텍스트가 중요
        chunk_overlap=250,
        default_k=7,
        search_type="mmr",
        mmr_diversity_score=0.4
    )


def get_config_for_technical():
    """기술 문서"""
    return PipelineConfig(
        chunk_size=900,
        chunk_overlap=200,
        default_k=5,
        embedding_model="BAAI/bge-m3"  # 전문용어 처리 우수
    )


# ===== 사용 예시 =====
if __name__ == "__main__":
    # 기본 설정
    config = PipelineConfig()
    print("기본 설정:")
    print(f"  - 청크 크기: {config.chunk_size}")
    print(f"  - 임베딩 모델: {config.embedding_model}")
    
    # 사전 정의 프로필
    config_fast = get_config_fast()
    print("\n빠른 처리 프로필:")
    print(f"  - 청크 크기: {config_fast.chunk_size}")
    print(f"  - 검색 개수: {config_fast.default_k}")
    
    # 도메인 특화
    config_table = get_config_for_tables()
    print("\n표 특화 프로필:")
    print(f"  - 청크 크기: {config_table.chunk_size}")
    print(f"  - 노이즈 패턴: {len(config_table.custom_noise_patterns)}개")
