# Advanced ESG RAG Pipeline (8-Step Workflow)

복잡한 표와 레이아웃을 가진 **한국어 ESG 보고서**를 위한 고성능 RAG 시스템입니다.
LlamaParse로 문서를 정밀하게 파싱하고, 8단계의 고급 검색 파이프라인을 통해 최적의 답변을 생성합니다.

## 🚀 주요 기능: 8-Step RAG Pipeline

이 프로젝트는 단순한 검색을 넘어, 다음과 같은 **8단계 고급 워크플로우**를 수행합니다:

1.  **Query Input**: 사용자 질문 입력
2.  **Query Processing** (LLM):
    *   **Query Rewriting**: 검색에 최적화된 형태로 질문 재작성
    *   **Entity Extraction**: 회사명, 연도 등 핵심 엔티티 추출
    *   **Intent Detection**: 질문 의도 파악
3.  **Pre-filtering**: 추출된 엔티티(회사명 등)로 메타데이터 필터링 적용
4.  **Vector Search**: 1차 후보 문서 검색 (Similarity / MMR)
5.  **Re-ranking** (Cross-Encoder): 검색된 문서들의 관련성 점수 재산정 (정밀도 향상)
6.  **Post-filtering**: 재순위화 점수 기반 하위 문서 제거
7.  **Top-k Selection**: 최종 상위 문서 선정
8.  **LLM Generation**: 최종 답변 생성 (GPT-4o 등)

---

## 📂 프로젝트 구조

```
esq_party/
├── 📄 .env                  # API 키 설정 (OpenAI, LlamaCloud)
├── 📄 config.py             # 파이프라인 전체 설정 (모델, 청크 크기 등)
│
├── 🧱 **Core Modules**
│   ├── esg_rag_pipeline.py  # [Ingestion] PDF 파싱, 노이즈 제거, 청킹, DB 적재
│   ├── advanced_retriever.py# [Retrieval] 8단계 검색 파이프라인 구현
│   ├── query_processor.py   # [Analysis] 쿼리 분석 및 엔티티 추출
│   └── reranker.py          # [Ranking] Cross-Encoder 기반 재순위화
│
├── 🚀 **Execution Scripts**
│   ├── demo_workflow.py     # [메인] 8단계 전체 워크플로우 실행 (질문 -> 답변)
│   ├── search_only.py       # [검색] 단순 문서 검색 (질문 -> 문서 내용 확인)
│   └── demo_retriever.py    # [관리] 데이터 적재(--reingest) 및 테스트
│
└── 💾 **Data**
    ├── pdf_folder/          # 분석할 PDF 파일 위치
    ├── chroma_db/           # 벡터 데이터베이스 (자동 생성)
    └── parsed_cache/        # 파싱된 텍스트 캐시 (비용 절약)
```

---

## 📦 설치 및 설정

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정
`.env` 파일을 생성하고 API 키를 입력하세요.
```bash
cp .env.template .env
```
```ini
# .env 파일 내용
LLAMA_CLOUD_API_KEY=llx-...  # PDF 파싱용 (필수)
OPENAI_API_KEY=sk-...        # 쿼리 분석 및 답변 생성용 (필수)
```

---

## 🏃‍♂️ 실행 방법

### 1. 데이터 적재 (Ingestion)
PDF 파일을 `pdf_folder/`에 넣고 아래 명령어를 실행하여 벡터 DB를 구축합니다.
```bash
python demo_retriever.py --reingest
```
*   최초 1회 실행 필요. PDF가 변경되면 다시 실행하세요.

### 2. 전체 파이프라인 실행 (추천)
질문을 입력하면 **분석 -> 검색 -> 재순위화 -> 답변 생성**까지 모든 과정을 보여줍니다.
```bash
python demo_workflow.py
```

### 3. 단순 검색 확인
답변 생성 없이, 어떤 문서가 검색되는지만 빠르게 확인하고 싶을 때 사용합니다.
```bash
python search_only.py
```

---

## 🛠️ 기술 스택

*   **PDF Parsing**: LlamaParse (표/레이아웃 보존)
*   **Embedding**: BAAI/bge-m3 (한국어 특화)
*   **Vector DB**: ChromaDB
*   **Re-ranking**: BAAI/bge-reranker-v2-m3 (Cross-Encoder)
*   **LLM**: OpenAI GPT-4o / GPT-3.5-turbo
*   **Framework**: LangChain, LlamaIndex

---

## 💡 주요 특징 상세

*   **노이즈 제거**: 정규식(Regex)을 사용하여 반복되는 헤더/푸터를 제거하고, 의미 없는 짧은 청크를 필터링합니다.
*   **메타데이터 보존**: 페이지 번호, 회사명 등을 메타데이터로 저장하여 출처를 명확히 하고 필터링에 활용합니다.
*   **캐싱 시스템**: LlamaParse 결과와 임베딩을 캐싱하여 API 비용과 시간을 절약합니다.
