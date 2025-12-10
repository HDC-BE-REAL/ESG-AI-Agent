# Retrieval & LLM 아키텍처 정리

## 전체 흐름
1. **문서 임베딩/벡터 DB 구축**
   - `vector_db/` 디렉터리(예: `vector_db/esg_all`)에는 ESG 보고서·규제 문서·협력사 자료를 임베딩해 놓은 Chroma 컬렉션이 저장된다.
   - `vector_db/esg_all.py` 또는 `retriever/` 내 파이프라인 스크립트가 PDF/OCR → 임베딩 → `persist_directory` 저장을 담당한다.

2. **툴별 RAG 체인**
   - 각 툴(`policy_tool.py`, `regulation_tool.py`, `risk/…`, `supplier_eval.py` 등)이 동일한 벡터 DB를 직접 로드하지만, **retriever 설정·프롬프트·LLM 파라미터는 서로 다르다.**
   - 예)
| 모듈 | 벡터 DB | Retriever/LLM 특성 |
| --- | --- | --- |
| `policy_tool.py` | `Chroma(persist_directory="vector_db/esg_all")` | 1) `vector_db/esg_all.py` 등으로 PDF를 chunk → BGE 임베딩 후 저장<br>2) 사용자 질의를 동일 모델로 벡터화해 `retriever.get_relevant_documents(k=5)`로 상위 문단을 가져옴<br>3) “정책 요약/비교/평가” 프롬프트에 `[관련 근거]` 블록 형태로 삽입해 LLM(GPT-4o mini)을 호출하는 정석 RAG 체인. |
| `regulation_tool.py` | `Chroma(collection_name="esg_regulations", persist_directory=vector_db/all_esg)` | Selenium+Tavily로 최신 문서를 수집/저장 후, 필요할 때만 RAG로 규제 요약. 동일한 BGE 임베딩이지만 크롤러 히스토리, 검색 범위, 스케줄링 로직이 다름. |
| `risk`/`supplier_eval` | 템플릿 기반 점수/보고서 | RAG 의존도가 낮지만, 필요 시 다문서 증거를 vector DB에서 가져와 항목별 근거를 생성. |

3. **FastAPI & 에이전트**
   - `backend/api.py`에서 `/api/agent/*` 요청을 받으면 `AgentManager`가 해당 툴을 실행한다.
   - 실행 과정에서 각 툴은 자신의 retriever/LLM 체인을 활용해 결과를 만들고, JSON/CSV/리포트 형태로 응답한다.

## `retriever/` 폴더 용도
- 공통 RAG 파이프라인(`retriever_pipeline.py` 등)을 넣어 둔 곳으로, **벡터 DB 구축/업데이트**에 재사용된다.
- 런타임에는 각 툴이 직접 Chroma를 로드하지만, 대규모 문서 적재나 새로운 컬렉션 생성 시 `retriever/` 코드를 활용해 쉽게 파이프라인을 돌릴 수 있다.

## 정확도 및 확장성
- 동일한 벡터 컬렉션을 공유하더라도, 툴마다 `k` 값, 스코어 함수, 프롬프트, 후처리 로직이 달라 **용도별 최적화**가 가능하다.
- 추후 정확도 개선이 필요하면 `retriever/` 폴더의 파이프라인을 활용해 인덱스를 다시 만들거나, 각 툴의 retriever 파라미터를 조정하면 된다.
