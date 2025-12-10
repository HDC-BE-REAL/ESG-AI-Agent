# ESG AI Agent 전체 아키텍처

## 백엔드 구조 (FastAPI)
- `backend/main.py`: FastAPI 앱, CORS 설정, `/static` 서빙.
- `backend/api.py`: 파일 업로드, `/api/agent/{type}`, `/api/chat`, `/api/chat/stream` 엔드포인트.
- `backend/manager.py`: Policy/Regulation/Risk/Report 에이전트 실행 및 Redis 기반 컨텍스트 관리, `run_custom_agent`는 LangGraph 파이프라인을 호출해 네 모듈을 한 번에 실행한다.
- `src/tools/`: policy/risk/regulation/report LangChain 모듈, 각자 VectorDB + LLM 체인을 보유.
- `vector_db/`: PDF → chunk → 임베딩(BGE) → Chroma 저장 스크립트(`vector_db/esg_all.py`).

## 프런트엔드 구조 (React/Vite)
- `/frontend/src/App.jsx`: 좌측 토글 사이드바 + 중앙 보고서/체크리스트 + 우측 챗봇 패널.
- `Sidebar`: 챗봇 기록/파일 업로드 UI.
- `MainContent`: LLM이 생성한 보고서/체크리스트 표시, 검색·스크롤 가능.
- `ChatBotPanel`: 단일 챗봇 UI. `/api/chat/stream`과 연동해 스트리밍 응답을 처리.
- `react-markdown`으로 챗봇 답변 마크다운 렌더링.

## 데이터 흐름
1. 사용자 업로드 → `/api/upload` → Redis context에 파일 목록 반영.
2. 특정 에이전트 요청(`/api/agent/*`) → LangChain 툴 실행 → 결과를 Redis context에 저장.
3. `/api/chat` 또는 `/api/chat/stream` → 자동으로 custom 에이전트 실행 → 정책/규제/리스크/보고서를 컨텍스트에 추가 → LLM 프롬프트 생성 → 응답/스트리밍.
4. 프런트는 `/api/chat/stream` 결과를 실시간 렌더링해 ChatGPT 스타일 대화를 제공.
