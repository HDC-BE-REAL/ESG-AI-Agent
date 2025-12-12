
# 🌍 ESG Insight Agent
AI-powered ESG Policy Analysis, Risk Diagnostics & Automated Reporting System

<div align="center">
  <img src="./images/banner.png" width="80%" alt="ESG Insight Agent Banner"/>
</div>

---

## 📌 Overview
**ESG Insight Agent**는 건설사를 위한 ESG 업무 자동화 AI 시스템으로,  
정책 분석 → 리스크 진단 → 보고서 생성 → 규제 모니터링을 통합적으로 수행합니다.

---

## 🧭 프로젝트 기간
2025-12-02 ~ 2025-12-11

---

## 👥 참여 인원
- 팀원: 박희선, 석상훈, 윤현이, 황정민

---

## 📚 데이터 출처
- K-ESG 공급망 대응 가이드라인
- SASB E&C Standards
- GRI Standards
- 국내 법령 자료
- 주요 건설사 ESG 보고서

---

# 🎯 목표
- ESG 문서 자동 요약 및 비교
- 프로젝트 기반 ESG 리스크 평가
- ESG 보고서 자동 생성(PDF/DOCX)
- 규제 변경 감지 자동화
- RAG 기반 문서 검색 및 분석

---

# 🧩 ESG Reasoning Modules

### ✔ 정책 분석 모듈
- K-ESG / SASB / GRI 자동 비교
- 정책 간 차이 분석 및 가이드 제공

### ✔ 리스크 진단 모듈
- 프로젝트 기반 E/S/G 리스크 자동 평가
- 체크리스트 생성 기능

### ✔ 보고서 자동 생성
- K-ESG 61개 항목 기반
- PDF/DOCX 자동 출력

### ✔ 규제 모니터링
- 기관별 규제 변경 감지
- 주간 리포트 생성


---


# 🔌 3. API Specification (현재 구현 기준)

| 메서드 | 경로 | 설명 |
|--------|------|--------|
| POST | /upload | 파일 업로드 및 컨텍스트 공유 |
| GET  | /context | 현재 공유 컨텍스트 조회 |
| POST | /agent/{agent_type} | policy / regulation / risk / report / custom 에이전트 실행 |
| POST | /chat | 비스트리밍 챗 응답 (컨텍스트·오케스트레이션 포함) |
| POST | /chat/stream | SSE 스트리밍 챗 응답 |

**기능 매핑 (agent_type별 동작)**
- `policy`: ESG 문서 요약/비교
- `regulation`: 규제 변경 감지/요약
- `risk`: 프로젝트 기반 리스크 분석
- `report`: ESG 보고서 초안 생성
- `custom`: 네 모듈을 동시에 실행해 컨텍스트 업데이트


---

# 🏗️ System Architecture

<div align="center">
  <img src="./images/system_architecture.png" width="80%" alt="System Architecture"/>
  <img src="./images/system_architecture1.png" width="80%" alt="System Architecture 1"/>
</div>

---

# 🔍 RAG Pipeline

<div align="center">
  <img src="./images/rag_pipeline.png" width="80%" alt="RAG Pipeline"/>
  <img src="./images/rag_pipeline1.png" width="80%" alt="RAG Pipeline 1"/>
</div>

---

# 🗂 WBS Diagram

<div align="center">
  <img src="./images/wbs_diagram.png" width="80%" alt="WBS Diagram"/>
</div>

---

# 📅 일정 계획 (Gantt Chart)

| 작업 항목 | 시작 | 종료 | 기간(일) |
|---|---|---|---|
| 프로젝트 정의 및 계획 수립 | 2025-12-02 | 2025-12-02 | 1 |
| 문서 수집 | 2025-12-02 | 2025-12-03 | 2 |
| 데이터 전처리 및 임베딩 | 2025-12-03 | 2025-12-04 | 2 |
| RAG 아키텍처 구성 | 2025-12-04 | 2025-12-05 | 2 |
| 핵심 모듈 개발 | 2025-12-05 | 2025-12-07 | 3 |
| UI 및 모니터링 기능 개발 | 2025-12-07 | 2025-12-08 | 2 |
| 통합 테스트/안정화 | 2025-12-08 | 2025-12-10 | 3 |
| 발표/인수 | 2025-12-11 | 2025-12-11 | 1 |

---

# 🧪 기술 스택

| 영역 | 기술 |
|---|---|
| Backend | FastAPI, Python |
| AI Engine | GPT-4o, LangChain, LangGraph, ChatOpenAI |
| Embedding | bge-m3 |
| DB | ChromaDB |
| Parsing | PyPDF, PyMuPDF, Tesseract |
| Frontend | React |

---

# 📁 문서
더 자세한 문서는 `docs/` 폴더에서 확인하세요.

---

# ▶ 실행 방법

### 1) 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2) 전체실행 방법
```bash
./run_app.sh
```

---
# 📚 문서 목록

- 📘 기획서: `docs/project_plan.md`
- 🛠 설계서: `README.md`
- 📊 상태 점검표: `docs/mid_review.md`

---

# 📄 License
MIT License
