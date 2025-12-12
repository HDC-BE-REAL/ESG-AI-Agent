# 📘 ESG Insight Agent – 프로젝트 기획서 

본 문서는 기존 프로젝트 기획서에 최신 아키텍처(Frontend–Backend–LangGraph–RAG–VectorDB)를 반영하여  
최종적으로 정리된 프로젝트 기획서이다.  
전략적 관점(비즈니스 효과)과 기술적 관점(구조·설계)을 결합한 **혼합형 문서**이다.

---

# 1. 프로젝트 정의

## 🎯 프로젝트 목표

ESG Insight Agent는 건설사를 위한 **ESG 문서 분석·리스크 진단·보고서 생성 자동화 시스템**이다.  
K-ESG, SASB, GRI 등 공개 ESG 문서들을 기반으로 **ESG 평가의 속도와 정확성을 혁신적으로 개선하는 AI Agent 플랫폼**을 구축한다.

주요 목표:

- ESG 문서 자동 분석 및 비교
- 프로젝트 기반 리스크 진단 자동화
- K-ESG 기반 ESG 보고서 자동 생성
- 규제 변경 실시간 모니터링
- React 기반 대시보드 제공
- LangGraph 기반 멀티 에이전트 구조 완성

---

# 2. 프로젝트 주요 내용

## 📅 프로젝트 기간  
**2025-12-02 ~ 2025-12-11 (총 10일)**

## 👥 참여 인원  
- 박희선  
- 석상훈  
- 윤현이  
- 황정민  

## 📚 데이터 출처

- K-ESG 공급망 대응 가이드라인  
- K-ESG 61개 평가 항목  
- SASB Engineering & Construction Standards  
- GRI Standards  
- ISO 31000 위험관리 지침  
- 고용부/환경부 규제 문서  
- 주요 건설사 ESG 보고서  

---

# 3. 일정 계획

아래 일정은 기존 계획에 최신 아키텍처 구성 작업을 반영한 최종 구조이다.

| 작업 항목 | 시작 날짜 | 종료 날짜 | 기간(일) |
|-----------|------------|------------|-----------|
| 프로젝트 정의 및 계획 수립 | 12/02 | 12/02 | 1 |
| 문서 조사 및 데이터 수집 | 12/02 | 12/03 | 2 |
| 전처리·임베딩·Vector DB 구축 | 12/03 | 12/05 | 3 |
| LangGraph 기반 정책·리스크 모듈 개발 | 12/05 | 12/07 | 3 |
| 보고서·규제 모듈 개발 | 12/07 | 12/08 | 2 |
| React UI 구축 | 12/07 | 12/08 | 2 |
| 통합 테스트 및 검수 | 12/08 | 12/10 | 3 |
| 발표 및 인수 | 12/11 | 12/11 | 1 |

---

# 4. 작업 분할 구조 (WBS)

## **1. 데이터 요구사항 분석**
- 1.1 ESG 문서 정의  
- 1.2 PDF 구조 및 OCR 필요성 검토  
- 1.3 메타데이터 정규화 기준 설계  

## **2. 데이터 구축**
- 2.1 PDF → 텍스트 변환  
- 2.2 OCR 처리  
- 2.3 텍스트 청크 분할  
- 2.4 bge-m3 기반 임베딩 생성  
- 2.5 Chroma Vector DB 구축  

## **3. 분석 모듈 구축**
- 3.1 정책 요약·비교 LLM 체인  
- 3.2 프로젝트 리스크 진단 체인  
- 3.3 보고서 자동 생성  
- 3.4 규제 크롤링 및 모니터링  

## **4. 시스템 개발**
- 4.1 FastAPI 백엔드  
- 4.2 React 프론트엔드  
- 4.3 LangGraph 기반 멀티 Agent Workflow  
- 4.4 RAG 엔진 구성(Retriever + Scoring + Filtering)  

## **5. 테스트 및 배포**
- 5.1 에이전트 기능 테스트  
- 5.2 UI/UX 개선  
- 5.3 통합 테스트  
- 5.4 최종 발표자료 준비  

---

# 5. 시스템 아키텍처 (최신)

아래는 실제 GitHub 폴더 구조 기반으로 재정리된 최종 아키텍처이다.

```
React Frontend
        ↓
FastAPI Backend
        ↓
LangGraph Workflow Engine
    ├ 정책 Agent (policy_tool)
    ├ 리스크 Agent (risk_tool)
    ├ 보고서 Agent (report_tool)
    └ 규제 Agent (regulation_tool)
        ↓
Retriever Pipeline (RAG)
        ↓
Chroma Vector DB
        ↓
Knowledge Base (PDF/JSON/XLSX/템플릿)
```

---

# 6. 요구사항 정의

## 6.1 기능 요구사항  
- ESG 문서 자동 전처리  
- Vector DB 기반 검색 & 요약  
- K-ESG / SASB / GRI 정책 비교  
- 프로젝트 기반 리스크 진단  
- PDF/MD 보고서 자동 생성  
- 규제 변경 감지 및 알림  
- React UI에서 결과 제공  

## 6.2 비기능 요구사항  
- 정확도 ≥ 85%  
- 응답 속도 ≤ 10초  
- 스케일 확장 지원  
- 예외 처리 및 안정성 확보  

---

# 7. 구성요소 상세

## 🔹 프론트엔드 (React + Vite + Tailwind)
위치: `/frontend/`

역할:
- ESG 질문/보고서 UI
- FastAPI와 API 통신
- 결과 시각화

---

## 🔹 백엔드 (FastAPI)
위치: `/backend/`

핵심 파일:
- `main.py`  
- `api.py`  
- `manager.py`  
- `kv_store.py`  

역할:
- 요청 수신 → LangGraph 실행 → 응답 반환  

---

## 🔹 LangGraph 멀티 에이전트
위치: `/src/workflows/`

- `graph.py`  
- `custom_graph.py`  

각 에이전트는 Tool Layer와 연결됨.

---

## 🔹 Tool Layer
위치: `/src/tools/`

- `policy_tool.py`  
- `risk_tool.py`  
- `risk_crawling_tool.py`  
- `regulation_tool.py`  
- `report_tool/`  

구조:
- 하나의 Tool = 하나의 ESG 기능 도메인

---

## 🔹 Retriever Layer
위치: `/retriever/`

- 질의 검증  
- 문서 검색  
- RAG 품질 개선  

---

## 🔹 Vector DB
위치: `/vector_db/`

- `esg_all` 컬렉션  
- `chroma.sqlite3`  

---

## 🔹 Knowledge Base
위치: `/data/`

포함 데이터:
- ISO31000 taxonomy  
- GRI Index  
- 공급사 평가 템플릿  
- ESG 관련 JSON  

---

# 8. 시각 자료

- 📌 WBS 다이어그램: `/images/wbs_diagram.png`  
- 📌 시스템 아키텍처: `/images/system_architecture.png`  
- 📌 RAG 파이프라인: `/images/rag_pipeline.png`  

---

# 9. 기대효과

- ESG 문서 분석 시간 **70% 감소**  
- 정책 비교의 객관성 향상  
- 현장 ESG 리스크 진단 자동화  
- 보고서 작성 시간 대폭 절감  
- 규제 변경 자동화로 리스크 최소화  

---

# 10. 결론

ESG Insight Agent는 건설사의 ESG 대응을 자동화·고도화하는 핵심 AI 시스템이다.  
본 기획서는 실제 기술 구조와 프로젝트 방향을 완전하게 반영하며  
설계서·중간 점검표와 함께 프로젝트 문서 기반이 된다.

---

_작성자:  Be-Real 팀_  
_작성일: 2025-12-11_
