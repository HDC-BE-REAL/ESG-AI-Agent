# ESG AI Agent

LangChain 기반 4가지 ESG 목적 Tool을 LangGraph 에이전트로 묶은 프로토타입입니다.

## 폴더 구조

```
ESG_AIagent/
├─ data/
├─ retriever/
├─ vector_db/
├─ src/
│  ├─ app.py                  # LangGraph 워크플로 진입점
│  ├─ tools/                  # 목적별 LangChain Tool
│  │  ├─ policy_tool.py - 정민
│  │  ├─ risk_tool.py - 현이
│  │  ├─ report_tool.py - 상훈  
│  │  └─ regulation_tool.py - 희선
│  └─ workflows/
│     ├─ graph.py             # detect_mode → execute_tool → generate_final_answer → END
│     └─ schema.py
└─ requirements.txt
```

## LangGraph 워크플로
1. `detect_mode`: 사용자 질문의 키워드를 분석하여 4개 Tool 중 하나를 선택
2. `execute_tool`: 선택된 Tool에 원문 질의를 그대로 전달하고 결과를 획득
3. `generate_final_answer`: 선택 정보와 Tool 응답을 묶어 최종 답변 생성

```
User Query → detect_mode → execute_tool → generate_final_answer → END
```

## 실행 방법
```bash
pip install -r requirements.txt
python -m src.app "ESG 보고서 자동 작성해줘"
```

## 확장 아이디어
- 실제 문서/규제 데이터 연결을 위해 Retriever 및 Vector DB 연동
- detect_mode에 LLM 분류기 또는 RAG 스코어 활용
- Tool 내부 로직을 외부 API 또는 사내 지식그래프와 연동
