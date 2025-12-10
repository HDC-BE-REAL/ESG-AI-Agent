from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os

from backend.manager import agent_manager

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    query: str
    agent_type: Optional[str] = "general"

class AgentRequest(BaseModel):
    query: str
    focus_area: Optional[str] = None  # 리스크 도구 등에서 안전/환경 등 영역을 지정할 때 사용
    audience: Optional[str] = None  # 보고서 초안 대상 (경영진, 이사회 등)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Update shared context (중복 제거 + 최대 50개 유지)
        current_files = agent_manager.get_context().get("uploaded_files", [])
        filtered = [entry for entry in current_files if entry.get("filename") != file.filename]
        relative_path = f"/static/uploads/{file.filename}"
        filtered.append({"filename": file.filename, "path": relative_path})
        if len(filtered) > 50:
            filtered = filtered[-50:]
        agent_manager.update_context("uploaded_files", filtered)
        
        return {"filename": file.filename, "status": "uploaded", "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
async def get_context():
    return agent_manager.get_context()

@router.post("/agent/regulation")
async def run_regulation_agent(request: AgentRequest):
    result = await agent_manager.run_regulation_agent(request.query)
    return {"result": result}

@router.post("/agent/{agent_type}")
async def run_agent(agent_type: str, request: AgentRequest):
    if agent_type == "policy":
        result = await agent_manager.run_policy_agent(request.query)
    elif agent_type == "risk":
        result = await agent_manager.run_risk_agent(request.query, request.focus_area)
    elif agent_type == "report":
        result = await agent_manager.run_report_agent(request.query, request.audience)
    elif agent_type == "custom":
        result = await agent_manager.run_custom_agent(
            request.query,
            focus_area=request.focus_area,
            audience=request.audience,
        )
    else:
        raise HTTPException(status_code=404, detail="Agent type not found")

    return {"result": result}

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Retrieve Shared Context
        context = agent_manager.get_context()

        # 1-1. 자동으로 policy/regulation/risk/report 실행 (custom 오케스트레이터)
        custom_result = await agent_manager.run_custom_agent(request.query)

        # 2. Construct System Prompt
        risk_assessment = context.get('risk_assessment')
        risk_summary = str(risk_assessment)[:500] + "..." if risk_assessment else "None"
        system_prompt = f"""
        You are an expert ESG AI Assistant. Your goal is to help the user with ESG (Environmental, Social, and Governance) related tasks.

        [Current Context]
        - Uploaded Files: {[f['filename'] for f in context.get('uploaded_files', [])]}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}
        
        [Instructions]
        - Answer the user's question based on the context provided above.
        - If the user asks about specific regulations or news, refer to the 'Latest Regulation Updates' section.
        - Be professional, concise, and helpful.
        - If you don't know the answer, admit it and suggest running a specific agent (Regulation, Policy, Risk, etc.).
        - Language: Korean (unless the user asks in English).
        """
        
        # 3. Call LLM (GPT-4o)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]

        response_msg = await llm.ainvoke(messages)
        response_text = response_msg.content
        
        #4. Update Chat History (Optional, for future context)
        current_history = context.get("chat_history", [])
        current_history.append({"role": "user", "content": request.query})
        current_history.append({"role": "assistant", "content": response_text})
        agent_manager.update_context("chat_history", current_history)
        
        return {"response": response_text}
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    try:
        context = agent_manager.get_context()
        custom_result = await agent_manager.run_custom_agent(request.query)
        risk_assessment = context.get('risk_assessment')
        risk_summary = str(risk_assessment)[:500] + "..." if risk_assessment else "None"
        system_prompt = f"""
        You are an expert ESG AI Assistant. Your goal is to help the user with ESG (Environmental, Social, and Governance) related tasks.

        [Current Context]
        - Uploaded Files: {[f['filename'] for f in context.get('uploaded_files', [])]}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}

        [Auto-Generated Insights]
        - Policy Summary: {custom_result.get('policy')}
        - Regulation Update: {custom_result.get('regulation')}
        - Risk Analysis: {custom_result.get('risk')}
        - Report Draft: {custom_result.get('report')}

        [Instructions]
        - Answer using the template below to emulate an expert ESG consultant.
        - Template:
            1) 요약 (2-3문장)
            2) 근거 (bullet로 최대 3개, 각 항목 앞에 •)
            3) 권고/다음 단계 (bullet)
        - 답변에 최신 규제/정책/리스크 정보를 자연스럽게 녹여라.
        - Be professional, concise, and helpful.
        - If you don't know the answer, admit it and suggest running a specific agent (Regulation, Policy, Risk, etc.).
        - Language: Korean (unless the user asks in English).
        """

        llm = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]

        async def event_generator():
            try:
                async for chunk in llm.astream(messages):
                    token = chunk.content or ""
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
