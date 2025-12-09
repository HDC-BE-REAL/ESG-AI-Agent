from fastapi import APIRouter, UploadFile, File, HTTPException, Body
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
        # 안전/환경/노동/거버넌스 등 특정 영역을 focus_area로 전달 가능
        result = await agent_manager.run_risk_agent(request.query, request.focus_area)
    elif agent_type == "report":
        result = await agent_manager.run_report_agent(request.query)
    elif agent_type == "custom":
        result = await agent_manager.run_custom_agent(request.query)
    else:
        raise HTTPException(status_code=404, detail="Agent type not found")
    
    return {"result": result}

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Retrieve Shared Context
        context = agent_manager.get_context()
        
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
        
        # 4. Update Chat History (Optional, for future context)
        # current_history = context.get("chat_history", [])
        # current_history.append({"role": "user", "content": request.query})
        # current_history.append({"role": "assistant", "content": response_text})
        # agent_manager.update_context("chat_history", current_history)
        
        return {"response": response_text}
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
