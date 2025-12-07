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

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update shared context
        current_files = agent_manager.get_context().get("uploaded_files", [])
        current_files.append({"filename": file.filename, "path": file_path})
        agent_manager.update_context("uploaded_files", current_files)
        
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
        result = await agent_manager.run_risk_agent(request.query)
    elif agent_type == "report":
        result = await agent_manager.run_report_agent(request.query)
    elif agent_type == "custom":
        result = await agent_manager.run_custom_agent(request.query)
    else:
        raise HTTPException(status_code=404, detail="Agent type not found")
    
    return {"result": result}

@router.post("/chat")
async def chat(request: ChatRequest):
    # Placeholder for LLM chat integration
    # In a real scenario, this would call the LLM with the shared context
    response = f"Echo: {request.query}. (LLM integration pending)"
    return {"response": response}
