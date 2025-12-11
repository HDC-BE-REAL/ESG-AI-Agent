from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import shutil
import os


from src.tools.report_tool.report_tool import generate_report_from_query
from src.tools.regulation_tool import _monitor_instance as regulation_monitor
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

@router.post("/agent/{agent_type}")
async def run_agent(agent_type: str, request: AgentRequest):
    if agent_type == "policy":
        result = await agent_manager.run_policy_agent(request.query)
    elif agent_type == "regulation":
        result = await agent_manager.run_regulation_agent(request.query)
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
        You are an expert ESG AI Assistant. Provide concise, tailored answers that reflect the user's goal and constraints.

        [Current Context]
        - Uploaded Files: {[f['filename'] for f in context.get('uploaded_files', [])]}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}
        
        [Instructions]
        - Start by tagging the user's goal/constraints in one line; if unclear, ask ONE short clarifying question, then proceed.
        - Use evidence in this priority: Regulation Updates → Policy Analysis → Risk Assessment → Report Draft → Uploaded Files → Chat History; if absent, note '해당 근거 없음'.
        - Keep internal reasoning to 3 short lines before responding.
        - Do not invent numbers/dates absent from context; flag missing data explicitly. When giving numbers, cite the source inline. If regulation/policy is mentioned, add a one-line note that this is not legal advice.
        - Tone: professional and friendly; keep sections 2–4 bullets/lines; keep the whole response concise (~200 words).
        - Language follows the user (default Korean); avoid mixing languages. Use - or * for bullets, **bold** for emphasis, `code` for technical terms.
        - If confidence is low, mark it (신뢰도: 높음/중간/낮음) and suggest what to check next (file/regulation/data).
        - ALWAYS use MARKDOWN formatting.

        [Output Format - keep structured but flexible]
        ## 🎯 목표/제약
        - (1줄; 모르면 질문 1개)

        ## 📊 요약
        - 2~3문장 핵심

        ## 🔍 근거 (신뢰도 표기)
        - 근거 1 (신뢰도: …)
        - 근거 2
        - 근거 3 또는 '추가 데이터 필요: ...'

        ## 💡 권고사항
        - 권고 1 (사용자 목표/제약 반영)
        - 권고 2

        ## ▶️ 다음 행동
        - 실행 제안 1~2개 + 필요한 확인사항 1개

        If you don't know, say so and recommend running the appropriate agent (Regulation, Policy, Risk, Report).
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
        
        # 0. Intent Detection (LLM-based)
        # Check if the user specifically wants to *generate* or *create* a report/checklist/document.
        # Simple references (e.g. "read my report") should be False.
        
        class IntentAnalysis(BaseModel):
            is_generation_request: bool = Field(description="True if user wants to CREATE/WRITE a report/checklist, False otherwise.")
            
        intent_system_prompt = """
        Analyze the user's latest query to determine if they want to GENERATE a new report, checklist, or document.
        
        True:
        - "Make a report about..."
        - "Create a checklist for..."
        - "Report please"
        - "보고서 만들어줘"
        - "체크리스트 작성해줘"
        
        False:
        - "Summarize this report"
        - "What is in the report?"
        - "Refer to the uploaded file"
        - "보고서 요약해줘"
        - "내가 올린 파일 참고해"
        """
        
        intent_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = intent_llm.with_structured_output(IntentAnalysis)
        intent_result = await structured_llm.ainvoke([
            SystemMessage(content=intent_system_prompt),
            HumanMessage(content=request.query)
        ])
        
        is_report_request = intent_result.is_generation_request
        
        if is_report_request:
            # Generate Report Content using ReportTool (GRI Standard)
            print(f"📄 Report generation intent detected for: {request.query}")
            try:
                # 1. Define Content Schema
                class MaterialIssue(BaseModel):
                    name: str = Field(description="Name of the material issue (e.g., 'Safety Training', 'Cardbon Emission')")
                    impact: int = Field(description="Importance to stakeholders (0-100)")
                    financial: int = Field(description="Financial impact (0-100)")
                    isMaterial: bool = Field(description="Always set to True for key issues")

                class ReportSection(BaseModel):
                    title: str = Field(description="Section heading (e.g., 'Safety Management System', 'Carbon Reduction Plan')")
                    content: str = Field(description="Section content in markdown (bullet points, tables, etc.)")

                class ReportContentGen(BaseModel):
                    company_name: str = Field(description="Exact Name of the company found in the [Uploaded File Content].")
                    esg_strategy: str = Field(description="Main ESG strategy sentence from the file.")
                    
                    # Standard Fields (Optional if custom sections used)
                    env_policy: Optional[str] = Field(description="Environmental policy summary (Standard K-ESG).")
                    social_policy: Optional[str] = Field(description="Social policy summary (Standard K-ESG).")
                    gov_structure: Optional[str] = Field(description="Governance structure summary (Standard K-ESG).")
                    
                    # Standard Materiality (Structured Table) - Optional
                    material_issues: Optional[List[MaterialIssue]] = Field(default=None, description="List of key material issues. If user wants a custom format, leave this empty and use custom_sections.")
                    
                    # Dynamic Fields
                    custom_sections: List[ReportSection] = Field(description="Dynamic sections for specific topics (e.g. if user asks for 'Safety Report', create sections like 'Risk Assessment', 'Safety Training').")

                # 2. Extract context from uploaded files
                uploaded_files = context.get("uploaded_files", [])
                file_context_str = ""
                
                if uploaded_files:
                    print(f"📂 Processing {len(uploaded_files)} uploaded files for context...")
                    import pypdf
                    
                    # User Request: Use ALL files uploaded in the current session.
                    # (History is cleared on server start via manager.py)
                    for text_file in uploaded_files: 
                        try:
                            fname = text_file.get("filename")
                            # Reconstruct absolute path
                            fpath = os.path.join(UPLOAD_DIR, fname)
                            
                            content = ""
                            if fname.lower().endswith(".pdf"):
                                reader = pypdf.PdfReader(fpath)
                                # Read ALL pages (User requested full context)
                                content = "\n".join([utils.extract_text() for utils in reader.pages])
                            else:
                                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read()
                            
                            # Increase limit to 100k chars for GPT-4o
                            file_context_str += f"\n=== File: {fname} ===\n{content[:100000]}\n" 
                            print(f"   - Read {len(content)} chars from {fname}")
                        except Exception as e:
                            print(f"⚠️ Failed to read file {fname}: {e}")

                # 3. Generate Content with LLM
                content_system_prompt = f"""
                You are an expert ESG consultant specializing in K-ESG (Korean ESG Guidelines).
                The user wants a report about: "{request.query}"
                
                [Context]
                - Industry: Construction (Default) or as implied by query.
                - Guidelines: K-ESG Guideline v2.0.
                
                [Uploaded File Content]
                {file_context_str if file_context_str else "No uploaded files found."}
                
                [Instructions]
                Generate REALISTIC and SPECIFIC content based ONLY on the [Uploaded File Content] above.
                
                **STRUCTURE RULE**:
                - IF the user asks for a **General/Standard Report**: Fill in `env_policy`, `social_policy`, `gov_structure`.
                - IF the user asks for a **Specific Topic** (e.g. "Safety Report", "Carbon Report"): 
                  - **IGNORE** the standard policy fields (leave them empty or brief).
                  - **CREATE** detailed `custom_sections` relevant to that topic (e.g. "Risk Assessment", "Safety Training", "Accident Stats").
                
                1. **Company Name Priority**: 
                   - **IF** the user mentioned a specific company name in the query (e.g., "Make a report for (주)SubCorp"), USE THAT NAME.
                   - **ELSE**, extract the company name strictly from the file.
                
                2. **Strategy/Policy**: Summarize the actual strategies found in the text.
                3. **Issues**: Identify 3-5 real material issues mentioned in the text.
                
                - Do NOT use GRI/SASB terms unless in the file.
                - Focus on local regulations and K-ESG indicators.
                - Do NOT use placeholders like "Input".
                - Write in Korean.
                """
                
                llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                structured_llm = llm.with_structured_output(ReportContentGen)
                
                print("🧠 Generating report content with LLM...")
                generated_data = await structured_llm.ainvoke([
                    SystemMessage(content=content_system_prompt),
                    HumanMessage(content=f"Generate K-ESG content for: {request.query}")
                ])
                
                report_data = generated_data.model_dump()
                report_data['report_year'] = "2025" # Default year
                
                # 3. Create Report
                report_content = generate_report_from_query(
                    query=request.query, 
                    extra_data=report_data,
                    standard="K-ESG"
                )
                report_error = None
            except Exception as e:
                print(f"❌ Report generation failed: {e}")
                import traceback
                traceback.print_exc()
                report_content = None
                report_error = f"Report Generation Error: {str(e)}"
        else:
            report_content = None
            report_error = None

        custom_result = await agent_manager.run_custom_agent(request.query)
        risk_assessment = context.get('risk_assessment')
        risk_summary = str(risk_assessment)[:500] + "..." if risk_assessment else "None"
        history = context.get("chat_history", [])
        history_text = "\n".join(
            [f"User: {entry['content']}" if entry.get('role') == 'user' else f"Assistant: {entry['content']}" for entry in history]
        )

        system_prompt = f"""
        You are an expert ESG AI Assistant. Provide concise, tailored answers that reflect the user's goal and constraints.

        [Current Context]
        - Uploaded Files: {[f['filename'] for f in context.get('uploaded_files', [])]}
        - Latest Regulation Updates: {str(context.get('regulation_updates'))[:500] + "..." if context.get('regulation_updates') else "None"}
        - Policy Analysis: {context.get('policy_analysis', 'None')}
        - Risk Assessment: {risk_summary}
        - Report Draft: {context.get('report_draft', 'None')}

        [Conversation History]
        {history_text if history_text else 'None'}

        [Auto-Generated Insights]
        - Policy Summary: {custom_result.get('policy')}
        - Regulation Update: {custom_result.get('regulation')}
        - Risk Analysis: {custom_result.get('risk')}
        - Report Draft: {custom_result.get('report')}

        [Instructions]
        - Start by tagging the user's goal/constraints in one line; if unclear, ask ONE short clarifying question, then proceed.
        - Use evidence priority: Regulation Updates → Policy Analysis → Risk Assessment → Report Draft → Uploaded Files → Chat History; if absent, note '해당 근거 없음'.
        - Keep internal reasoning to 3 short lines before responding.
        - Do not invent numbers/dates absent from context; flag missing data explicitly. When giving numbers, cite the source inline. If regulation/policy is mentioned, add a one-line note that this is not legal advice.
        - Tone: professional and friendly; keep sections 2–4 bullets/lines; keep the whole response concise (~200 words).
        - Language follows the user (default Korean); avoid mixing languages. Use - or * for bullets, **bold** for emphasis, `code` for technical terms.
        - If confidence is low, mark it (신뢰도: 높음/중간/낮음) and suggest what to check next (file/regulation/data).
        - 답변에 최신 규제/정책/리스크/파일 인사이트를 자연스럽게 녹여라.
        - ALWAYS use MARKDOWN formatting.

        [Output Format - keep structured but flexible]
        ## 🎯 목표/제약
        - (1줄; 모르면 질문 1개)
        ## 📊 요약
        - 2~3문장 핵심

        ## 🔍 근거 (신뢰도 표기)
        - 근거 1 (신뢰도: …)
        - 근거 2
        - 근거 3 또는 '추가 데이터 필요: ...'

        ## 💡 권고사항
        - 권고 1 (사용자 목표/제약 반영)
        - 권고 2

        ## ▶️ 다음 행동
        - 실행 제안 1~2개 + 필요한 확인사항 1개

        If you don't know, say so and recommend running the appropriate agent (Regulation, Policy, Risk, Report).
        """
        
        # If report was generated, modify the system prompt or the conversational response to reflect that.
        if report_content:
             system_prompt += "\n[System Note]\nA report has just been generated and displayed to the user. Briefly mention this in your response (e.g., '요청하신대로 보고서를 생성하여 화면에 띄워드렸습니다.')."

        llm = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]

        async def event_generator():
            try:
                # 1. If report generated, send it first
                if report_content:
                    yield f"data: {json.dumps({'report': report_content})}\n\n"
                
                # 1b. If report failed, send error
                if report_error:
                    yield f"data: {json.dumps({'error': report_error})}\n\n"
                
                # 2. Stream conversational response
                async for chunk in llm.astream(messages):
                    token = chunk.content or ""
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"❌ [API Error] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
