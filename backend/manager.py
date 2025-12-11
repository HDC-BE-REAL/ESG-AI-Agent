import sys
import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.regulation_tool import _monitor_instance as regulation_monitor
from src.tools.risk import RiskToolOrchestrator
from src.tools.policy_tool import policy_guideline_tool
from src.tools.report_tool import draft_report
from src.workflows.custom_graph import run_langgraph_pipeline
from backend.kv_store import kv_store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

LOGGER = logging.getLogger(__name__)
CONVERSATION_VECTOR_DIR = Path("vector_db/conversations")


class AgentManager:
    DEFAULT_TITLE = "새 대화"

    def __init__(self):
        # ① 업로드된 파일·규제 업데이트·정책 분석 등 모든 컨텍스트를 저장
        default_context: Dict[str, Any] = {
            "uploaded_files": [],
            "regulation_updates": None,
            "policy_analysis": None,
            "risk_assessment": None,
            "report_draft": None,
            "chat_history": [],  # legacy
            # 대화방별로 메시지를 보관하기 위한 저장소
            "conversations": {},
        }
        persisted = kv_store.load_context() or {}
        # ④ Redis에 저장된 값이 있다면 기본 컨텍스트 위에 덮어써 복원
        default_context.update(persisted)
        
        # [Strict Session] 서버 시작 시 과거 업로드 파일 기록은 초기화함 (User Request)
        # Persistent context should keep generic things, but files should be current session only.
        default_context["uploaded_files"] = [] 
        
        self.shared_context = default_context
        self._risk_orchestrator = RiskToolOrchestrator()
        CONVERSATION_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        # 업로드 파일용 임베딩/텍스트 분할기 (벡터DB에 재사용)
        # 업로드 파일을 Chroma에 넣기 위한 임베딩/청크 분리기
        self._conv_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self._conv_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        self._title_llm: Optional[ChatOpenAI] = None

    def get_context(self) -> Dict[str, Any]:
        return self.shared_context

    def update_context(self, key: str, value: Any):
        self.shared_context[key] = value
        self._persist_context()

    def _persist_context(self):
        # ⑤ Redis 사용 가능 시 전체 컨텍스트를 JSON으로 동기화
        if not kv_store.save_context(self.shared_context):
            LOGGER.warning("Redis 컨텍스트 저장 실패 - 메모리 모드로 지속")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_conversations(self) -> Dict[str, Any]:
        # Redis 복원 시 conversations 키가 없을 수 있으므로 setdefault 사용
        return self.shared_context.setdefault("conversations", {})

    def list_conversations(self) -> List[Dict[str, Any]]:
        conversations = self._get_conversations()
        summaries: List[Dict[str, Any]] = []
        for convo in conversations.values():
            messages = convo.get("messages", [])
            last_message = messages[-1]["content"] if messages else ""
            summaries.append({
                "id": convo.get("id"),
                "title": convo.get("title", "새 대화"),
                "updated_at": convo.get("updated_at"),
                "last_message": last_message,
            })
        return sorted(
            summaries,
            key=lambda item: item.get("updated_at") or "",
            reverse=True,
        )

    def create_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        # ChatGPT처럼 UUID 기반 세션을 생성
        conv_id = str(uuid.uuid4())
        now = self._now()
        conversation = {
            "id": conv_id,
            "title": title or self.DEFAULT_TITLE,
            "messages": [],
            "files": [],
            "reports": [],
            "created_at": now,
            "updated_at": now,
        }
        conversations = self._get_conversations()
        conversations[conv_id] = conversation
        self.update_context("conversations", conversations)
        return conversation

    def delete_conversation(self, conversation_id: str) -> bool:
        conversations = self._get_conversations()
        if conversation_id in conversations:
            conversations.pop(conversation_id)
            self.update_context("conversations", conversations)
            return True
        return False

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._get_conversations().get(conversation_id)

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        conversation = self.get_conversation(conversation_id)
        return conversation.get("messages", []) if conversation else []

    def list_conversation_files(self, conversation_id: str) -> List[Dict[str, Any]]:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        files = conversation.get("files", [])
        return [
            {
                "id": entry.get("id"),
                "filename": entry.get("filename"),
                "size_bytes": entry.get("size_bytes"),
                "uploaded_at": entry.get("uploaded_at"),
                "path": entry.get("path"),
            }
            for entry in files
        ]

    def append_conversation_message(self, conversation_id: str, role: str, content: str):
        conversations = self._get_conversations()
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        now = self._now()
        conversation.setdefault("messages", []).append({
            "role": role,
            "content": content,
            "timestamp": now,
        })
        if role == "user":
            title = conversation.get("title", "")
            if not title or title == self.DEFAULT_TITLE:
                conversation["title"] = self._guess_conversation_title(content)
        conversation["updated_at"] = now
        self.update_context("conversations", conversations)

    def add_conversation_file(
        self,
        conversation_id: str,
        *,
        filename: str,
        path: str,
        size_bytes: int,
        text: str,
    ):
        conversations = self._get_conversations()
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        file_entry = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "path": path,
            "size_bytes": size_bytes,
            "uploaded_at": self._now(),
            "text": (text or "")[:10000],
        }
        conversation.setdefault("files", []).append(file_entry)
        # 전역 uploaded_files에도 정보 남겨두어 기존 로직 영향 최소화
        uploaded = self.shared_context.setdefault("uploaded_files", [])
        uploaded = [entry for entry in uploaded if entry.get("filename") != filename]
        uploaded.append({"filename": filename, "path": path})
        if len(uploaded) > 50:
            uploaded = uploaded[-50:]
        self.shared_context["uploaded_files"] = uploaded
        conversation["updated_at"] = self._now()
        # 대화방 전용 Chroma에 즉시 임베딩 upsert
        try:
            self._upsert_conversation_embeddings(conversation_id, text, filename)
        except Exception as exc:  # pragma: no cover - 임베딩 실패 시 로그만 남김
            LOGGER.warning("대화방 임베딩 추가 실패(%s): %s", conversation_id, exc)
        self.update_context("conversations", conversations)

    def add_conversation_report(self, conversation_id: str, report_data: Dict[str, Any]):
        conversations = self._get_conversations()
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        
        # report_data expected to have id, title, content, creates_at etc. 
        # If ID is missing, generate one
        if "id" not in report_data:
            report_data["id"] = str(uuid.uuid4())
        if "created_at" not in report_data:
            report_data["created_at"] = self._now()
            
        conversation.setdefault("reports", []).append(report_data)
        conversation["updated_at"] = self._now()
        self.update_context("conversations", conversations)

    def list_conversation_reports(self, conversation_id: str) -> List[Dict[str, Any]]:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        return conversation.get("reports", [])

    def _guess_conversation_title(self, content: str) -> str:
        """LLM을 사용해 대화 제목을 생성하고 실패 시 간단 요약으로 대체"""
        generated = self._generate_title_with_llm(content)
        if generated:
            return generated
        first_line = content.strip().splitlines()[0].strip()
        if first_line.endswith("?"):
            first_line = first_line[:-1]
        if len(first_line) > 20:
            first_line = first_line[:20] + "..."
        return first_line or self.DEFAULT_TITLE

    def build_file_context(self, conversation_id: str, *, max_total_chars: int = 4000) -> str:
        files = self.get_conversation_files_with_text(conversation_id)
        if not files:
            return ""
        # 개수만큼 분배해 너무 긴 텍스트 방지
        slice_len = max_total_chars // len(files) if files else max_total_chars
        if slice_len <= 0:
            slice_len = max_total_chars
        contexts = []
        for entry in files:
            text = (entry.get("text") or "")[:slice_len]
            if not text:
                continue
            contexts.append(f"[파일: {entry.get('filename')}]\n{text}")
        return "\n\n".join(contexts)

    def get_conversation_files_with_text(self, conversation_id: str) -> List[Dict[str, Any]]:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        return conversation.get("files", [])

    def _get_conversation_vector_path(self, conversation_id: str) -> Path:
        return CONVERSATION_VECTOR_DIR / conversation_id

    def _get_conversation_vectorstore(self, conversation_id: str) -> Chroma:
        persist_dir = str(self._get_conversation_vector_path(conversation_id))
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(
            collection_name=f"convo_{conversation_id}",
            embedding_function=self._conv_embeddings,
            persist_directory=persist_dir,
        )

    def _upsert_conversation_embeddings(self, conversation_id: str, text: str, filename: str):
        """대화방 전용 Chroma 컬렉션에 파일 청크를 업로드"""
        if not text:
            return
        chunks = self._conv_splitter.split_text(text)
        if not chunks:
            return
        vectorstore = self._get_conversation_vectorstore(conversation_id)
        metadatas = [
            {
                "filename": filename,
                "chunk": idx,
            }
            for idx, _ in enumerate(chunks)
        ]
        ids = [f"{filename}-{uuid.uuid4()}" for _ in chunks]
        vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    def retrieve_conversation_snippets(self, conversation_id: str, query: str, k: int = 4) -> List[str]:
        """대화방별 업로드 문서에서 쿼리와 유사한 청크를 검색"""
        vector_path = self._get_conversation_vector_path(conversation_id)
        if not vector_path.exists() or not any(vector_path.iterdir()):
            return []
        try:
            vectorstore = self._get_conversation_vectorstore(conversation_id)
            docs = vectorstore.similarity_search(query, k=k)
        except Exception as exc:  # pragma: no cover - 오류 발생 시 빈 컨텍스트 반환
            LOGGER.warning("대화방 RAG 검색 실패(%s): %s", conversation_id, exc)
            return []
        return [f"[파일:{doc.metadata.get('filename')}]{doc.page_content}" for doc in docs]

    def _generate_title_with_llm(self, content: str) -> Optional[str]:
        try:
            if self._title_llm is None:
                # 짧은 제목만 필요하므로 낮은 temperature와 max_tokens 설정
                self._title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=32)
            messages = [
                SystemMessage(
                    content=(
                        "너는 대화 주제를 한 줄 제목으로 요약하는 비서야. "
                        "12자 내외의 한국어나 영어 명사구로 답하고 따옴표나 마침표를 붙이지 마."
                    )
                ),
                HumanMessage(content=content),
            ]
            response = self._title_llm.invoke(messages)
            title = (response.content or "").strip()
            if len(title) > 20:
                title = title[:20] + "..."
            return title or None
        except Exception as exc:
            LOGGER.warning("대화 제목 생성 LLM 호출 실패: %s", exc)
            return None

    async def run_regulation_agent(self, query: str = "ESG 규제 동향") -> str:
        """
        Runs the Regulation Monitor tool and updates the shared context.
        """
        print(f"🚀 [AgentManager] Starting Regulation Agent with query: {query}")
        
        # Run the existing monitor logic
        # Note: monitor_all is synchronous, might block if not careful, 
        # but for now we run it directly. In production, use a thread pool or background task.
        try:
            # report = regulation_monitor.monitor_all(query)
            # Use generate_report for instant response (browsing happens in background)
            # ② regulation/policy/risk/report agent 실행
            report = regulation_monitor.generate_report(query)
            self.update_context("regulation_updates", report)
            return report
        except Exception as e:
            error_msg = f"Error running regulation agent: {str(e)}"
            print(error_msg)
            return error_msg

    async def run_policy_agent(self, query: str) -> str:
        try:
            result = policy_guideline_tool(query)
            self.update_context("policy_analysis", result)
            return result
        except Exception as exc:
            error_msg = f"Policy agent 실행 오류: {exc}"
            LOGGER.error(error_msg)
            return error_msg

    async def run_risk_agent(self, query: str, focus_area: Optional[str] = None) -> str:
        """리스크 오케스트레이터를 호출해 ISO31000/Materiality 결과를 생성하고 컨텍스트에 저장"""
        try:
            result = self._risk_orchestrator.run(query=query, focus_area=focus_area)
            # ③ 최신 리스크 분석 리포트를 공유 컨텍스트에 넣어 챗봇·리포트 에이전트에서 활용
            self.update_context("risk_assessment", result)
            return result
        except Exception as exc:
            error_msg = f"Risk agent 실행 오류: {exc}"
            print(error_msg)
            return error_msg

    async def run_report_agent(self, query: str, audience: Optional[str] = None) -> str:
        try:
            result = draft_report(query, audience)
            self.update_context("report_draft", result)
            return result
        except Exception as exc:
            error_msg = f"Report agent 실행 오류: {exc}"
            LOGGER.error(error_msg)
            return error_msg

    async def run_custom_agent(self, query: str, *, focus_area: Optional[str] = None, audience: Optional[str] = None) -> Dict[str, str]:
        """LangGraph 기반 파이프라인으로 4개 모듈을 동시에 실행"""
        result = run_langgraph_pipeline(query, focus_area, audience)
        self.update_context("policy_analysis", result.get("policy"))
        self.update_context("regulation_updates", result.get("regulation"))
        self.update_context("risk_assessment", result.get("risk"))
        self.update_context("report_draft", result.get("report"))
        return {
            "policy": result.get("policy", ""),
            "regulation": result.get("regulation", ""),
            "risk": result.get("risk", ""),
            "report": result.get("report", ""),
        }

# Singleton instance
agent_manager = AgentManager()
