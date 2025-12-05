"""ESG 에이전트를 위한 리트리버 파이프라인 구성."""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import ConfigDict, Field

from pydantic import Field, ConfigDict

DEFAULT_VECTOR_DIR = Path("vector_db/esg_all")
DEFAULT_COLLECTION = "esg_all"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def load_vectorstore(
    persist_directory: Path | str = DEFAULT_VECTOR_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """벡터 구축 단계와 동일한 임베딩으로 저장된 Chroma를 불러온다."""

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return Chroma(
        persist_directory=str(persist_directory),
        collection_name=DEFAULT_COLLECTION,
        embedding_function=embeddings,
    )


@dataclass
class QueryRewriter:
    """도메인 키워드 확장을 위한 LLM 기반 쿼리 리라이팅."""

    llm: BaseChatModel
    prompt: ChatPromptTemplate | None = None

    def __post_init__(self) -> None:
        if not self.prompt:
            self.prompt = ChatPromptTemplate.from_template(
                """
                아래 사용자의 질문을 ESG 보고서 검색 쿼리로 재작성하라.
                - 회사명, ESG 영역(E/S/G), 정책 키워드, 영어 동의어를 포함할 것
                - bullet 없이 한 줄로 작성할 것
                질문: {question}
                메타데이터 필터: {filter}
                """.strip()
            )

    def rewrite(self, question: str, metadata_filter: Dict | None = None) -> str:
        chain = self.prompt | self.llm
        response = chain.invoke({"question": question, "filter": metadata_filter or {}})
        return response.content.strip() or question


class CrossEncoderReranker:
    """FlagEmbedding cross-encoder를 활용한 선택적 재정렬기."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        use_fp16: bool = True,
    ) -> None:
        from FlagEmbedding import FlagReranker  # lazy import

        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, docs: Sequence[Document], top_k: int) -> List[Document]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.compute_score(pairs, normalize=True)
        ranked = sorted(zip(scores, docs), key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]


PostFilter = Callable[[Document], bool]


def default_post_filter(doc: Document) -> bool:
    """명백한 노이즈 청크(OCR 잡음 등)를 제거한다."""

    if doc.metadata.get("ocr") and len(doc.page_content) < 60:
        return False
    return True


class ESGRetriever(BaseRetriever):
    """리라이팅 → 벡터 검색 → 재정렬 → 후처리 흐름을 묶은 리트리버."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    query_rewriter: QueryRewriter | None = None
    metadata_filter: Dict | None = Field(default_factory=dict)
    reranker: CrossEncoderReranker | None = None
    post_filter: PostFilter | None = default_post_filter
    top_k: int = 6
    fetch_k: int = 30
    mmr_lambda: float = 0.7

    @staticmethod
    def _parse_input(
        query: Union[str, Dict[str, Union[str, Dict]]]
    ) -> Tuple[str, Dict | None]:
        if isinstance(query, str):
            return query, None
        if "question" in query:
            return str(query["question"]), query.get("metadata_filter")
        if "query" in query:
            return str(query["query"]), query.get("metadata_filter")
        raise ValueError("Unsupported retriever input format")

    def _search(self, query: str, metadata_filter: Dict | None) -> List[Document]:
        base_filter = self.metadata_filter or {}
        filter_payload = {**base_filter, **(metadata_filter or {})}
        # fetch_k ensures we have enough variety for reranking/post filtering.
        docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=min(self.fetch_k, 50),
            fetch_k=self.fetch_k,
            lambda_mult=self.mmr_lambda,
            filter=filter_payload or None,
        )
        return docs

    def _apply_post_filter(self, docs: Iterable[Document]) -> List[Document]:
        if not self.post_filter:
            return list(docs)
        return [doc for doc in docs if self.post_filter(doc)]

    def _get_relevant_documents(
        self, query: Union[str, Dict[str, Union[str, Dict]]]
    ) -> List[Document]:
        text_query, metadata_filter = self._parse_input(query)
        rewritten = (
            self.query_rewriter.rewrite(text_query, metadata_filter)
            if self.query_rewriter
            else text_query
        )
        candidates = self._search(rewritten, metadata_filter)
        if self.reranker:
            candidates = self.reranker.rerank(rewritten, candidates, self.top_k)
        else:
            candidates = candidates[: self.top_k]

        filtered = self._apply_post_filter(candidates)
        return filtered[: self.top_k]

    async def _aget_relevant_documents(
        self, query: Union[str, Dict[str, Union[str, Dict]]]
    ) -> List[Document]:
        return self._get_relevant_documents(query)


def build_retriever(
    llm: BaseChatModel,
    *,
    vectorstore: Chroma | None = None,
    metadata_filter: Optional[Dict] = None,
    use_reranker: bool = True,
    top_k: int = 6,
    fetch_k: int = 30,
    mmr_lambda: float = 0.7,
) -> ESGRetriever:
    """Factory helper for LangGraph/LangChain nodes."""

    vectordb = vectorstore or load_vectorstore()
    rewriter = QueryRewriter(llm)
    reranker = CrossEncoderReranker() if use_reranker else None
    return ESGRetriever(
        vectorstore=vectordb,
        query_rewriter=rewriter,
        metadata_filter=metadata_filter,
        reranker=reranker,
        top_k=top_k,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
    )


if __name__ == "__main__":  # pragma: no cover
    try:
        from langchain_openai import ChatOpenAI  # example LLM
    except ModuleNotFoundError:
        raise SystemExit(
            "langchain-openai 패키지가 필요합니다. `pip install langchain-openai` 후 다시 실행하세요."
        )

    # .env 파일에서 OPENAI_API_KEY를 불러오는 간단한 헬퍼
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    import os

                    os.environ.setdefault("OPENAI_API_KEY", key)
                break
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit(
            "OPENAI_API_KEY가 환경 변수에 없습니다. .env에 추가하거나 직접 export 해주세요."
        )

    question = "DL건설의 현재 탄소배출 정보를 알려줘"
    llm = ChatOpenAI(model="gpt-4o-mini")
    retriever = build_retriever(llm)
    docs = retriever.invoke({"question": question, "metadata_filter": {"source_type": "companies"}})
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"[{idx}] {meta.get('source_file')} p.{meta.get('page')} (ocr={meta.get('ocr', False)})")
        print(doc.page_content[:400])
        print("-" * 80)
