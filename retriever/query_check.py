"""벡터 DB 검색 결과를 빠르게 확인하는 간단한 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from .retriever_pipeline import DEFAULT_VECTOR_DIR, ESGRetriever, load_vectorstore


def parse_metadata_filters(items: List[str]) -> Dict[str, str]:
    """key=value 형태의 인자를 Dict로 변환한다."""

    filters: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"metadata 필터는 key=value 형태여야 합니다: {item}"
            )
        key, value = item.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="벡터 검색 결과 미리보기")
    parser.add_argument("question", help="검색하고 싶은 자연어 질문")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="metadata key=value 형식 (복수 지정 가능)",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--fetch-k", type=int, default=30)
    parser.add_argument("--mmr", type=float, default=0.7)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help="기존 persist_directory 경로",
    )
    return parser


def print_docs(docs: List[Document]) -> None:
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(
            f"[{idx}] {meta.get('source_file')} p.{meta.get('page')} "
            f"(type={meta.get('source_type')}, ocr={meta.get('ocr', False)})"
        )
        content = doc.page_content.replace("\n", " ")
        print(content[:400])
        print("-" * 80)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    metadata_filter = parse_metadata_filters(args.filter)
    vectorstore = load_vectorstore(persist_directory=args.vector_dir)

    # QueryRewriter / LLM 없이 ESGRetriever를 직접 구성해 검색 품질만 확인
    retriever = ESGRetriever(
        vectorstore=vectorstore,
        query_rewriter=None,
        metadata_filter=metadata_filter,
        reranker=None,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        mmr_lambda=args.mmr,
    )

    docs = retriever.invoke({"question": args.question})
    if not docs:
        print("검색 결과가 없습니다.")
        return
    print_docs(docs)


if __name__ == "__main__":
    main()
