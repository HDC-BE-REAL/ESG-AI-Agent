from pathlib import Path
import shutil
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from collections import Counter
from typing import Iterable
import numpy as np
import hashlib

try:
    from openparse.doc_parser import DocumentParser
except ImportError:  # pragma: no cover - optional dependency
    DocumentParser = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langdetect import detect


# 0. 기본 설정
DATA_DIR = Path("data")
VECTOR_DIR = "vector_db/esg_all"

# HuggingFace 임베딩 (4060 GPU 활용 가능)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",      # 다국어 지원, 성능/속도 괜찮음
    # encode_kwargs={"normalize_embeddings": True},  # 선택 옵션
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

LAYOUT_KEYWORDS = {
    "CONTENTS",
    "TABLE OF CONTENTS",
    "INDEX",
    "SUSTAINABILITY REPORT",
}

NAV_MENU_WORDS = {
    "OVERVIEW",
    "ENVIRONMENTAL",
    "SOCIAL",
    "GOVERNANCE",
    "APPENDIX",
}

COUNTRY_BY_SOURCE = {
    "domestic": "KR",
    "companies": "KR",
    "global": "GLOBAL",
}

OPENPARSE_TARGET_FILES: set[str] | None = None  # 전체 companies 문서에 OpenParse 적용


def infer_pdf_metadata(pdf_path: Path, source_type: str) -> dict:
    """Extract company/year/country metadata from filename and folder."""

    stem = pdf_path.stem
    company = stem.split("_")[0].strip()
    year_match = re.search(r"(20\\d{2})", stem)
    year = year_match.group(1) if year_match else None

    meta = {}
    if company:
        meta["company"] = company
    if year:
        meta["year"] = year

    country = COUNTRY_BY_SOURCE.get(source_type)
    if country:
        meta["country"] = country
    return meta


# -------------------------------------------------------
# 1. 텍스트/OCR 추출 도우미
# -------------------------------------------------------
def _load_pdf_pages_pymupdf(pdf_path, source_type):
    doc = fitz.open(pdf_path)
    pages = []
    for idx, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": Path(pdf_path).name,
                    "source_type": source_type,
                    "page": idx + 1,
                },
            )
    )
    return pages


_OPENPARSE_PARSER = None
OPENPARSE_PREVIEW_NODES = 2  # OpenParse 결과를 빠르게 확인하기 위한 샘플 개수


def should_use_openparse(pdf_path: Path, source_type: str) -> bool:
    return DocumentParser is not None and source_type == "companies"


def get_openparse_parser() -> DocumentParser:
    """OpenParse 문서 파서를 1회만 생성."""

    if DocumentParser is None:
        raise RuntimeError("openparse 패키지가 설치되어 있지 않습니다.")

    global _OPENPARSE_PARSER
    if _OPENPARSE_PARSER is None:
        table_args = {
            "parsing_algorithm": "table-transformers",
            "table_output_format": "markdown",
            "min_table_confidence": 0.4,
            "min_cell_confidence": 0.2,
        }
        _OPENPARSE_PARSER = DocumentParser(table_args=table_args)
    return _OPENPARSE_PARSER


def _node_to_text(node) -> str:
    parts = []
    for element in getattr(node, "elements", []):
        text = getattr(element, "text", "")
        if text:
            parts.append(text.strip())
    return "\n".join(part for part in parts if part).strip()


def _node_page(node) -> int | None:
    bboxes = getattr(node, "bbox", [])
    pages = [bbox.page for bbox in bboxes if hasattr(bbox, "page")]
    return min(pages) if pages else None


def _load_pdf_pages_openparse(pdf_path: Path, source_type: str):
    parser = get_openparse_parser()
    parsed = parser.parse(str(pdf_path), ocr=False)
    documents = []
    for idx, node in enumerate(parsed.nodes, start=1):
        text = _node_to_text(node)
        if not text:
            continue
        page_no = _node_page(node) or idx
        metadata = {
            "source_file": pdf_path.name,
            "source_type": source_type,
            "page": page_no,
            "parser": "openparse",
        }
        if idx <= OPENPARSE_PREVIEW_NODES:
            print(
                f"\n[OpenParse] {pdf_path.name} node {idx} preview:\n"
                f"{text[:500]}\n{'-' * 60}"
            )
        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )
    return documents


def load_pdf_pages(pdf_path, source_type):
    """특정 companies 문서는 OpenParse, 나머지는 PyMuPDF 사용."""

    pdf_path = Path(pdf_path)
    if should_use_openparse(pdf_path, source_type):
        try:
            return _load_pdf_pages_openparse(pdf_path, source_type)
        except Exception as exc:
            print(f"[OpenParse] 실패, PyMuPDF로 대체 ({pdf_path.name}): {exc}")
    return _load_pdf_pages_pymupdf(str(pdf_path), source_type)


def extract_images_from_pdf(pdf_path, target_pages=None):
    doc = fitz.open(pdf_path)
    texts = []
    targets = set(target_pages or [])

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        if targets and page_number not in targets:
            continue

        images = page.get_images()
        for img_index, img in enumerate(images):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))

            ocr_text = perform_ocr(pil_img)
            ocr_text = normalize_ocr_text(ocr_text)
            if len(ocr_text.strip()) > 10:
                texts.append((page_number, ocr_text))

    return texts  # [(page, text), ...]


# -------------------------------------------------------
# 2. 자동 헤더/푸터 탐지
# -------------------------------------------------------
def looks_like_navigation_ui(text: str) -> bool:
    upper = text.upper()
    nav_hits = sum(1 for word in NAV_MENU_WORDS if word in upper)
    if nav_hits >= 4:
        return True
    return any(keyword in upper for keyword in LAYOUT_KEYWORDS)


def is_navigation_line(line: str) -> bool:
    """헤더/목차 전용 라인을 탐지하여 필터링한다."""

    stripped = line.strip()
    if not stripped:
        return False

    upper = stripped.upper()
    if upper in NAV_MENU_WORDS or upper in LAYOUT_KEYWORDS:
        return True

    tokens = [tok for tok in re.split(r"[\s·|]+", upper) if tok]
    if not tokens:
        return False

    nav_hits = sum(1 for tok in tokens if tok in NAV_MENU_WORDS)
    if nav_hits >= 3 and nav_hits == len(tokens):
        return True
    if nav_hits >= 4:
        return True
    return False


def is_valid_header_footer_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 60:
        return False
    token_count = len(re.findall(r"[A-Za-z가-힣]+", stripped))
    if token_count >= 10:
        return False
    if looks_like_navigation_ui(stripped):
        return False
    return True


def detect_repeating_headers_footers(page_texts, top_n=3, bottom_n=3):
    header_counter = Counter()
    footer_counter = Counter()

    def filtered_lines(lines):
        return [line.strip() for line in lines if is_valid_header_footer_line(line)]

    for txt in page_texts:
        lines = txt.split("\n")
        header_counter.update(filtered_lines(lines[:top_n]))
        footer_counter.update(filtered_lines(lines[-bottom_n:]))

    total_pages = len(page_texts)
    threshold = max(2, int(total_pages * 0.6))
    common_headers = {h for h, c in header_counter.items() if c >= threshold}
    common_footers = {f for f, c in footer_counter.items() if c >= threshold}

    return common_headers, common_footers


# -------------------------------------------------------
# 3. 본문 정제 함수
# -------------------------------------------------------
def drop_garbage_lines(text: str) -> str:
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if is_navigation_line(stripped):
            continue
        if stripped.isdigit():
            continue
        if re.fullmatch(r"[IVXLCDM]+", stripped):
            continue
        if re.fullmatch(r"[A-Z]{1,3}", stripped):
            continue
        words = re.findall(r"[A-Za-z가-힣]+", stripped)
        if words and all(len(w) <= 2 for w in words):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def clean_text_basic(text):
    if not text:
        return None

    filtered = drop_garbage_lines(text)
    if not filtered:
        return None

    t = filtered.strip()

    # 길이가 너무 짧으면 제거
    if len(t) < 10:
        return None

    # 숫자/기호 비율이 너무 높은 라인은 garbage
    if sum(c.isalpha() for c in t) / (len(t) + 1) < 0.2:
        return None

    return t


def strip_header_footer(text, headers, footers):
    lines = text.split("\n")
    cleaned = [
        line for line in lines
        if line.strip() not in headers and line.strip() not in footers
    ]
    return "\n".join(cleaned).strip()


def should_skip_page(text: str, page_number: int) -> (bool, str | None):
    if page_number == 1:
        return True, "cover"

    upper = text.upper()

    if page_number <= 3:
        if looks_like_navigation_ui(text):
            return True, "nav_ui"
        if any(keyword in upper for keyword in LAYOUT_KEYWORDS):
            return True, "layout_keyword"

    return False, None


def page_needs_ocr(text: str) -> bool:
    if not text or not text.strip():
        return True
    alpha_chars = sum(c.isalpha() for c in text)
    return alpha_chars < 15


# -------------------------------------------------------
# 4. 단일 PDF → Documents
# -------------------------------------------------------
def perform_ocr(image: Image.Image) -> str:
    """PaddleOCR가 설치되어 있으면 한글+영문 모델로 인식하고, 없으면 Tesseract 사용."""

    if PaddleOCR is not None:
        if "_PADDLE_OCR" not in globals():
            # 한국어/영문 모두 포함하는 다국어 모델 초기화 (비동기 로딩 방지)
            globals()["_PADDLE_OCR"] = PaddleOCR(
                lang="korean",
                use_angle_cls=True,
                show_log=False,
            )
        ocr_engine = globals()["_PADDLE_OCR"]
        np_img = np.array(image.convert("RGB"))
        result = ocr_engine.ocr(np_img, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                value = line[1][0]
                if value:
                    texts.append(value)
        if texts:
            return "\n".join(texts)

    # 백업: 기존 Tesseract 전략 유지
    return pytesseract.image_to_string(image, lang="kor+eng")


def normalize_ocr_text(text: str) -> str:
    """간단한 언어 감지 후 한글/영문 전용 정규화를 적용한다."""

    stripped = text.strip()
    if not stripped:
        return stripped

    try:
        lang = detect(stripped)
    except Exception:
        lang = "ko"

    if lang.startswith("ko"):
        return normalize_korean_text(stripped)
    return normalize_english_text(stripped)


def normalize_korean_text(text: str) -> str:
    # 자모 분리/결합 문제를 간단히 정리하고, 특수문자 노이즈를 제거한다.
    cleaned = re.sub(r"[^0-9A-Za-z가-힣.,;:()\-\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_english_text(text: str) -> str:
    # 영어 OCR 결과는 ASCII 문자 위주로 정리하고 다중 공백을 축소한다.
    cleaned = re.sub(r"[^0-9A-Za-z.,;:()'\-\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def process_pdf(pdf_path, source_type):
    print(f"\n========== Processing: {pdf_path.name} ==========\n")

    pages = load_pdf_pages(str(pdf_path), source_type)
    page_texts = [p.page_content for p in pages]

    headers, footers = detect_repeating_headers_footers(page_texts)

    base_meta = infer_pdf_metadata(pdf_path, source_type)

    cleaned_pages = []
    ocr_targets = []
    qc_events = []
    for i, p in enumerate(pages):
        raw_text = p.page_content
        page_num = i + 1

        skip, reason = should_skip_page(raw_text, page_num)
        if skip:
            qc_events.append((page_num, "skip", reason))
            continue

        # 헤더/푸터 제거
        cleaned = strip_header_footer(raw_text, headers, footers)
        cleaned = clean_text_basic(cleaned)
        if not cleaned:
            if page_needs_ocr(raw_text):
                ocr_targets.append(page_num)
                qc_events.append((page_num, "ocr_candidate", "low_text"))
            else:
                qc_events.append((page_num, "drop", "clean_failed"))
            continue

        # 메타데이터 추가
        p.page_content = cleaned
        metadata = p.metadata or {}
        metadata.update({
            "source_file": pdf_path.name,
            "source_type": source_type,
            "page": page_num,
        })
        metadata.update(base_meta)
        p.metadata = metadata
        cleaned_pages.append(p)

    # 이미지 OCR 추가 (텍스트가 부족한 페이지만)
    ocr_list = extract_images_from_pdf(str(pdf_path), target_pages=ocr_targets)
    ocr_docs = []
    for page_num, text in ocr_list:
        cleaned = clean_text_basic(text)
        if cleaned:
            ocr_meta = {
                "source_file": pdf_path.name,
                "source_type": source_type,
                "ocr": True,
                "page": page_num,
            }
            ocr_meta.update(base_meta)
            ocr_docs.append(
                Document(
                    page_content=cleaned,
                    metadata=ocr_meta,
                )
            )

    # chunking
    chunks = text_splitter.split_documents(cleaned_pages)
    # 같은 페이지에서 나온 중복 청크는 build 단계 전에 정리한다.
    chunks = deduplicate_chunks(chunks)

    # OCR chunk (수동 구조)
    for od in ocr_docs:
        assign_chunk_id(od)
        chunks.append(od)

    return chunks, headers, footers, qc_events


# -------------------------------------------------------
# 5. 전체 PDF ingest
# -------------------------------------------------------
def assign_chunk_id(doc: Document) -> str:
    payload = "|".join(
        [
            str(doc.metadata.get("source_file")),
            str(doc.metadata.get("page")),
            doc.page_content.strip(),
        ]
    )
    chunk_id = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    doc.metadata["chunk_id"] = chunk_id
    return chunk_id


def deduplicate_chunks(docs: Iterable[Document]):
    """같은 chunk_id가 반복되면 첫 청크만 유지."""

    unique = []
    seen_ids = set()
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        chunk_id = assign_chunk_id(doc)
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique.append(doc)
    return unique


def load_existing_chunk_ids(persist_dir: Path):
    if not persist_dir.exists():
        return set(), None
    vectordb = Chroma(
        persist_directory=str(persist_dir),
        collection_name="esg_all",
        embedding_function=embedding_model,
    )
    existing = vectordb.get(include=["metadatas"])
    chunk_ids = set()
    for meta in existing.get("metadatas", []) or []:
        chunk_id = meta.get("chunk_id") if meta else None
        if chunk_id:
            chunk_ids.add(chunk_id)
    return chunk_ids, vectordb


def build_vector_db(clear_existing: bool = False):
    persist_dir = Path(VECTOR_DIR)
    if clear_existing and persist_dir.exists():
        print(f"[VectorDB] 기존 저장소 삭제 → {persist_dir}")
        shutil.rmtree(persist_dir)

    existing_ids, vectordb = load_existing_chunk_ids(persist_dir)
    new_chunks = []

    for folder in ["domestic", "global", "companies"]:
        path = DATA_DIR / folder
        if not path.exists():
            continue

        for pdf_file in path.glob("*.pdf"):
            chunks, headers, footers, qc_events = process_pdf(pdf_file, folder)
            for doc in chunks:
                chunk_id = doc.metadata.get("chunk_id") or assign_chunk_id(doc)
                if chunk_id in existing_ids:
                    continue
                existing_ids.add(chunk_id)
                new_chunks.append(doc)

            # ---- 샘플 QC 출력 ----
            # print("\n[QC] 헤더 패턴 탐지 결과:")
            # print(headers)
            # print("[QC] 푸터 패턴 탐지 결과:")
            # print(footers)

            # print("[QC] 페이지 처리 결과 (앞 5개):")
            # for event in qc_events[:5]:
            #     page_no, status, reason = event
            #     print(f"  - page {page_no}: {status} ({reason})")

            # print(f"\n[QC] 샘플 Chunk 출력 (앞 2개):")
            # for c in chunks[:2]:
            #     print("\n----- CHUNK SAMPLE -----")
            #     print(c.page_content[:400])
            #     print(c.metadata)

    if not new_chunks:
        print("\n⚠️  추가할 신규 청크가 없습니다. 기존 VectorDB를 유지합니다.\n")
        return

    if vectordb is None:
        vectordb = Chroma.from_documents(
            documents=new_chunks,
            embedding=embedding_model,
            persist_directory=VECTOR_DIR,
            collection_name="esg_all",
        )
    else:
        vectordb.add_documents(new_chunks)

    vectordb.persist()
    print(f"\n🚀 VectorDB 업데이트 완료 (신규 청크 {len(new_chunks)}개) → {VECTOR_DIR}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESG VectorDB builder")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="기존 vector_db/esg_all 디렉터리를 삭제한 뒤 전체 재구축",
    )
    args = parser.parse_args()

    build_vector_db(clear_existing=args.clear)
