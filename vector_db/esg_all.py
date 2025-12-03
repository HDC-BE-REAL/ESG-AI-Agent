from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


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
def load_pdf_pages(pdf_path, source_type):
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

            ocr_text = pytesseract.image_to_string(pil_img, lang="kor+eng")
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

    # OCR chunk (수동 구조)
    for od in ocr_docs:
        chunks.append(od)

    return chunks, headers, footers, qc_events


# -------------------------------------------------------
# 5. 전체 PDF ingest
# -------------------------------------------------------
def build_vector_db():
    all_chunks = []

    for folder in ["domestic", "global", "companies"]:
        path = DATA_DIR / folder
        if not path.exists():
            continue

        for pdf_file in path.glob("*.pdf"):
            chunks, headers, footers, qc_events = process_pdf(pdf_file, folder)
            all_chunks.extend(chunks)

            # ---- 샘플 QC 출력 ----
            print("\n[QC] 헤더 패턴 탐지 결과:")
            print(headers)
            print("[QC] 푸터 패턴 탐지 결과:")
            print(footers)

            print("[QC] 페이지 처리 결과 (앞 5개):")
            for event in qc_events[:5]:
                page_no, status, reason = event
                print(f"  - page {page_no}: {status} ({reason})")

            print(f"\n[QC] 샘플 Chunk 출력 (앞 2개):")
            for c in chunks[:2]:
                print("\n----- CHUNK SAMPLE -----")
                print(c.page_content[:400])
                print(c.metadata)

    # 저장
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DIR,
        collection_name="esg_all"
    )

    vectordb.persist()
    print(f"\n🚀 VectorDB 저장 완료 → {VECTOR_DIR}\n")


if __name__ == "__main__":
    build_vector_db()
