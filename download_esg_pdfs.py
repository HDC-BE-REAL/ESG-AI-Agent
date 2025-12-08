import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# PDF 저장 폴더
SAVE_DIR = "data/pdf"
os.makedirs(SAVE_DIR, exist_ok=True)

# 수집 대상 URL
TARGET_URLS = {
    "k_esg": "https://check.esgi.or.kr/contents/esgGuide/",
    "gri": "https://www.globalreporting.org/standards/",
    "sasb": "https://sasb.org/standards/",
    "issb": "https://www.ifrs.org/issued-standards/list-of-standards/",
    "ungc": "https://www.unglobalcompact.org/what-is-gc/mission/principles",
    "oecd": "https://mneguidelines.oecd.org/oecd-due-diligence-guidance-for-responsible-business-conduct.htm"
}

def find_pdfs(url):
    """해당 URL 페이지에서 PDF 링크 추출"""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] URL 접속 실패: {url} → {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if ".pdf" in href.lower():
            pdf_url = urljoin(url, href)
            pdf_links.append(pdf_url)

    return list(set(pdf_links))


def download_pdf(url):
    """PDF 다운로드"""
    filename = url.split("/")[-1].split("?")[0]
    save_path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(save_path):
        print(f"[SKIP] 이미 존재함: {filename}")
        return

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[OK] 다운로드 완료: {filename}")
    except Exception as e:
        print(f"[ERROR] 다운로드 실패: {url} → {e}")


def main():
    print("=== ESG PDF 자동 스크래핑 및 다운로드 시작 ===\n")

    for name, url in TARGET_URLS.items():
        print(f"\n🔍 {name} PDF 탐색 중 → {url}")

        pdfs = find_pdfs(url)

        if not pdfs:
            print(f"❗ PDF를 찾지 못함: {name}")
            continue

        print(f"→ {len(pdfs)}개 PDF 발견")

        for pdf in pdfs:
            download_pdf(pdf)

    print("\n=== 모든 다운로드 완료 ===")


if __name__ == "__main__":
    main()
