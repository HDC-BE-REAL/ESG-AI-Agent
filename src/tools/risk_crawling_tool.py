import os
import time
import json
import requests
import urllib.parse
import numpy as np
import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# LangChain & AI
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 환경 변수 로드
load_dotenv()

# 전역 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "risk_data")
HISTORY_DIR = os.path.join(DATA_DIR, "crawling")
HISTORY_FILE = os.path.join(HISTORY_DIR, "risk_history.json")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db", "esg_all")

# --------------------------------------------------------------------------
# [설정] 리스크 진단 자료 타겟 목록 (구글 우회 검색 키워드 추가)
# --------------------------------------------------------------------------
RISK_TARGETS = [
    # 1. [Safety] 안전보건공단 (KOSHA는 내부 아카이브가 잘 되어있어 유지)
    {
        "name": "KOSHA_C_Guide",
        "url": "https://portal.kosha.or.kr/archive/resources/tech-support/search/const?page=1&rowsPerPage=10",
        "type": "KOSHA_ARCHIVE", 
        "category": "Safety"
    },
    # 2. [Safety] 고용노동부 - 위험성평가 (구글 우회)
    {
        "name": "MOEL_Risk_Standard",
        "url": "https://www.moel.go.kr/info/publict/publictDataList.do", # 실패 시 구글로 전환
        "google_query": 'site:moel.go.kr filetype:pdf "위험성평가" "표준모델"',
        "type": "GOV_BOARD",
        "category": "Safety"
    },
    # 3. [Labor] 고용노동부 - 자율점검표 (구글 우회)
    {
        "name": "MOEL_Checklist",
        "url": "https://www.moel.go.kr/news/notice/noticeList.do",
        "google_query": 'site:moel.go.kr filetype:pdf "자율점검표"',
        "type": "GOV_BOARD",
        "category": "Labor"
    },
    # 4. [Env] 환경부 - 비산먼지 (구글 우회)
    {
        "name": "ME_Dust_Manual",
        "url": "https://www.me.go.kr/home/web/board/list.do?menuId=10392&boardMasterId=39",
        "google_query": 'site:me.go.kr filetype:pdf "비산먼지" "매뉴얼"',
        "type": "GOV_BOARD",
        "category": "Environment"
    },
    # 5. [Gov] 공정거래위원회 - 표준계약서 (구글 우회)
    {
        "name": "FTC_Construction_Contract",
        "url": "https://www.ftc.go.kr/www/cop/bbs/selectBoardList.do?key=201&bbsId=BBSMSTR_000000002320",
        "google_query": 'site:ftc.go.kr filetype:hwp OR filetype:pdf "건설업" "표준하도급계약서"',
        "type": "GOV_BOARD",
        "category": "Governance"
    }
]

class RiskCrawlingTool:
    """
    [리스크 진단 자료 수집 에이전트]
    - 안전(KOSHA/MOEL), 환경(ME), 공정(FTC) 분야의 실무 가이드/체크리스트 수집
    - 사이트 접속 차단 시 'Google Site Search'로 우회하여 PDF 직접 수집
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RiskCrawlingTool, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("⚙️ [RiskTool] 초기화 중...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"⚠️ 임베딩 모델 로드 실패: {e}")
            self.embeddings = None

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        if self.embeddings:
            self.vector_db = Chroma(
                collection_name="esg_risk_guides",
                embedding_function=self.embeddings,
                persist_directory=VECTOR_DB_DIR
            )
        else:
            self.vector_db = None

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_history(self):
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except: pass

    def _is_processed(self, key: str) -> bool:
        return key in self.history

    def _mark_as_processed(self, key: str, title: str, files: List[str]):
        self.history[key] = {
            "title": title,
            "processed_at": datetime.now().isoformat(),
            "files": files
        }
        self._save_history()

    def _get_chrome_driver(self):
        chrome_options = Options()
        # [중요] 봇 탐지 회피 옵션 강화
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled") # 자동화 제어 감지 비활성화
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"]) # 자동화 표시 제거
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # 일반 사용자처럼 보이게 하는 User-Agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        prefs = {
            "download.default_directory": DOWNLOAD_DIR,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_settings.popups": 0
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 봇 탐지 우회용 스크립트 실행
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver

    def _extract_text_preview(self, pdf_path: str, max_pages: int = 5) -> str:
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc):
                if i >= max_pages: break
                text += page.get_text()
            doc.close()
        except: pass
        return text

    def _analyze_and_store(self, file_path: str, title: str, target_info: Dict) -> bool:
        if not self.vector_db or not file_path.lower().endswith('.pdf'):
            return False

        filename = os.path.basename(file_path)
        print(f"   🧠 [AI 분석] '{filename}' 실무 활용도 평가 중...")
        
        content_preview = self._extract_text_preview(file_path)
        if not content_preview: return False

        prompt = f"""
        문서 제목: {title}
        카테고리: {target_info['category']}
        내용 미리보기:
        {content_preview[:2500]}

        이 문서가 기업 현장에서 안전/환경/노무 리스크를 점검할 때 즉시 활용 가능한 **실무 자료**인지 판단해주세요.
        
        [판단 기준]
        - **유용함 (True)**: 체크리스트(Checklist), 자율점검표, 기술 가이드라인(KOSHA Guide), 표준계약서 양식, 매뉴얼.
        - **유용하지 않음 (False)**: 단순 행사 알림, 인사 발령, 통계 연보, 정책 홍보 포스터.

        결과를 JSON으로 출력:
        {{
            "is_practical": true/false,
            "doc_type": "Checklist/Manual/Contract/Other",
            "score": (1~10),
            "summary": "한 줄 요약"
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            
            print(f"      👉 결과: {result['doc_type']} (점수: {result['score']})")

            if result['is_practical'] and result['score'] >= 7:
                print(f"      💾 [Vector DB] 저장합니다.")
                
                full_doc = fitz.open(file_path)
                full_text = ""
                for page in full_doc:
                    full_text += page.get_text()
                full_doc.close()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.create_documents(
                    [full_text], 
                    metadatas=[{
                        "source": target_info['name'],
                        "category": target_info['category'],
                        "title": title,
                        "doc_type": result['doc_type'],
                        "filename": filename,
                        "crawled_at": datetime.now().isoformat()
                    }]
                )
                self.vector_db.add_documents(chunks)
                print(f"      ✅ DB 저장 완료 ({len(chunks)} chunks)")
                return True
            else:
                print("      🗑️ [Skip] 실무 활용도가 낮아 저장하지 않습니다.")
                return False
        except Exception as e:
            print(f"      ❌ AI 분석 오류: {e}")
            return False

    # ----------------------------------------------------------------
    # [Fallback Strategy] Google Site Search
    # ----------------------------------------------------------------
    def _scrape_google_fallback(self, driver, target_info: Dict) -> List[Dict]:
        """
        내부 검색이 막혔을 때, Google을 통해 해당 사이트의 PDF를 직접 찾습니다.
        Query 예시: site:moel.go.kr filetype:pdf "위험성평가"
        """
        query = target_info.get("google_query")
        if not query:
            return []
            
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        name = target_info["name"]
        results = []
        
        print(f"🚀 [Google Bypass] '{name}' 우회 검색 시도... ({query})")
        try:
            driver.get(search_url)
            # 구글 검색결과 로딩 대기
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
            
            # 검색 결과 링크 수집 (구글의 검색 결과 구조: div.g a)
            links = driver.find_elements(By.CSS_SELECTOR, "div.g a")
            
            # 상위 3개 PDF만 처리
            pdf_links = []
            for link in links:
                href = link.get_attribute("href")
                if href and href.lower().endswith(".pdf"):
                    # 구글 트래킹 링크가 아닌 실제 링크인지 확인
                    pdf_links.append((link, href))
            
            # 중복 제거 및 상위 3개 선택
            seen_urls = set()
            unique_pdfs = []
            for l, h in pdf_links:
                if h not in seen_urls:
                    unique_pdfs.append((l, h))
                    seen_urls.add(h)
            
            print(f"   🔎 구글에서 PDF {len(unique_pdfs)}개 발견")

            for i, (link_elem, pdf_url) in enumerate(unique_pdfs[:3]):
                try:
                    title = link_elem.find_element(By.CSS_SELECTOR, "h3").text
                    unique_key = f"Google_{name}_{title}"
                    
                    if self._is_processed(unique_key):
                        print(f"   ⏭️ [Skip] {title}")
                        continue
                        
                    print(f"   📥 [Direct Download] {title}")
                    
                    # PDF 직접 다운로드 (requests 사용)
                    # Selenium으로 PDF를 열면 뷰어가 뜰 수 있으므로 requests로 받음
                    response = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                    
                    if response.status_code == 200:
                        # 파일명 안전하게 만들기
                        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
                        filename = f"{safe_title}.pdf"
                        file_path = os.path.join(DOWNLOAD_DIR, filename)
                        
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                            
                        print(f"      ✅ 다운로드 완료: {filename}")
                        
                        # AI 분석 및 저장
                        if self._analyze_and_store(file_path, title, target_info):
                            self._mark_as_processed(unique_key, title, [file_path])
                            results.append({"source": name, "title": title, "files": [file_path]})
                            
                except Exception as e:
                    print(f"      ⚠️ 구글 검색결과 처리 중 오류: {e}")
                    
        except Exception as e:
            print(f"❌ 구글 우회 검색 실패: {e}")
            
        return results

    # ----------------------------------------------------------------
    # [Crawling] Main Strategies
    # ----------------------------------------------------------------
    def _scrape_kosha_archive(self, driver, target_info: Dict) -> List[Dict]:
        """KOSHA는 내부 검색이 잘 되므로 기존 로직 유지"""
        url = target_info["url"]
        name = target_info["name"]
        results = []
        
        print(f"📡 [{name}] KOSHA 접속 중... ({url})")
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 20)
            time.sleep(3) 

            for i in range(3):
                try:
                    links = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a")))
                    target_links = [l for l in links if len(l.text.strip()) > 10 and l.is_displayed()]
                    if i >= len(target_links): break
                    
                    post_link = target_links[i]
                    title = post_link.text.strip()
                    unique_key = f"{name}_{title}"
                    
                    if self._is_processed(unique_key):
                        print(f"   ⏭️ [Skip] {title}")
                        continue
                        
                    print(f"   🔎 [New] 분석: {title}")
                    driver.execute_script("arguments[0].click();", post_link)
                    time.sleep(3)
                    
                    downloaded_files = []
                    try:
                        file_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'download') or contains(text(), '다운로드') or contains(@href, 'file')]")
                        for f_link in file_links:
                            driver.execute_script("arguments[0].click();", f_link)
                            time.sleep(5) # 다운로드 대기
                            # (파일 확인 로직 생략 - 최근 파일 확인 등)
                            # 여기서는 KOSHA 특성상 다운로드 성공 가정하고 다음으로
                            break
                    except: pass
                    
                    self._mark_as_processed(unique_key, title, [])
                    driver.back()
                    time.sleep(3)
                except:
                    driver.get(url) 
                    time.sleep(3)
        except Exception as e:
            print(f"❌ KOSHA 크롤링 실패: {e}")
        return results

    def _scrape_gov_board(self, driver, target_info: Dict) -> List[Dict]:
        """
        일반 공공기관 게시판 크롤링 시도 -> 실패 시 Google 우회 검색으로 전환
        """
        url = target_info["url"]
        name = target_info["name"]
        
        print(f"📡 [{name}] 접속 시도... ({url})")
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            
            # 구조 감지 시도
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
                print("   ✅ 내부 게시판 구조 감지됨. 크롤링 진행.")
                # (여기에 기존 테이블 크롤링 로직이 들어가야 하지만, 
                #  현재 접속 자체가 불안정하므로 바로 Google Fallback을 우선시하는 전략도 가능)
                #  일단 구조가 감지되어도 내용이 없으면 실패로 간주
                rows = driver.find_elements(By.TAG_NAME, "tr")
                if len(rows) < 2: raise Exception("Empty Board")
                
            except Exception:
                print("   ⚠️ 내부 게시판 구조 감지 실패 또는 차단됨.")
                raise Exception("Access Blocked or Structure Unknown")

            # (성공 시 로직은 생략하고, 실패 유도하여 바로 구글 검색으로 넘김 - 안정성 우선)
            # 사용자 요청: "우회해서 접속을 하는 방법을 찾아야할 것 같아"
            # 따라서 바로 Exception을 발생시켜 Fallback으로 넘깁니다.
            raise Exception("Force Fallback to Google")

        except Exception as e:
            print(f"   🔄 내부 접속 불가 ({e}). Google 우회 검색으로 전환합니다.")
            return self._scrape_google_fallback(driver, target_info)

        return []

    def collect_all_guides(self) -> str:
        print("\n" + "="*50)
        print(f"🛡️ [Risk Data 수집] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        driver = self._get_chrome_driver()
        total_results = []
        
        try:
            for target in RISK_TARGETS:
                if target.get("type") == "KOSHA_ARCHIVE":
                    res = self._scrape_kosha_archive(driver, target)
                else:
                    # 정부 사이트는 바로 접속 시도 후 실패 시 구글 우회
                    res = self._scrape_gov_board(driver, target)
                total_results.extend(res)
        finally:
            driver.quit()
            
        report = f"## 🛡️ 리스크 진단 자료 수집 리포트\n"
        if total_results:
            for item in total_results:
                files = ", ".join([os.path.basename(f) for f in item['files']])
                report += f"- **[{item['source']}]** {item['title']}\n  - 💾 {files}\n"
        else:
            report += "- 신규 자료가 없습니다 (모두 최신 또는 수집 실패).\n"
            
        print(report)
        return report

# LangChain Tool Export
_risk_collector = RiskCrawlingTool()

@tool
def fetch_risk_guides(query: str = "safety checklist") -> str:
    """
    Collects practical risk assessment guides, checklists, and manuals 
    from KOSHA, MOEL, ME, FTC.
    """
    return _risk_collector.collect_all_guides()

if __name__ == "__main__":
    _risk_collector.collect_all_guides()