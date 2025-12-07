import os
import time
import json
import schedule
import requests
import numpy as np
import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Selenium (브라우저 제어용)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# LangChain & AI
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# 1. 환경 변수 로드
load_dotenv()

# 전역 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "domestic")
HISTORY_DIR = os.path.join(DATA_DIR, "crawling")
HISTORY_FILE = os.path.join(HISTORY_DIR, "crawl_history.json")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "chroma_db")  # 벡터DB 저장 경로

# [변경] 모니터링 타겟 목록
# law.go.kr은 별도 로직으로 처리하기 위해 type을 구분하거나 URL로 식별
MINISTRY_TARGETS = [
    {
        "name": "환경부(국가법령센터)",
        "url": "https://www.law.go.kr/nwRvsLsPop.do?cptOfi=1482000",
        "type": "LAW_GO_KR",  # 전용 타입 지정
        "page_param": None
    },
    {
        "name": "고용노동부(MOEL)",
        "url": "https://www.moel.go.kr/info/lawinfo/lawmaking/list.do", 
        "type": "GENERIC_BOARD",
        "page_param": "pageIndex"
    },
    {
        "name": "국토교통부(MOLIT)",
        "url": "http://www.molit.go.kr/USR/LEGAL/m_35/lst.jsp",        
        "type": "GENERIC_BOARD",
        "page_param": "page"
    }
]

# [변경] 신뢰할 수 있는 뉴스 소스 도메인 목록
TRUSTED_NEWS_DOMAINS = [
    "yna.co.kr",       # 연합뉴스
    "mk.co.kr",        # 매일경제
    "hankyung.com",    # 한국경제
    "sedaily.com",     # 서울경제
    "lawtimes.co.kr",  # 법률신문
    "korea.kr",        # 대한민국 정책브리핑
    "chosun.com",      # 조선일보
    "joongang.co.kr",  # 중앙일보
    "donga.com",       # 동아일보
    "khan.co.kr",      # 경향신문
    "etnews.com",      # 전자신문
    "mt.co.kr",        # 머니투데이
    "me.go.kr",        # 환경부
    "motie.go.kr",     # 산업통상자원부
    "fsc.go.kr"        # 금융위원회
]

class RegulationMonitor:
    """
    [규제 모니터링 엔진 - AI Enhanced]
    1. Selenium으로 보고서 및 법령안 자동 다운로드 (금융위/GMI + 환경/국토/노동부)
    2. 국가법령정보센터(law.go.kr) 전용 크롤러 탑재 (텍스트 추출 -> 파일 저장)
    3. GPT-4o를 이용해 문서의 중요도 평가 및 선별 (Filtering)
    4. 선별된 중요 문서만 Vector DB에 자동 저장 (RAG 준비)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RegulationMonitor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("⚙️ [RegulationMonitor] 초기화 중...")
        
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
                collection_name="esg_regulations",
                embedding_function=self.embeddings,
                persist_directory=VECTOR_DB_DIR
            )
        else:
            self.vector_db = None

        self.tavily = TavilySearchResults(
            max_results=5,
            include_domains=TRUSTED_NEWS_DOMAINS
        )
        
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_history(self):
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 히스토리 저장 실패: {e}")

    def _is_processed(self, url: str) -> bool:
        return url in self.history

    def _mark_as_processed(self, url: str, title: str, files: List[str]):
        self.history[url] = {
            "title": title,
            "processed_at": datetime.now().isoformat(),
            "files": files
        }
        self._save_history()

    def _extract_text_preview(self, file_path: str, max_pages: int = 3) -> str:
        """파일 내용 프리뷰 추출 (PDF 및 TXT 지원)"""
        text_preview = ""
        try:
            if file_path.lower().endswith('.pdf'):
                doc = fitz.open(file_path)
                for i, page in enumerate(doc):
                    if i >= max_pages: break
                    text_preview += page.get_text()
                doc.close()
            elif file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_preview = f.read(3000) # 앞부분 3000자
            else:
                text_preview = "(지원되지 않는 파일 형식입니다)"
        except Exception as e:
            print(f"⚠️ 파일 읽기 실패 ({os.path.basename(file_path)}): {e}")
        return text_preview

    def _analyze_and_store(self, file_path: str, title: str, source: str) -> bool:
        if not self.vector_db:
            return False

        filename = os.path.basename(file_path)
        print(f"   🧠 [AI 분석] '{filename}' 중요도 평가 중...")

        content_preview = self._extract_text_preview(file_path)
        if not content_preview:
            return False

        prompt = f"""
        당신은 ESG 및 산업 안전, 환경 규제 전문가입니다. 
        출처: '{source}'
        문서 제목: '{title}'
        내용 미리보기:
        {content_preview[:2000]}

        이 문서가 기업의 ESG 경영, 환경 규제 준수, 산업 안전, 혹은 컴플라이언스에 영향을 미치는 **중요한 법령/가이드라인**인지 판단해주세요.
        
        [판단 기준]
        - 중요(High): 법률/시행령 개정안, 입법예고, 처벌 기준 강화, 공시 의무화 가이드라인
        - 보통(Medium): 단순 실태조사 결과, 캠페인성 알림, 인사 발령
        - 낮음(Low): 행사/세미나 알림, 단순 홍보물

        결과를 JSON 형식으로 출력:
        {{
            "is_important": true/false,
            "score": (1~10),
            "reason": "한 줄 요약",
            "category": "법령개정/가이드라인/단순알림"
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(response_text)
            
            is_important = analysis.get("is_important", False)
            score = analysis.get("score", 0)
            
            print(f"      👉 결과: 중요도 {score}점 ({analysis.get('reason')})")

            if is_important and score >= 6:
                print(f"      💾 [Vector DB] 중요 문서로 식별되어 DB에 저장합니다.")
                
                full_text = ""
                # PDF 처리
                if file_path.lower().endswith('.pdf'):
                    full_doc = fitz.open(file_path)
                    for page in full_doc:
                        full_text += page.get_text()
                    full_doc.close()
                # TXT 처리 (law.go.kr 등)
                elif file_path.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                
                if full_text:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.create_documents(
                        [full_text], 
                        metadatas=[{
                            "source": source,
                            "title": title,
                            "filename": filename,
                            "category": analysis.get("category", "Uncategorized"),
                            "crawled_at": datetime.now().isoformat()
                        }]
                    )
                    self.vector_db.add_documents(chunks)
                    print(f"      ✅ DB 저장 완료 ({len(chunks)} chunks)")
                return True
            else:
                print(f"      🗑️ [Discard] 중요도가 낮아 DB에 저장하지 않습니다.")
                return False

        except Exception as e:
            print(f"      ❌ AI 분석 중 오류: {e}")
            return False

    def _get_chrome_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new") 
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
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
        return driver

    def _fetch_law_go_kr(self, driver, target_info: Dict) -> List[Dict]:
        """
        [전용] 국가법령정보센터(law.go.kr) 크롤러
        - 구조: 리스트 -> 클릭 -> 본문 텍스트 뷰어 (첨부파일 다운로드가 까다로움)
        - 전략: 본문 텍스트를 추출하여 .txt 파일로 저장
        """
        url = target_info["url"]
        source_name = target_info["name"]
        results = []

        print(f"📡 [{source_name}] 접속 중... ({url})")
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 15)
            # law.go.kr 리스트 테이블 대기 (tbody)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
            
            # 상위 3개 항목
            for i in range(3):
                try:
                    row_index = i + 1
                    # 제목 링크 찾기 (보통 2번째 td의 a 태그, 혹은 text align left)
                    # law.go.kr은 구조가 가변적이라 tr 내부의 'a' 태그 중 텍스트가 있는 것을 찾음
                    row = wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, f"tbody tr:nth-child({row_index})")
                    ))
                    links = row.find_elements(By.TAG_NAME, "a")
                    
                    target_link = None
                    title = ""
                    for link in links:
                        text = link.text.strip()
                        if text and len(text) > 5: # 제목일 가능성이 높은 링크
                            target_link = link
                            title = text
                            break
                    
                    if not target_link: continue

                    unique_key = f"{source_name}_{title}"
                    if self._is_processed(unique_key):
                        print(f"   ⏭️ [Skip] {source_name}: {title}")
                        continue

                    print(f"   🔎 [New] {source_name} 분석: {title}")
                    
                    # 상세 페이지 진입 (law.go.kr은 클릭 시 페이지 이동/AJAX 로딩)
                    driver.execute_script("arguments[0].click();", target_link)
                    time.sleep(3) # 로딩 대기
                    
                    # 본문 텍스트 추출 시도 (법령 본문 영역)
                    # law.go.kr 본문 ID 후보: contentBody, conScroll, viewArea 등
                    content_text = ""
                    try:
                        # 여러 선택자 시도
                        body_elem = None
                        for selector in ["#contentBody", ".lawCon", "#conScroll", "body"]:
                            try:
                                body_elem = driver.find_element(By.CSS_SELECTOR, selector)
                                if len(body_elem.text) > 100:
                                    break
                            except: continue
                        
                        if body_elem:
                            content_text = body_elem.text
                    except Exception as e:
                        print(f"      ⚠️ 본문 추출 실패: {e}")

                    downloaded_files = []
                    if content_text:
                        # 텍스트 파일로 저장
                        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
                        file_name = f"{safe_title}.txt"
                        file_path = os.path.join(DOWNLOAD_DIR, file_name)
                        
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(f"제목: {title}\n출처: {url}\n\n{content_text}")
                        
                        print(f"      ✅ 본문 텍스트 저장 완료: {file_name}")
                        downloaded_files.append(file_path)
                        
                        # AI 분석 및 저장
                        self._analyze_and_store(file_path, title, source_name)

                    self._mark_as_processed(unique_key, title, downloaded_files)
                    results.append({"source": source_name, "title": title, "files": downloaded_files})
                    
                    # 목록으로 돌아가기 (뒤로가기 혹은 URL 재접속)
                    driver.get(url)
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "tbody")))
                    
                except Exception as e:
                    print(f"      ⚠️ 게시글 처리 중 오류: {e}")
                    driver.get(url)
                    time.sleep(2)

        except Exception as e:
            print(f"❌ [{source_name}] 크롤링 실패: {e}")
            
        return results

    def _scrape_generic_board(self, driver, target_info: Dict) -> List[Dict]:
        """[공통] 일반 게시판 크롤링"""
        base_url = target_info["url"]
        source_name = target_info["name"]
        page_param = target_info.get("page_param")
        results = []

        max_pages = 3 if page_param else 1
        
        for page in range(1, max_pages + 1):
            if page_param:
                sep = "&" if "?" in base_url else "?"
                target_url = f"{base_url}{sep}{page_param}={page}"
            else:
                target_url = base_url

            print(f"📡 [{source_name}] 접속 중 (Page {page})...")
            try:
                driver.get(target_url)
                wait = WebDriverWait(driver, 15)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
                
                for i in range(3):
                    try:
                        row_index = i + 1
                        # 일반적인 게시판: n번째 행의 제목 링크 찾기
                        # 구조가 다양하므로, 행 내부에서 가장 긴 텍스트를 가진 a태그를 제목으로 추정
                        row = wait.until(EC.presence_of_element_located(
                            (By.CSS_SELECTOR, f"table tbody tr:nth-child({row_index})")
                        ))
                        links = row.find_elements(By.TAG_NAME, "a")
                        
                        post_link = None
                        title = ""
                        for link in links:
                            text = link.text.strip()
                            if len(text) > 5: # 제목일 가능성
                                post_link = link
                                title = text
                                break
                        
                        if not post_link: continue
                        
                        unique_key = f"{source_name}_{title}"
                        
                        if self._is_processed(unique_key):
                            print(f"   ⏭️ [Skip] {source_name}: {title}")
                            continue
                            
                        print(f"   🔎 [New] {source_name} 분석: {title}")
                        
                        driver.execute_script("arguments[0].click();", post_link)
                        time.sleep(2)
                        
                        downloaded_files = []
                        potential_links = driver.find_elements(By.TAG_NAME, "a")
                        file_links = []
                        for link in potential_links:
                            href = link.get_attribute("href")
                            text = link.text.strip()
                            if href and ("down" in href.lower() or "file" in href.lower() or "download" in href.lower()) and any(ext in text.lower() for ext in ['.pdf', '.hwp', '.doc']):
                                file_links.append(link)
                        
                        for link in file_links[:1]:
                            f_name = link.text.strip()
                            print(f"      📥 다운로드 시도: {f_name}")
                            before_files = set(os.listdir(DOWNLOAD_DIR))
                            driver.execute_script("arguments[0].click();", link)
                            
                            for _ in range(10):
                                time.sleep(1)
                                new_files = set(os.listdir(DOWNLOAD_DIR)) - before_files
                                if new_files:
                                    new_file = list(new_files)[0]
                                    if not new_file.endswith('.crdownload'):
                                        full_path = os.path.join(DOWNLOAD_DIR, new_file)
                                        downloaded_files.append(full_path)
                                        print(f"      ✅ 다운로드 완료: {new_file}")
                                        self._analyze_and_store(full_path, title, source_name)
                                        break
                        
                        self._mark_as_processed(unique_key, title, downloaded_files)
                        results.append({"source": source_name, "title": title, "files": downloaded_files})
                        
                        driver.back()
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"      ⚠️ 게시글 처리 중 스킵: {e}")
                        if target_url not in driver.current_url:
                            driver.back()
                            time.sleep(1)

            except Exception as e:
                print(f"❌ [{source_name}] Page {page} 크롤링 실패: {e}")
                
        return results

    def _fetch_gmi_reports_selenium(self) -> List[Dict]:
        target_url = "https://www.gmi.go.kr/np/boardList.do?menuCd=2090&seCd=2"
        results = []
        
        print(f"📡 [GMI] 접속 및 스캔 시작 ({target_url})")
        driver = self._get_chrome_driver()
        
        try:
            driver.get(target_url)
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            
            for i in range(3):
                try:
                    row_index = i + 1
                    post_link = wait.until(EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, f"table tbody tr:nth-child({row_index}) a")
                    ))
                    
                    title = post_link.text.strip() or driver.execute_script("return arguments[0].innerText;", post_link).strip()
                    unique_key = f"GMI_{title}"
                    
                    if self._is_processed(unique_key):
                        print(f"   ⏭️ [Skip] 이미 수집된 보고서: {title}")
                        continue
                        
                    print(f"   🔎 [New] 신규 보고서 분석: {title}")
                    driver.execute_script("arguments[0].click();", post_link)
                    time.sleep(2)
                    
                    downloaded_files = []
                    file_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='downloadAttach']")
                    if not file_links:
                        file_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='FileDown']")

                    for link in file_links:
                        f_name = link.text.strip() or driver.execute_script("return arguments[0].innerText;", link).strip()
                        if 'pdf' in f_name.lower():
                            print(f"      📥 다운로드 시도: {f_name}")
                            before_files = set(os.listdir(DOWNLOAD_DIR))
                            driver.execute_script("arguments[0].click();", link)
                            for _ in range(15):
                                time.sleep(1)
                                new_files = set(os.listdir(DOWNLOAD_DIR)) - before_files
                                if new_files:
                                    downloaded_file = list(new_files)[0]
                                    if not downloaded_file.endswith('.crdownload'):
                                        full_path = os.path.join(DOWNLOAD_DIR, downloaded_file)
                                        downloaded_files.append(full_path)
                                        print(f"      ✅ 다운로드 완료: {downloaded_file}")
                                        self._analyze_and_store(full_path, title, "GMI")
                                        break
                    
                    self._mark_as_processed(unique_key, title, downloaded_files)
                    results.append({"source": "GMI", "title": title, "files": downloaded_files})
                    driver.back()
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
                    time.sleep(1)
                except Exception as e:
                    print(f"      ⚠️ 게시글 처리 오류: {e}")
                    if "boardList.do" not in driver.current_url:
                        driver.back()
                        time.sleep(2)
        except Exception as e:
            print(f"❌ [GMI] 크롤링 실패: {e}")
        finally:
            driver.quit()
        return results

    def _fetch_fsc_reports_selenium(self) -> List[Dict]:
        base_url = "https://www.fsc.go.kr/no010101"
        results = []
        
        print(f"📡 [FSC] 접속 및 스캔 시작 (1~3 페이지 확인)")
        driver = self._get_chrome_driver()
        
        try:
            for page in range(1, 4):
                target_url = f"{base_url}?curPage={page}"
                print(f"   📄 FSC Page {page} 스캔 중...")
                
                driver.get(target_url)
                wait = WebDriverWait(driver, 20)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".board-list .subject a")))
                
                list_items = driver.find_elements(By.CSS_SELECTOR, ".board-list .subject a")
                keywords = ["ESG", "공시", "지속가능", "녹색", "기후", "택소노미"]
                
                target_items = []
                for item in list_items:
                    text = item.text.strip()
                    if any(k in text for k in keywords):
                        href = item.get_attribute("href")
                        target_items.append((text, href))
                
                for title, link in target_items:
                    if self._is_processed(link):
                        print(f"      ⏭️ [Skip] {title}")
                        continue
                    
                    print(f"      🔎 [New] 분석: {title}")
                    driver.get(link)
                    time.sleep(2)
                    
                    downloaded_files = []
                    file_links = driver.find_elements(By.CSS_SELECTOR, ".file-list a")
                    
                    for f_link in file_links:
                        f_name = f_link.text.strip()
                        if any(ext in f_name.lower() for ext in ['.pdf', '.hwp']):
                            print(f"         📥 다운로드 클릭: {f_name}")
                            before_files = set(os.listdir(DOWNLOAD_DIR))
                            f_link.click()
                            for _ in range(15):
                                time.sleep(1)
                                new_files = set(os.listdir(DOWNLOAD_DIR)) - before_files
                                if new_files:
                                    new_file = list(new_files)[0]
                                    if not new_file.endswith('.crdownload'):
                                        full_path = os.path.join(DOWNLOAD_DIR, new_file)
                                        downloaded_files.append(full_path)
                                        if new_file.lower().endswith('.pdf'):
                                            self._analyze_and_store(full_path, title, "FSC")
                                        break
                    
                    self._mark_as_processed(link, title, downloaded_files)
                    results.append({"source": "FSC", "title": title, "files": downloaded_files})
                    
                    driver.get(target_url)
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".board-list .subject a")))
                    
        except Exception as e:
            print(f"❌ [FSC] 크롤링 실패: {e}")
        finally:
            driver.quit()
            
        return results

    def _fetch_legal_updates(self) -> List[Dict]:
        results = []
        driver = self._get_chrome_driver()
        try:
            for target in MINISTRY_TARGETS:
                try:
                    # [변경] 사이트 타입에 따라 전용 크롤러 사용
                    if target.get("type") == "LAW_GO_KR":
                        site_results = self._fetch_law_go_kr(driver, target)
                    else:
                        site_results = self._scrape_generic_board(driver, target)
                    results.extend(site_results)
                except Exception as e:
                    print(f"❌ {target['name']} 처리 중 오류: {e}")
        finally:
            driver.quit()
        return results

    def _deduplicate_news(self, news_list: List[Dict], threshold=0.85) -> List[Dict]:
        if not news_list or not self.embeddings: return news_list
        texts = [item['content'] for item in news_list]
        vectors = self.embeddings.embed_documents(texts)
        matrix = np.array(vectors)
        processed = [False] * len(news_list)
        unique_news = []
        for i in range(len(news_list)):
            if processed[i]: continue
            current_cluster = [news_list[i]]
            processed[i] = True
            vec_i = matrix[i].reshape(1, -1)
            if i + 1 < len(news_list):
                remaining_vectors = matrix[i+1:]
                similarities = cosine_similarity(vec_i, remaining_vectors)[0]
                for idx, score in enumerate(similarities):
                    real_idx = i + 1 + idx
                    if not processed[real_idx] and score >= threshold:
                        current_cluster.append(news_list[real_idx])
                        processed[real_idx] = True
            representative = max(current_cluster, key=lambda x: len(x['content']))
            representative['related_count'] = len(current_cluster) - 1
            unique_news.append(representative)
        return unique_news

    def monitor_all(self, query: str = "ESG 규제 동향") -> str:
        print("\n" + "="*50)
        print(f"🔄 [모니터링 실행] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

        # 1. 보고서 수집 (GMI, FSC)
        gmi_reports = self._fetch_gmi_reports_selenium()
        fsc_reports = self._fetch_fsc_reports_selenium()
        
        # 2. 법령 업데이트 수집
        legal_updates = self._fetch_legal_updates()
        
        reports = gmi_reports + fsc_reports + legal_updates
        
        # 3. 뉴스 검색
        news_results = []
        if os.getenv("TAVILY_API_KEY"):
            queries = list(set([query, "ESG 공시 의무화", "환경부 입법예고", "중대재해처벌법 개정"]))
            for q in queries:
                try:
                    raw = self.tavily.invoke(q)
                    for item in raw:
                        news_results.append({
                            "title": item['content'][:30] + "...", 
                            "content": item['content'],
                            "url": item['url'],
                            "source": "Web News"
                        })
                except Exception as e:
                    print(f"⚠️ Tavily 검색 실패 ({q}): {e}")
        
        clean_news = self._deduplicate_news(news_results)
        
        # 결과 포맷팅
        result_str = f"## 🌍 ESG 규제 & 법령 모니터링 리포트 ({time.strftime('%Y-%m-%d')})\n\n"
        
        if reports:
            result_str += "### 🆕 신규 보고서 및 법령 개정안\n"
            for r in reports:
                files_msg = ", ".join([os.path.basename(f) for f in r['files']]) if r['files'] else "파일 없음"
                result_str += f"- **[{r['source']}]** {r['title']}\n"
                result_str += f"  - 💾 다운로드: `{files_msg}`\n"
        else:
            result_str += "### 🆕 신규 보고서 및 법령 개정안\n"
            result_str += "- 새롭게 변경된 정책이 없습니다.\n"
            
        result_str += "\n### 📰 주요 뉴스 및 입법 동향 (AI 요약)\n"
        if clean_news:
            # 상위 3개 뉴스만 요약
            top_news = clean_news[:3]
            for i, n in enumerate(top_news):
                print(f"   🤖 [AI 요약] 뉴스 {i+1}/{len(top_news)} 요약 중...")
                try:
                    prompt = f"""
                    다음 뉴스 기사를 한국어로 3줄 요약해주세요. 핵심 내용 위주로 간결하게 작성하세요.
                    
                    기사 내용: {n['content']}
                    """
                    summary_res = self.llm.invoke(prompt)
                    summary = summary_res.content.strip()
                    
                    result_str += f"**{i+1}. {n['title']}**\n"
                    result_str += f"{summary}\n"
                    result_str += f"🔗 [원문 보기]({n['url']})\n\n"
                except Exception as e:
                    print(f"      ⚠️ 요약 실패: {e}")
                    result_str += f"- {n['content'][:100]}...\n  🔗 [기사]({n['url']})\n"
        else:
            result_str += "- 관련 주요 뉴스가 없습니다.\n"
        
        print(result_str)
        return result_str

# LangChain Tool Export
_monitor_instance = RegulationMonitor()

@tool
def fetch_regulation_updates(query: str = "ESG regulatory updates") -> str:
    """
    Monitors ESG updates using Selenium and History Tracking to detect NEW reports only.
    Use GPT to filter important documents and store them in Vector DB.
    """
    return _monitor_instance.monitor_all(query)

def run_continuously(interval_days: int = 1):
    print(f"\n⏰ 스케줄러 시작: {interval_days}일마다 자동 실행됩니다.")
    _monitor_instance.monitor_all()
    schedule.every(interval_days).days.do(_monitor_instance.monitor_all)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    # [Mode 1] 단순 테스트 모드
    print("🧪 [Test Mode] 1회 크롤링 및 분석 실행...")
    _monitor_instance.monitor_all()

    # [Mode 2] 백그라운드 스케줄러 모드
    # run_continuously(interval_days=1)