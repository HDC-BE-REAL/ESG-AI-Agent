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
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, ElementClickInterceptedException
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
# [설정] 리스크 진단 자료 타겟 목록
# --------------------------------------------------------------------------
RISK_TARGETS = [
    # 1. [ESG Hub] ESG 금융 추진단 (E/S/G 태그 수집)
    {
        "name": "ESG_Finance_Hub",
        "url": "https://www.esgfinancehub.or.kr/portal/report/imgList/vw/20211222092216000024",
        "type": "ESG_HUB", 
        "category": "ESG_General"
    },
    # 2. [Safety] 안전보건공단 자료마당
    {
        "name": "KOSHA_Construction_Guide",
        "url": "https://portal.kosha.or.kr/archive/resources/tech-support/search/const?page=1&rowsPerPage=10",
        "type": "KOSHA_ARCHIVE", 
        "category": "Safety"
    },
    # 3. [Safety] 고용노동부 - 위험성평가 (Google 우회)
    {
        "name": "MOEL_Risk_Standard",
        "url": "https://www.moel.go.kr/info/publict/publictDataList.do", 
        "google_query": 'site:moel.go.kr filetype:pdf "위험성평가" "표준모델"',
        "type": "GOV_BOARD",
        "category": "Safety"
    },
    # 4. [Labor] 고용노동부 - 자율점검표 (Google 우회)
    {
        "name": "MOEL_Checklist",
        "url": "https://www.moel.go.kr/news/notice/noticeList.do",
        "google_query": 'site:moel.go.kr filetype:pdf "자율점검표"',
        "type": "GOV_BOARD",
        "category": "Labor"
    },
    # 5. [Env] 환경부 - 비산먼지 (Google 우회)
    {
        "name": "ME_Dust_Manual",
        "url": "https://www.me.go.kr/home/web/board/list.do?menuId=10392&boardMasterId=39",
        "google_query": 'site:me.go.kr filetype:pdf "비산먼지" "매뉴얼"',
        "type": "GOV_BOARD",
        "category": "Environment"
    },
    # 6. [Gov] 공정거래위원회 - 표준계약서 (Google 우회)
    {
        "name": "FTC_Construction_Contract",
        "url": "https://www.ftc.go.kr/www/cop/bbs/selectBoardList.do?key=201&bbsId=BBSMSTR_000000002320",
        "google_query": 'site:ftc.go.kr filetype:hwp OR filetype:pdf "건설업" "표준하도급계약서"',
        "type": "GOV_BOARD",
        "category": "Governance"
    }
]

class RiskCrawlingTool:
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
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
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
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        prefs = {
            "download.default_directory": DOWNLOAD_DIR,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
            "profile.default_content_settings.popups": 0,
            "profile.content_settings.exceptions.automatic_downloads.*.setting": 1
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
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

        category_context = target_info['category']
        # ESG Hub의 경우 이미 수집된 sub_category(E/S/G)를 활용
        if target_info.get("type") == "ESG_HUB":
            category_context = f"ESG_Specialized ({target_info.get('sub_category', 'General')})"

        prompt = f"""
        문서 제목: {title}
        카테고리: {category_context}
        내용 미리보기:
        {content_preview[:2500]}

        이 문서가 기업 현장에서 안전/환경/노무/거버넌스 리스크를 점검하거나 ESG 경영에 활용할 수 있는 **실무 자료**인지 판단해주세요.
        
        [판단 기준]
        - **유용함 (True)**: 체크리스트, 가이드라인, 매뉴얼, 표준계약서, ESG 평가 지표 해설.
        - **유용하지 않음 (False)**: 단순 행사 알림, 뉴스레터, 인사 발령.

        결과를 JSON으로 출력:
        {{
            "is_practical": true/false,
            "doc_type": "Checklist/Manual/Contract/Guide",
            "score": (1~10),
            "esg_tag": "E/S/G/Common",
            "summary": "한 줄 요약"
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            
            print(f"      👉 결과: {result['doc_type']} (점수: {result['score']}, 태그: {result.get('esg_tag')})")

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
                        "esg_tag": result.get('esg_tag', 'Common'),
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

    def _wait_for_download(self, before_files: set, title: str, target_info: Dict) -> bool:
        """다운로드 완료 대기 (시간 증가)"""
        # 30초 대기
        for i in range(30):
            time.sleep(1)
            current_files = set(os.listdir(DOWNLOAD_DIR))
            new_files = current_files - before_files
            
            if new_files:
                for new_file in new_files:
                    if not new_file.endswith('.crdownload') and not new_file.endswith('.tmp'):
                        full_path = os.path.join(DOWNLOAD_DIR, new_file)
                        if os.path.getsize(full_path) > 0:
                            print(f"      ✅ 다운로드 완료: {new_file}")
                            self._analyze_and_store(full_path, title, target_info)
                            return True
        return False

    # ----------------------------------------------------------------
    # [Crawling] 3. ESG Finance Hub (메뉴 클릭 + 체크박스 + 검색 버튼)
    # ----------------------------------------------------------------
    def _scrape_esg_finance_hub(self, driver, target_info: Dict) -> List[Dict]:
        """
        ESG 금융 추진단 보고서 크롤러 (개선 버전)
        - 메인 페이지에서 메뉴 클릭으로 접근
        - E/S/G 체크박스 클릭
        - 하위 항목 선택
        - 검색 버튼 클릭 (핵심!)
        - button.file-btn으로 PDF 다운로드
        """
        name = target_info["name"]
        results = []
        
        print(f"📡 [{name}] 접속 중...")
        try:
            # Step 1: 메인 페이지 접속
            main_url = "https://www.esgfinancehub.or.kr"
            driver.get(main_url)
            time.sleep(3)
            
            print("   🔎 메뉴 탐색 중...")
            
            # Step 2: "가이드라인" > "ESG공시" 메뉴 클릭
            try:
                from selenium.webdriver.common.action_chains import ActionChains
                # 가이드라인 메뉴 호버
                guideline_menu = driver.find_element(By.XPATH, "//a[contains(text(), '가이드라인')]")
                actions = ActionChains(driver)
                actions.move_to_element(guideline_menu).perform()
                time.sleep(1)
                
                # ESG공시 서브메뉴 클릭
                esg_submenu = driver.find_element(By.XPATH, "//a[contains(text(), 'ESG공시')]")
                driver.execute_script("arguments[0].click();", esg_submenu)
                time.sleep(4)
                print("   ✓ ESG공시 페이지 접속 완료")
                
            except Exception as e:
                print(f"   ⚠️ 메뉴 클릭 실패, 직접 URL 시도: {e}")
                # 대체: 직접 URL
                driver.get(target_info["url"])
                time.sleep(4)
            
            # Step 3: 체크박스 로딩 대기
            wait = WebDriverWait(driver, 15)
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='checkbox']")))
                print("   ✓ 페이지 로딩 완료")
                time.sleep(2)
            except TimeoutException:
                print("   ⚠️ 타임아웃")
            
            # Step 4: E, S, G 각 카테고리 순회
            esg_categories = [
                {'code': 'E', 'name': 'Environment'},
                {'code': 'S', 'name': 'Social'},
                {'code': 'G', 'name': 'Governance'}
            ]
            
            for esg_cat in esg_categories:
                try:
                    print(f"\n{'='*60}")
                    print(f"   🎯 [{esg_cat['code']}] 카테고리 처리 시작")
                    print(f"{'='*60}")
                    
                    # 페이지 새로고침
                    driver.refresh()
                    time.sleep(4)
                    
                    # Step 5: 메인 카테고리 체크박스 찾기
                    category_checkbox = None
                    all_checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                    
                    for cb in all_checkboxes:
                        try:
                            parent = cb.find_element(By.XPATH, "./..")
                            text = parent.text.strip()
                            
                            # "E (33)", "S (10)", "G (5)" 패턴 매칭
                            if text.startswith(f"{esg_cat['code']} ("):
                                category_checkbox = cb
                                print(f"      ✓ 발견: {text}")
                                break
                        except:
                            continue
                    
                    if not category_checkbox:
                        print(f"      ❌ {esg_cat['code']} 체크박스를 찾을 수 없음")
                        continue
                    
                    # Step 6: 메인 카테고리 클릭 (펼치기)
                    driver.execute_script("arguments[0].scrollIntoView(true);", category_checkbox)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", category_checkbox)
                    time.sleep(2)
                    print(f"      ✓ {esg_cat['code']} 펼침")
                    
                    # Step 7: 하위 항목 찾기
                    print(f"      🔍 하위 항목 검색 중...")
                    time.sleep(2)
                    
                    all_checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                    sub_items = []
                    
                    for cb in all_checkboxes:
                        try:
                            parent = cb.find_element(By.XPATH, "./..")
                            label = parent.text.strip()
                            
                            # 메인 카테고리 제외
                            if (label and 
                                not label.startswith('E (') and
                                not label.startswith('S (') and
                                not label.startswith('G (') and
                                2 < len(label) < 50):
                                
                                sub_items.append({
                                    'checkbox': cb,
                                    'label': label
                                })
                        except:
                            continue
                    
                    print(f"      📋 {len(sub_items)}개 하위 항목 발견")
                    
                    # Step 8: 각 하위 항목 처리 (최대 2개로 제한 - 빠른 테스트)
                    for idx, sub_item in enumerate(sub_items[:2]):
                        try:
                            sub_label = sub_item['label']
                            print(f"      [{idx+1}] {sub_label}")
                            
                            # 하위 체크박스 클릭
                            sub_checkbox = sub_item['checkbox']
                            driver.execute_script("arguments[0].scrollIntoView(true);", sub_checkbox)
                            time.sleep(0.5)
                            
                            if not sub_checkbox.is_selected():
                                driver.execute_script("arguments[0].click();", sub_checkbox)
                                time.sleep(1)
                            
                            # Step 9: **검색 버튼 클릭** (핵심!)
                            print(f"         🔍 검색 버튼 클릭 중...")
                            try:
                                search_button = driver.find_element(By.XPATH, "//button[contains(text(), '검색')]")
                                driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
                                time.sleep(0.5)
                                driver.execute_script("arguments[0].click();", search_button)
                                time.sleep(3)
                                print(f"         ✓ 검색 완료")
                            except Exception as search_err:
                                print(f"         ⚠️ 검색 버튼 오류: {search_err}")
                            
                            # Step 10: PDF 다운로드 버튼 찾기
                            print(f"         📄 PDF 파일 찾기 중...")
                            
                            # button.file-btn 찾기
                            download_buttons = driver.find_elements(By.CSS_SELECTOR, "button.file-btn")
                            
                            if not download_buttons:
                                # onclick에 fileDown 포함된 버튼 찾기
                                all_buttons = driver.find_elements(By.TAG_NAME, "button")
                                download_buttons = [btn for btn in all_buttons 
                                                  if 'fileDown' in (btn.get_attribute('onclick') or '')]
                            
                            print(f"         📥 {len(download_buttons)}개 다운로드 버튼 발견")
                            
                            # 최대 1개만 다운로드 (빠른 처리)
                            for btn_idx, dl_button in enumerate(download_buttons[:1]):
                                try:
                                    file_name = dl_button.text.strip() or f"{sub_label}_{btn_idx+1}.pdf"
                                    
                                    unique_key = f"{name}_{esg_cat['code']}_{sub_label}_{file_name}"
                                    
                                    if self._is_processed(unique_key):
                                        print(f"         ⏭️ [Skip] {file_name[:50]}")
                                        continue
                                    
                                    print(f"         📥 [{btn_idx+1}] {file_name[:50]}")
                                    
                                    before_files = set(os.listdir(DOWNLOAD_DIR))
                                    driver.execute_script("arguments[0].click();", dl_button)
                                    time.sleep(2)
                                    
                                    # 다운로드 대기
                                    target_info_with_sub = target_info.copy()
                                    target_info_with_sub['sub_category'] = esg_cat['name']
                                    
                                    downloaded_files = []
                                    if self._wait_for_download(before_files, file_name, target_info_with_sub):
                                        downloaded_files.append("downloaded")
                                    
                                    # 처리 완료 표시
                                    self._mark_as_processed(unique_key, file_name, downloaded_files)
                                    results.append({
                                        "source": name,
                                        "category": esg_cat['code'],
                                        "sub_category": sub_label,
                                        "title": file_name,
                                        "files": downloaded_files
                                    })
                                    
                                except Exception as dl_err:
                                    print(f"         ⚠️ 다운로드 오류: {dl_err}")
                            
                            # 체크박스 해제
                            if sub_checkbox.is_selected():
                                driver.execute_script("arguments[0].click();", sub_checkbox)
                                time.sleep(0.5)
                            
                            print(f"      ✓ [{idx+1}] {sub_label} 처리 완료")
                                
                        except Exception as sub_err:
                            print(f"      ⚠️ 하위 항목 오류: {sub_err}")
                            continue
                    
                    print(f"   ✅ [{esg_cat['code']}] 카테고리 처리 완료!\n")
                        
                except Exception as cat_err:
                    print(f"   ❌ {esg_cat['code']} 카테고리 오류: {cat_err}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"❌ ESG Hub 크롤링 실패: {e}")
            import traceback
            traceback.print_exc()
            
        return results

    # ... (KOSHA, Google Fallback 등 기존 메서드 유지) ...
    def _scrape_kosha_archive(self, driver, target_info: Dict) -> List[Dict]:
        # (기존 KOSHA 크롤러 로직 유지)
        url = target_info["url"]
        name = target_info["name"]
        results = []
        print(f"📡 [{name}] KOSHA 접속 중... ({url})")
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            time.sleep(3) 
            rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            for i in range(min(5, len(rows))):
                try:
                    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                    if i >= len(rows): break
                    row = rows[i]
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) < 5: continue
                    title = cols[2].text.strip()
                    unique_key = f"{name}_{title}"
                    if self._is_processed(unique_key):
                        print(f"   ⏭️ [Skip] {title}")
                        continue
                    print(f"   🔎 [New] 분석: {title}")
                    file_col = cols[4]
                    target_btn = None
                    try: target_btn = file_col.find_element(By.CSS_SELECTOR, "a.download")
                    except:
                        try: target_btn = file_col.find_element(By.CSS_SELECTOR, "a[class*='down']")
                        except:
                            try:
                                img = file_col.find_element(By.TAG_NAME, "img")
                                target_btn = img.find_element(By.XPATH, "./..")
                            except: pass
                    if target_btn:
                        before_files = set(os.listdir(DOWNLOAD_DIR))
                        driver.execute_script("arguments[0].click();", target_btn)
                        time.sleep(3)
                        downloaded_files = []
                        if self._wait_for_download(before_files, title, target_info):
                            downloaded_files.append("downloaded")
                        self._mark_as_processed(unique_key, title, downloaded_files)
                        results.append({"source": name, "title": title, "files": downloaded_files})
                except Exception as e: print(f"      ⚠️ Row {i} Error: {e}")
        except Exception as e: print(f"❌ KOSHA Error: {e}")
        return results

    def _scrape_google_fallback(self, driver, target_info: Dict) -> List[Dict]:
        # (기존 Google Fallback 로직 유지)
        query = target_info.get("google_query")
        if not query: return []
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        name = target_info["name"]
        results = []
        print(f"🚀 [Google Bypass] '{name}' 우회 검색... ({query})")
        try:
            driver.get(search_url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "search")))
            links = driver.find_elements(By.CSS_SELECTOR, "a")
            pdf_links = []
            for link in links:
                href = link.get_attribute("href")
                if href and (href.lower().endswith(".pdf") or href.lower().endswith(".hwp")):
                    pdf_links.append((link, href))
            seen_urls = set()
            unique_files = []
            for l, h in pdf_links:
                if h not in seen_urls:
                    unique_files.append((l, h))
                    seen_urls.add(h)
            for i, (link_elem, file_url) in enumerate(unique_files[:3]):
                try:
                    title = link_elem.text or "Untitled"
                    unique_key = f"Google_{name}_{title}"
                    if self._is_processed(unique_key):
                        print(f"   ⏭️ [Skip] {title}")
                        continue
                    print(f"   📥 [Direct Download] {title}")
                    response = requests.get(file_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                    if response.status_code == 200:
                        ext = os.path.splitext(file_url)[1] or ".pdf"
                        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_', '.')]).rstrip()[:50]
                        filename = f"{safe_title}{ext}"
                        file_path = os.path.join(DOWNLOAD_DIR, filename)
                        with open(file_path, 'wb') as f: f.write(response.content)
                        print(f"      ✅ 다운로드 완료: {filename}")
                        if self._analyze_and_store(file_path, title, target_info):
                            self._mark_as_processed(unique_key, title, [file_path])
                            results.append({"source": name, "title": title, "files": [file_path]})
                except Exception as e: print(f"      ⚠️ File Error: {e}")
        except Exception as e: print(f"❌ Google Error: {e}")
        return results

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
                elif target.get("type") == "ESG_HUB":
                    res = self._scrape_esg_finance_hub(driver, target)
                else:
                    res = self._scrape_google_fallback(driver, target)
                total_results.extend(res)
        finally:
            driver.quit()
            
        report = f"## 🛡️ 리스크 진단 자료 수집 리포트\n"
        if total_results:
            for item in total_results:
                files = f"{len(item['files'])}개 파일" if item['files'] else "없음"
                report += f"- **[{item['source']}]** {item['title']} (💾 {files})\n"
        else:
            report += "- 신규 자료가 없습니다.\n"
            
        print(report)
        return report

_risk_collector = RiskCrawlingTool()

@tool
def fetch_risk_guides(query: str = "safety checklist") -> str:
    """
    Collects practical risk assessment guides and checklists from KOSHA, MOEL, ME, FTC, and ESG Finance Hub.
    Uses Google Search fallback for government sites.
    """
    return _risk_collector.collect_all_guides()

if __name__ == "__main__":
    _risk_collector.collect_all_guides()