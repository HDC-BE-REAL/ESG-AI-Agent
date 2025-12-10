import sys
import os
import json
import pandas as pd
import re
from datetime import datetime
import fitz  # PyMuPDF

# src 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.tools import policy_tool
from src.tools.esg_policy_tool import _engine

# 파일 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(BASE_DIR, "data", "company", "삼성물산_ESG보고서.pdf")
OUTPUT_PATH = "esg_analysis_report.md"
SUBCONTRACTOR_INFO = "윤주건설 (철근콘크리트 및 비계 공사 전문)"

def extract_text_from_pdf(pdf_path, specific_pages=None):
    """PDF에서 특정 페이지 텍스트 추출 (0-indexed)"""
    print(f"📄 PDF 로딩: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    
    # 페이지 지정이 없으면 앞 5페이지만 (기존 로직)
    if not specific_pages:
        target_indices = range(5)
    else:
        target_indices = specific_pages

    print(f"   Reading pages: {target_indices}")
    for i in target_indices:
        if i < len(doc):
            page = doc[i]
            text += f"\n--- Page {i+1} ---\n"
            text += page.get_text()
            
    doc.close()
    return text

def find_kpi_target_pages(pdf_path, max_search_pages=30, top_k=3):
    """PDF에서 KPI/목표 테이블이 있는 페이지들을 자동으로 탐색 (상위 k개)"""
    print(f"🔎 KPI 목표 페이지 자동 탐색 중... (최대 {max_search_pages}페이지, 상위 {top_k}개)")
    doc = fitz.open(pdf_path)
    
    keywords = {
        "KPI": 3, "목표": 2, "실적": 2, "tCO2e": 3, 
        "재생에너지": 2, "중대재해": 2, "%": 1, "달성": 1
    }
    
    page_scores = []
    
    for i, page in enumerate(doc):
        if i >= max_search_pages: break
        
        text = page.get_text()
        score = 0
        for kw, points in keywords.items():
            score += text.count(kw) * points
            
        # 테이블 헤더 추정 ("목표"와 "실적"이 같이 나오면 가산점)
        if "목표" in text and "실적" in text:
            score += 5
            
        # 의미 있는 점수만 저장 (threshold = 10)
        if score > 10:
            page_scores.append((i, score))
            
    doc.close()
    
    # 점수 내림차순 정렬
    page_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 k개 추출
    top_pages = [idx for idx, _ in page_scores[:top_k]]
    
    if top_pages:
        print(f"   ✅ Best KPI Pages Found: {[p+1 for p in top_pages]}")
        return top_pages
    else:
        print("   ⚠️ KPI Page not found, defaulting to Page 13")
        return [12] # Default fallback

def generate_report():
    print("🚀 ESG 리포트 생성 시작...")
    
    if not _engine:
        print("❌ 엔진 초기화 실패")
        return

    # 1. 원청 데이터 로드 (삼성물산 보고서)
    # 분석 결과: Page 15, 16 (규제/폐기물/중대재해), 25 (법규), 42 (환경), 59 (안전보건법)
    # 0-based index: 14, 15, 24, 41, 58
    key_pages = [14, 15, 24, 41, 58]
    raw_text = extract_text_from_pdf(REPORT_PATH, specific_pages=key_pages)
    
    # 2. 리포트 작성
    report_content = "# 🏗️ 협력사용 ESG 가이드라인 분석 리포트\n"
    report_content += f"**대상 원청**: 삼성물산 (출처: {os.path.basename(REPORT_PATH)})\n"
    report_content += f"**수신**: {SUBCONTRACTOR_INFO}\n\n"
    
    # [Section 1] 지침 해설
    print("   🔍 [1/3] 지침 해설 생성 중...")
    explanation = _engine.explain_guideline(raw_text[:30000]) # 텍스트 길이 상향 (3만자)
    report_content += "## 1. 원청 ESG 지침 해설\n"
    report_content += explanation + "\n\n"
    
    # [Section 2] 체크리스트
    print("   📝 [2/3] 체크리스트 생성 중 (Excel용 대량 생성)...")
    checklist_json_str = _engine.generate_checklist(raw_text[:30000], SUBCONTRACTOR_INFO)
    
    # JSON Parsing & Excel Export
    try:
        # Markdown Code Block 제거 (```json ... ```)
        cleaned_json = re.sub(r"```json\s*|\s*```", "", checklist_json_str, flags=re.DOTALL).strip()
        checklist_data = json.loads(cleaned_json)
        
        # DataFrame 생성
        df = pd.DataFrame(checklist_data)
        
        # [User Request] 'importance' 컬럼 제거
        if 'importance' in df.columns:
            df = df.drop(columns=['importance'])
            
        # [User Request] 'item' 또는 'question' 컬럼에서 '(Yes/No)' 텍스트 제거
        for col in ['item', 'question', '점검항목']:
            if col in df.columns:
                # 정규식으로 (Yes/No), (yes/no) 등 제거하고 공백 정리
                df[col] = df[col].astype(str).str.replace(r"\(Yes/No\)", "", case=False, regex=True).str.strip()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"esg_checklist_{timestamp}.xlsx"
        df.to_excel(excel_filename, index=False)
        print(f"   💾 Excel 저장 완료: {excel_filename} ({len(df)} items)")
        
        report_content += f"## 2. 현장 실무자용 체크리스트 ({SUBCONTRACTOR_INFO})\n"
        report_content += f"**✅ 상세 체크리스트는 별도 엑셀 파일로 생성되었습니다: `{excel_filename}`**\n\n"
        report_content += f"총 {len(df)}개의 점검 항목이 포함되어 있습니다. (안전, 환경, 인권 등)\n\n"
        
    except Exception as e:
        print(f"   ⚠️ Excel 생성 실패 (JSON 파싱 오류 등): {e}")
        report_content += f"## 2. 현장 실무자용 체크리스트\n(Excel 생성 실패로 텍스트로 대체합니다)\n{checklist_json_str}\n\n"

    
    # [Section 3] Gap Analysis (Target from Page 13)
    print("   📊 [3/3] 목표 vs 실적 비교 분석 중...")
    
    # Page 13 (Index 12) - KPI 목표 테이블 페이지 추출
    # [Dynamic] 자동으로 KPI 페이지 탐색 (다중 페이지)
    target_page_indices = find_kpi_target_pages(REPORT_PATH)
    target_page_text = extract_text_from_pdf(REPORT_PATH, specific_pages=target_page_indices)
    
    # 하청 실적 (가상 데이터 - 삼성물산 목표 항목에 맞춰 구체화)
    actual_data = """
    - 온실가스 배출량: 1,200 tCO2e (전년 대비 3% 감소)
    - 재생에너지 사용률: 15% (태양광 패널 일부 설치)
    - 중대재해: 3건
    - 폐기물 재활용률: 85%
    - 안전교육 이수율: 95%
    """
    
    gap_analysis = _engine.analyze_gap(target_page_text, actual_data)
    report_content += "## 3. 목표 대비 성과 분석 (Gap Analysis)\n"
    report_content += f"> **비교 기준**: 삼성물산 2025 지속가능경영보고서 내 'KPI 이행현황 및 목표' (자동 탐지된 Pages {[p+1 for p in target_page_indices]})\n\n"
    report_content += gap_analysis + "\n\n"
    
    # 파일 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"✅ 리포트 생성 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_report()