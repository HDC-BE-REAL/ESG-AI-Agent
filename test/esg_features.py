import sys
import os

# src 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.tools import policy_tool
from src.tools.esg_policy_tool import _engine

def test_esg_features():
    print("🧪 ESG 기능 테스트 시작")
    
    if not _engine:
        print("❌ 엔진 초기화 실패")
        return

    # 1. 지침 해설 테스트
    print("\n[Test 1] 지침 해설 (Explanation)")
    guideline_text = "협력회사는 ISO 14001 인증을 취득하고, 온실가스 배출량을 Scope 1, 2 기준으로 연 1회 보고해야 한다."
    res1 = _engine.explain_guideline(guideline_text)
    print(f"👉 결과:\n{res1}")
    
    # 2. 체크리스트 생성 테스트
    print("\n[Test 2] 체크리스트 생성 (Checklist)")
    res2 = _engine.generate_checklist(guideline_text)
    print(f"👉 결과:\n{res2}")
    
    # 3. Gap 분석 테스트
    print("\n[Test 3] Gap 분석 (Target vs Actual)")
    target = "폐기물 재활용률 80% 이상 달성"
    actual = "2023년 폐기물 재활용률 75%"
    res3 = _engine.analyze_gap(target, actual)
    print(f"👉 결과:\n{res3}")
    
    # 4. 용어 검색 테스트 (RAG)
    print("\n[Test 4] 용어 검색 (Terminology)")
    term_query = "Scope 1과 Scope 2의 차이가 뭐야?"
    res4 = _engine.search_terminology(term_query)
    print(f"👉 결과:\n{res4}")

    print("\n✅ 테스트 완료")

if __name__ == "__main__":
    test_esg_features()