COMPARE_PROMPT = """
당신은 글로벌 ESG 컴플라이언스 전문가입니다.
두 정책 내용을 비교하여 구조적 GAP 분석을 수행하세요.

[비교 기준]
1. 공통 요소(Common Elements)
2. 차이점(Differences)
3. 강점(Strengths)
4. 부족 요소(Gaps)
5. 글로벌 기준 정합성 점수(0~100)
   - GRI
   - SASB
   - K-ESG
6. 개선 방향(Recommendations)

[입력 A]
{policy_a}

[입력 B]
{policy_b}

[출력 형식]
- 공통 요소:
- 차이점:
- 강점:
- 부족 요소(Gaps):
- 글로벌 정합성 점수:
- 개선 방향:
"""
