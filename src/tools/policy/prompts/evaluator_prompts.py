EVALUATE_PROMPT = """
당신은 ESG 정책 성숙도 평가 전문가입니다.
아래 정책을 5개 기준으로 평가하고,
점수(0~5) + 총평 + 개선사항을 제시하세요.

[평가 기준]
1. 명확성(Clarity)
2. 측정 가능성(Measurability)
3. 책임성(Accountability)
4. 투명성(Transparency)
5. 글로벌 표준 정합성(Global Alignment)

[정책 텍스트]
{text}

[출력(JSON)]
{
  "clarity": ...,
  "measurability": ...,
  "accountability": ...,
  "transparency": ...,
  "alignment": ...,
  "summary": "...",
  "improvements": ["...", "..."]
}
"""
