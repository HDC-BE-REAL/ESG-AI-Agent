from __future__ import annotations

from typing import List, Tuple

from .iso31000 import RiskAssessmentEntry, identify_risks
from .utils import sentence_tokenize, to_csv


INCREASE_KEYWORDS = ["증가", "악화", "심화", "빈번", "확대", "상승"]
DECREASE_KEYWORDS = ["감소", "개선", "완화", "축소", "하락"]
HISTORIC_PATTERNS = [
    ("최근", INCREASE_KEYWORDS, "증가"),
    ("최근", DECREASE_KEYWORDS, "감소"),
    ("지난", INCREASE_KEYWORDS, "증가"),
    ("지난", DECREASE_KEYWORDS, "감소"),
]

FINANCIAL_KEYWORDS = ["벌금", "과태료", "중단", "지연", "비용", "투자", "손실", "재무", "규제"]
FINANCIAL_LOSS_TERMS = ["손실", "매출 감소", "영업이익", "비용 증가", "정산", "배상"]
SUPPLY_CHAIN_KEYWORDS = ["협력", "공급망", "하도급", "협력사", "vendor", "partner"]
STAKEHOLDER_KEYWORDS = ["주민", "근로자", "투자자", "감리", "노조", "민원"]
SYSTEMIC_KEYWORDS = ["산업", "전반", "시장", "글로벌", "법제", "정책"]
KPI_KEYWORDS = ["kpi", "지표", "측정", "달성률", "모니터링"]

ACTION_LIBRARY = {
    "안전": "고위험 작업 허가제 및 현장 특별점검",
    "환경": "배출/누출 모니터링 자동화 및 인허가 재점검",
    "노동": "근로시간·임금 데이터 실시간 모니터링",
    "거버넌스": "윤리위반 제보 처리 SLA 강화",
}

FINANCIAL_HAZARD_WEIGHTS = {
    "환경": 1.2,
    "거버넌스": 1.1,
    "안전": 1.0,
    "노동": 0.95,
}

DIMENSION_TEMPLATES = [
    "근거 '{snippet}' → {keyword} 영향 징후",
    "문장 '{snippet}' 에서 {keyword} 언급",
    "해당 근거('{snippet}') 기준 {keyword} 관련 영향",
    "'{keyword}' 키워드 기반으로 파악된 영향 (근거: {snippet})",
]

AREA_TOPIC_MAP = {
    "안전": "Safety",
    "환경": "Environment",
    "노동": "Labor",
    "거버넌스": "Governance",
}


def _build_action_plan(top_risks: List[RiskAssessmentEntry]) -> List[str]:
    if not top_risks:
        return ["1) ISO 31000 기반 정기 리스크 검토", "2) 공급망 협력사 ESG 실사 강화", "3) KPI 연동 모니터링 대시보드 구축"]
    actions = []
    seen = set()
    for entry in top_risks:
        area = entry.hazard.area
        if area in seen:
            continue
        seen.add(area)
        action = ACTION_LIBRARY.get(area)
        if action:
            actions.append(f"- {area}: {action}")
    if not actions:
        actions.append("- 공통: ISO 31000 기반 CAPA 실행")
    return actions


def _detect_sentence_trend(sentence: str) -> str | None:
    lowered = sentence.lower()
    if any(keyword in lowered for keyword in INCREASE_KEYWORDS):
        return "증가"
    if any(keyword in lowered for keyword in DECREASE_KEYWORDS):
        return "감소"
    return None


def _trend_summary(risks: List[RiskAssessmentEntry], context: str) -> Tuple[str, str, str]:
    lowered = context.lower()
    sentences = sentence_tokenize(context)
    summary = "정체"
    sentence_trends = [(_detect_sentence_trend(sentence), idx) for idx, sentence in enumerate(sentences)]
    sentence_trends = [(label, idx) for label, idx in sentence_trends if label]
    if sentence_trends:
        last_three = sentence_trends[-3:]
        vote = max(set(label for label, _ in last_three), key=lambda l: sum(1 for label, _ in last_three if label == l))
        summary = vote
    else:
        inc_count = sum(lowered.count(keyword) for keyword in INCREASE_KEYWORDS)
        dec_count = sum(lowered.count(keyword) for keyword in DECREASE_KEYWORDS)
        if inc_count > dec_count and inc_count:
            summary = "증가"
        elif dec_count > inc_count and dec_count:
            summary = "감소"
    for trigger, keywords, label in HISTORIC_PATTERNS:
        if trigger in lowered and any(keyword in lowered for keyword in keywords):
            summary = label
            break
    drivers = []
    if "법" in lowered or "규제" in lowered:
        drivers.append("법규/정책 변화")
    if any(entry.rating in {"High", "Extreme"} for entry in risks):
        drivers.append("High Risk 빈도")
    mentions = sum(lowered.count(entry.hazard.event.lower()) for entry in risks)
    if mentions > len(risks):
        drivers.append("반복 언급 증가")
    if any(keyword in lowered for keyword in SUPPLY_CHAIN_KEYWORDS):
        drivers.append("공급망 영향 언급")
    if any(keyword in lowered for keyword in KPI_KEYWORDS):
        drivers.append("KPI/지표 압박")
    if not drivers:
        drivers.append("문서 기반 일반 추정")
    evidence = "문서에서 명시된 근거 문장 필요"
    if risks and risks[0].evidences:
        evidence = risks[0].evidences[0].sentence
    return summary, ", ".join(dict.fromkeys(drivers)), evidence


def _materiality_level(entry: RiskAssessmentEntry, context: str) -> Tuple[str, str]:
    impact_level = "High" if entry.impact >= 4 else "Medium" if entry.impact >= 2.5 else "Low"
    evidence_texts = " ".join(e.sentence for e in entry.evidences).lower()
    keyword_hits = sum(evidence_texts.count(keyword) for keyword in FINANCIAL_KEYWORDS)
    loss_hits = sum(evidence_texts.count(keyword) for keyword in FINANCIAL_LOSS_TERMS)
    similarity_weight = sum(getattr(e, "similarity", 0.0) for e in entry.evidences)
    similarity_weight = similarity_weight or 0.4
    hazard_bias = FINANCIAL_HAZARD_WEIGHTS.get(entry.hazard.area, 1.0)
    financial_signal = (entry.score / 5) * hazard_bias + keyword_hits * 0.4 + loss_hits * 0.7 + similarity_weight * 0.5
    if financial_signal >= 6.5:
        financial = "High"
    elif financial_signal >= 3.8:
        financial = "Medium"
    else:
        financial = "Low"
    return impact_level, financial


def _format_topic(entry: RiskAssessmentEntry) -> str:
    area = AREA_TOPIC_MAP.get(entry.hazard.area, entry.hazard.area)
    event = entry.hazard.event or "Risk"
    source = entry.hazard.source or entry.hazard.area
    return f"{area} > {event} ({source})"


def _dimension_template(keyword: str, snippet: str) -> str:
    if not snippet:
        return f"{keyword} 관련 근거 필요"
    suffix = "..." if len(snippet) > 80 else ""
    trimmed = snippet[:80] + suffix
    index = abs(hash(f"{keyword}-{trimmed}")) % len(DIMENSION_TEMPLATES)
    template = DIMENSION_TEMPLATES[index]
    return template.format(keyword=keyword, snippet=trimmed)


def _analyze_dimension(entry: RiskAssessmentEntry, keywords: List[str], base_message: str) -> str:
    if not entry.evidences:
        return base_message
    evidence_texts = " ".join(e.sentence for e in entry.evidences).lower()
    matched = None
    for keyword in keywords:
        if keyword in evidence_texts:
            matched = keyword
            break
    if matched:
        snippet = entry.evidences[0].sentence or ""
        return _dimension_template(matched, snippet)
    return base_message


def analyze_materiality(context: str, question: str = "") -> str:
    if not context.strip():
        return "Trend/Materiality 분석을 위해 문서(context)가 필요합니다."
    risks = identify_risks(context)
    if not risks:
        return "문서에서 리스크 근거를 찾지 못했습니다. 추가 데이터를 제공해 주세요."
    summary, drivers, evidence = _trend_summary(risks, context)
    double_rows = []
    for entry in risks:
        impact_level, financial_level = _materiality_level(entry, context)
        topic = _format_topic(entry)
        evidence_sentence = entry.evidences[0].sentence if entry.evidences else "근거 부족"
        double_rows.append(
            (
                topic,
                f"{impact_level} (Impact {entry.impact:.1f})",
                f"{financial_level} (Score {entry.score:.1f})",
                entry.hazard.source or entry.hazard.area,
                evidence_sentence,
            )
        )
    double_csv = to_csv(["Topic", "Impact Materiality", "Financial Materiality", "Risk Source", "Evidence"], double_rows)
    triple_rows = []
    for entry in risks:
        topic = _format_topic(entry)
        supply = _analyze_dimension(entry, SUPPLY_CHAIN_KEYWORDS, "협력사 관리 성숙도 필요")
        stakeholder = _analyze_dimension(entry, STAKEHOLDER_KEYWORDS, "주요 이해관계자 민감도 고려")
        systemic = _analyze_dimension(entry, SYSTEMIC_KEYWORDS, "산업/정책 수준 파급 잠재")
        triple_rows.append((topic, supply, stakeholder, systemic))
    triple_csv = to_csv(["Topic", "Value Chain Impact", "Stakeholder Impact", "Systemic Impact"], triple_rows)
    top_risks = sorted(risks, key=lambda r: r.score, reverse=True)[:5]
    top_lines = [
        f"- {entry.hazard.area}/{entry.hazard.event}: {entry.rating} (Score {entry.score:.1f}, Acceptance {entry.acceptance})"
        for entry in top_risks
    ]
    action_lines = _build_action_plan(top_risks)
    output_sections = [
        f"Trend Summary: {summary}",
        f"Trend Drivers: {drivers}",
        f"근거 문장: {evidence}",
        "",
        "[Double Materiality]",
        double_csv,
        "",
        "[Triple Materiality]",
        triple_csv,
        "",
        "고위험 리스크 요약",
        "\n".join(top_lines) if top_lines else "- 고위험 항목 없음",
        "",
        "중요성 평가 결과 요약",
        "- Impact/Financial Materiality를 종합하여 중점 관리 리스크를 도출",
        "",
        "Action Plan",
        "\n".join(action_lines),
    ]
    return "\n".join(output_sections)
