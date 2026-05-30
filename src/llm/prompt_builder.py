"""LLM 프롬프트 빌더.

구조화된 입력을 프롬프트 텍스트로 변환한다.
"의미 보존"을 "자연스러움"보다 우선한다.
도메인별 보수성 정책을 프롬프트 레벨에서도 반영한다.
"""

from __future__ import annotations

from .provider import LLMInput

# 병원/민원처럼 오역 시 실제 피해가 큰 도메인
HIGH_RISK_DOMAINS = {"hospital", "public_service"}

SYSTEM_PROMPT = """당신은 한국수어(KSL) 통역 시스템의 문맥 보정기입니다.

역할:
- 수어 인식기가 생성한 한국어 초안과 gloss/NMS 힌트를 바탕으로 최종 한국어 문장을 정제합니다.
- 인식기가 보지 못한 내용을 임의로 추가하지 않습니다.
- 의미 보존이 자연스러운 표현보다 우선입니다.
- 신뢰도가 낮은 경우 보수적으로 출력하거나 재표현을 요청합니다.
- 병원/민원 등 고위험 도메인에서는 특히 보수적으로 동작합니다.

금지 사항:
- 인식기 출력에 없는 내용 추가
- 저신뢰 입력을 과감하게 자연화해 오역 유발
- 내부 추론 과정을 사용자 출력에 포함

출력 형식 (JSON):
{
  "final_text": "최종 한국어 문장",
  "uncertain_spans": [{"text": "불확실한 부분", "reason": "이유"}],
  "retry_or_clarify": false,
  "normalization_notes": "내부 메모"
}"""

# 도메인별 추가 지침
_DOMAIN_GUIDELINES: dict[str, str] = {
    "hospital": (
        "【병원 도메인 주의】 증상·약명·처방 등의 의료 정보는 오역이 심각한 결과를 초래할 수 있습니다. "
        "인식기 출력에 없는 의료 용어를 절대 추가하지 마세요. "
        "불확실한 부분은 uncertain_spans에 반드시 표시하고, 필요 시 retry_or_clarify를 true로 설정하세요."
    ),
    "public_service": (
        "【공공 민원 도메인 주의】 행정 절차·법적 용어는 정확하게 유지하세요. "
        "원문에 없는 조건이나 기한을 추가하지 마세요."
    ),
    "directions": (
        "【길 안내 도메인】 방향·장소명은 인식기 출력 그대로 유지하세요. "
        "추론으로 목적지를 바꾸지 마세요."
    ),
    "order_payment": (
        "【주문/결제 도메인】 수량·금액·메뉴명은 인식기 출력 그대로 유지하세요."
    ),
    "reservation": (
        "【예약/확인 도메인】 날짜·시간·인원은 인식기 출력 그대로 유지하세요."
    ),
    "help": (
        "【도움 요청 도메인】 긴급도를 임의로 높이거나 낮추지 마세요."
    ),
}

# 신뢰도 구간별 추가 지시
_LOW_CONFIDENCE_INSTRUCTION = (
    "⚠️ 신뢰도가 낮습니다. 초안을 최대한 보존하고, "
    "확신이 없는 부분은 uncertain_spans에 표시하세요. "
    "내용을 임의로 완성하거나 추론하지 마세요."
)
_MID_CONFIDENCE_INSTRUCTION = (
    "신뢰도가 중간입니다. 명확히 오류인 부분만 교정하고, "
    "불확실한 부분은 uncertain_spans에 표시하세요."
)


def build_prompt(llm_input: LLMInput) -> str:
    """LLMInput을 구조화된 프롬프트로 변환한다."""
    prev_turns_text = "\n".join(
        f"  [{i+1}] {t}" for i, t in enumerate(llm_input.previous_turns[-5:])
    ) or "  (없음)"

    gloss_text = ", ".join(llm_input.top_k_gloss) or "(없음)"
    nms_text = _format_nms(llm_input.nms_summary)
    confidence_level = _confidence_level(llm_input.confidence)
    domain_guideline = _domain_guideline(llm_input.domain, llm_input.confidence)
    confidence_instruction = _confidence_instruction(llm_input.confidence)

    prompt = f"""## 입력 정보

도메인: {llm_input.domain}
신뢰도: {llm_input.confidence:.2f} ({confidence_level})
재질문 상태: {"예" if llm_input.retry_or_clarify else "아니오"}

## 수어 인식 결과

한국어 초안: {llm_input.korean_draft}
상위 gloss: {gloss_text}
비수지신호(NMS): {nms_text}

## 이전 발화 문맥 (참고용)

{prev_turns_text}
{domain_guideline}
## 지시

{confidence_instruction}
위 정보를 바탕으로 한국어 초안을 정제해 JSON 형식으로 출력하세요.
신뢰도가 낮으면 불확실한 부분을 uncertain_spans에 표시하고, 필요 시 retry_or_clarify를 true로 설정하세요."""

    return prompt


def _confidence_level(conf: float) -> str:
    if conf >= 0.8:
        return "높음"
    elif conf >= 0.5:
        return "중간"
    else:
        return "낮음"


def _domain_guideline(domain: str, confidence: float) -> str:
    guideline = _DOMAIN_GUIDELINES.get(domain, "")
    if not guideline:
        return "\n"
    # 고위험 도메인 + 저신뢰이면 guideline을 더 강조
    if domain in HIGH_RISK_DOMAINS and confidence < 0.5:
        return f"\n## 도메인 주의사항\n\n{guideline}\n🔴 현재 신뢰도가 낮습니다. 가능하면 재질문을 요청하세요.\n\n"
    return f"\n## 도메인 주의사항\n\n{guideline}\n\n"


def _confidence_instruction(conf: float) -> str:
    if conf < 0.5:
        return _LOW_CONFIDENCE_INSTRUCTION + "\n"
    elif conf < 0.8:
        return _MID_CONFIDENCE_INSTRUCTION + "\n"
    return ""


def _format_nms(nms: dict) -> str:
    if not nms:
        return "(없음)"
    parts = []
    for k, v in nms.items():
        if v is not None:
            parts.append(f"{k}={v}")
    return ", ".join(parts) or "(없음)"
