"""LLM 프롬프트 빌더.

구조화된 입력을 프롬프트 텍스트로 변환한다.
"의미 보존"을 "자연스러움"보다 우선한다.
"""

from __future__ import annotations

from .provider import LLMInput

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


def build_prompt(llm_input: LLMInput) -> str:
    """LLMInput을 구조화된 프롬프트로 변환한다."""
    prev_turns_text = "\n".join(
        f"  [{i+1}] {t}" for i, t in enumerate(llm_input.previous_turns[-5:])
    ) or "  (없음)"

    gloss_text = ", ".join(llm_input.top_k_gloss) or "(없음)"
    nms_text = _format_nms(llm_input.nms_summary)

    confidence_level = _confidence_level(llm_input.confidence)

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

## 지시

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


def _format_nms(nms: dict) -> str:
    if not nms:
        return "(없음)"
    parts = []
    for k, v in nms.items():
        if v is not None:
            parts.append(f"{k}={v}")
    return ", ".join(parts) or "(없음)"
