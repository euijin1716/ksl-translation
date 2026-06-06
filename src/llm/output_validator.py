"""LLM 출력 검증기.

"인식기가 보지 못한 내용을 임의로 추가하지 않는다"는 원칙을 코드 레벨에서 강제한다.
LLM 응답을 무조건 정답으로 채택하지 않는다.
"""

from __future__ import annotations

import logging
from typing import Any

from .provider import LLMInput, LLMOutput

logger = logging.getLogger(__name__)

# 오역 시 실제 피해가 큰 도메인 (schema.DOMAINS 기준: 병원·민원)
HIGH_RISK_DOMAINS = {"hospital", "public"}

# 도메인별 최대 허용 길이 비율 (draft 대비)
_DOMAIN_MAX_EXPANSION: dict[str, float] = {
    "hospital": 1.5,
    "public": 1.5,
    "directions": 2.0,
    "order": 2.0,
    "reservation": 2.0,
    "help": 1.8,
    "unknown": 2.0,
}
_DEFAULT_MAX_EXPANSION = 2.0


def validate_output(
    llm_output: LLMOutput,
    llm_input: LLMInput,
) -> LLMOutput:
    """LLM 출력이 프로젝트 원칙에 부합하는지 검사하고 필요 시 fallback한다.

    검사 항목:
    1. 빈 final_text → draft로 대체
    2. 고위험 도메인 + 저신뢰 + 내용 변경 → draft + retry 플래그 (가장 엄격)
    3. 길이 폭발(도메인별 threshold 초과) → draft로 대체
    4. retry_or_clarify=True인데 uncertain_spans 없음 → 자동 보강
    """
    draft = llm_input.korean_draft
    final = llm_output.final_text

    # 1. 빈 출력 방어
    if not final or not final.strip():
        logger.warning("LLM returned empty final_text; falling back to draft.")
        return _fallback(llm_output, draft, "empty_output")

    # 2. 고위험 도메인 + 저신뢰 + 내용 변경 (가장 엄격한 정책을 먼저 적용)
    if (
        llm_input.domain in HIGH_RISK_DOMAINS
        and llm_input.confidence < 0.5
        and final.strip() != draft.strip()
        and draft  # draft가 있을 때만
    ):
        logger.warning(
            "High-risk domain '%s' with low confidence %.2f and modified output. Falling back.",
            llm_input.domain, llm_input.confidence,
        )
        return LLMOutput(
            final_text=draft,
            uncertain_spans=[{"text": draft, "reason": "low_confidence_high_risk_domain"}],
            retry_or_clarify=True,
            normalization_notes="high_risk_low_confidence_rejected",
            raw_response=llm_output.raw_response,
        )

    # 3. 길이 폭발 체크
    max_ratio = _DOMAIN_MAX_EXPANSION.get(llm_input.domain, _DEFAULT_MAX_EXPANSION)
    if draft and len(final) > len(draft) * max_ratio:
        logger.warning(
            "LLM output length %d exceeds draft length %d × %.1f (domain=%s). Falling back.",
            len(final), len(draft), max_ratio, llm_input.domain,
        )
        return _fallback(llm_output, draft, "length_expansion_rejected")

    # 4. retry_or_clarify이면 uncertain_spans 보강
    if llm_output.retry_or_clarify and not llm_output.uncertain_spans:
        updated_spans: list[dict[str, Any]] = [{"text": final, "reason": "retry_requested"}]
        return LLMOutput(
            final_text=final,
            uncertain_spans=updated_spans,
            retry_or_clarify=True,
            normalization_notes=llm_output.normalization_notes,
            raw_response=llm_output.raw_response,
        )

    return llm_output


def _fallback(original: LLMOutput, draft: str, note: str) -> LLMOutput:
    return LLMOutput(
        final_text=draft,
        uncertain_spans=original.uncertain_spans,
        retry_or_clarify=original.retry_or_clarify,
        normalization_notes=note,
        raw_response=original.raw_response,
    )
