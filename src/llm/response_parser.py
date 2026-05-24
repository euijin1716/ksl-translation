"""LLM 응답 파서.

JSON 응답을 LLMOutput으로 변환한다.
파싱 실패 시 fallback으로 korean_draft를 그대로 반환한다.
"""

from __future__ import annotations

import json
import logging
import re

from .provider import LLMOutput

logger = logging.getLogger(__name__)


def parse_response(raw: str, fallback_text: str = "") -> LLMOutput:
    """LLM 응답 문자열을 LLMOutput으로 파싱한다.

    JSON 추출 실패 시 fallback_text를 final_text로 반환한다.
    LLM이 반환한 문장을 무조건 정답으로 채택하지 않는다.
    """
    try:
        # JSON 블록 추출 시도
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return LLMOutput(
                final_text=data.get("final_text", fallback_text) or fallback_text,
                uncertain_spans=data.get("uncertain_spans", []),
                retry_or_clarify=bool(data.get("retry_or_clarify", False)),
                normalization_notes=data.get("normalization_notes", ""),
                raw_response=raw,
            )
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"LLM response parse failed: {e}. Using fallback.")

    return LLMOutput(
        final_text=fallback_text,
        uncertain_spans=[],
        retry_or_clarify=False,
        raw_response=raw,
    )
