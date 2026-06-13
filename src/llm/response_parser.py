"""LLM 응답 파서.

JSON 응답을 LLMOutput으로 변환한다.
balanced-brace 방식으로 outermost JSON을 추출해 greedy regex의 취약점을 방지한다.
파싱 실패 시 fallback으로 korean_draft를 그대로 반환한다.
"""

from __future__ import annotations

import json
import logging

from .provider import LLMOutput

logger = logging.getLogger(__name__)


def parse_response(raw: str, fallback_text: str = "") -> LLMOutput:
    """LLM 응답 문자열을 LLMOutput으로 파싱한다.

    JSON 추출 실패 시 fallback_text를 final_text로 반환한다.
    """
    data = _extract_json(raw)
    if data is not None:
        return LLMOutput(
            final_text=data.get("final_text", fallback_text) or fallback_text,
            uncertain_spans=data.get("uncertain_spans", []),
            retry_or_clarify=bool(data.get("retry_or_clarify", False)),
            normalization_notes=data.get("normalization_notes", ""),
            raw_response=raw,
        )

    logger.warning("LLM response parse failed; using fallback.")
    return LLMOutput(
        final_text=fallback_text,
        uncertain_spans=[],
        retry_or_clarify=False,
        normalization_notes="parse_failed",
        raw_response=raw,
    )


def _extract_json(raw: str) -> dict | None:
    """응답 문자열에서 outermost JSON 객체를 추출한다.

    1) 전체 문자열을 직접 파싱 시도
    2) balanced-brace 탐색으로 첫 번째 완전한 {…} 블록 추출
    """
    stripped = raw.strip()

    # 직접 파싱
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # balanced-brace 탐색
    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(raw[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
                # 이 블록 파싱 실패 → 다음 {를 찾아 재시도
                next_start = raw.find("{", i + 1)
                if next_start == -1:
                    return None
                return _extract_json(raw[next_start:])

    return None
