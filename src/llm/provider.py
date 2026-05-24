"""LLM provider 추상 인터페이스.

LLM은 문맥 보정기로만 사용한다.
비전 입력을 직접 LLM에 넣지 않는다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMInput:
    """LLM 문맥 보정기 입력 구조."""
    korean_draft: str
    top_k_gloss: list[str]               # 상위 gloss 가설
    nms_summary: dict[str, Any]          # 비수지신호 요약
    confidence: float                    # 인식기 신뢰도 (0~1)
    previous_turns: list[str]            # 이전 발화 한국어 (최대 5개)
    domain: str                          # 발화 도메인
    retry_or_clarify: bool = False       # 재질문 상태


@dataclass
class LLMOutput:
    """LLM 문맥 보정기 출력 구조."""
    final_text: str
    uncertain_spans: list[dict[str, Any]] = field(default_factory=list)
    retry_or_clarify: bool = False
    normalization_notes: str = ""        # 내부용 메모 (사용자 노출 금지)
    raw_response: str = ""               # 원본 응답 (디버깅용)


class LLMProvider(ABC):
    """Provider-agnostic LLM 인터페이스."""

    @abstractmethod
    def correct(self, llm_input: LLMInput) -> LLMOutput:
        """한국어 초안을 문맥 보정해 최종 문장을 반환한다."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Provider 연결 상태를 확인한다."""
        ...
