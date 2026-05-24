"""ContextCorrector - 인퍼런스 파이프라인이 사용하는 LLM 보정기 wrapper."""

from __future__ import annotations

from typing import Any

from .provider import LLMInput, LLMOutput, LLMProvider
from .adapters.dummy_adapter import DummyLLMAdapter


class ContextCorrector:
    """LLM provider를 감싸는 문맥 보정기.

    Args:
        provider: LLMProvider 구현체 (없으면 DummyLLMAdapter 사용)
        max_prev_turns: 유지할 이전 발화 최대 수
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        max_prev_turns: int = 5,
    ) -> None:
        self.provider = provider or DummyLLMAdapter()
        self.max_prev_turns = max_prev_turns
        self._turn_history: list[str] = []

    def correct(
        self,
        korean_draft: str,
        gloss_hypotheses: list[str],
        nms_summary: dict[str, Any],
        confidence: float,
        domain: str,
        retry_or_clarify: bool = False,
    ) -> LLMOutput:
        """한국어 초안을 문맥 보정해 최종 문장을 반환한다.

        low confidence이면 문장을 확정적으로 꾸미지 않는다.
        """
        llm_input = LLMInput(
            korean_draft=korean_draft,
            top_k_gloss=gloss_hypotheses,
            nms_summary=nms_summary,
            confidence=confidence,
            previous_turns=self._turn_history[-self.max_prev_turns:],
            domain=domain,
            retry_or_clarify=retry_or_clarify,
        )
        output = self.provider.correct(llm_input)

        # 성공한 발화를 history에 추가
        if not output.retry_or_clarify and output.final_text:
            self._turn_history.append(output.final_text)
            if len(self._turn_history) > self.max_prev_turns * 2:
                self._turn_history = self._turn_history[-self.max_prev_turns:]

        return output

    def reset_history(self) -> None:
        self._turn_history.clear()
