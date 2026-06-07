"""ContextCorrector - 인퍼런스 파이프라인이 사용하는 LLM 보정기 wrapper."""

from __future__ import annotations

from typing import Any

from .output_validator import validate_output
from .provider import LLMInput, LLMOutput, LLMProvider
from .adapters.dummy_adapter import DummyLLMAdapter


class ContextCorrector:
    """LLM provider를 감싸는 문맥 보정기.

    Args:
        provider: LLMProvider 구현체 (없으면 DummyLLMAdapter 사용)
        max_prev_turns: 유지할 이전 발화 최대 수
        low_confidence_threshold: 이 미만이면 uncertain_spans를 자동 보강
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        max_prev_turns: int = 5,
        low_confidence_threshold: float = 0.4,
    ) -> None:
        self.provider = provider or DummyLLMAdapter()
        self.max_prev_turns = max_prev_turns
        self.low_confidence_threshold = low_confidence_threshold
        self._turn_history: list[str] = []

    def correct(
        self,
        korean_draft: str,
        gloss_hypotheses: list[str],
        nms_summary: dict[str, Any],
        confidence: float,
        domain: str,
        retry_or_clarify: bool = False,
        gloss_confidences: list[float] | None = None,
    ) -> LLMOutput:
        """한국어 초안을 문맥 보정해 최종 문장을 반환한다.

        low confidence이면 문장을 확정적으로 꾸미지 않는다.
        출력은 validate_output()을 통해 프로젝트 원칙을 강제한다.
        """
        llm_input = LLMInput(
            korean_draft=korean_draft,
            top_k_gloss=gloss_hypotheses,
            nms_summary=nms_summary,
            confidence=confidence,
            previous_turns=self._turn_history[-self.max_prev_turns:],
            domain=domain,
            retry_or_clarify=retry_or_clarify,
            gloss_confidences=gloss_confidences or [],
        )

        raw_output = self.provider.correct(llm_input)

        # 출력 검증 — 의미 보존 원칙 강제
        output = validate_output(raw_output, llm_input)

        # confidence 기반 uncertain_spans 자동 보강
        output = self._apply_confidence_policy(output, llm_input)

        # 성공한 발화를 history에 추가 (재질문 발화는 제외)
        if not output.retry_or_clarify and output.final_text:
            self._turn_history.append(output.final_text)
            self._turn_history = self._turn_history[-self.max_prev_turns:]

        return output

    def reset_history(self) -> None:
        self._turn_history.clear()

    def _apply_confidence_policy(
        self, output: LLMOutput, llm_input: LLMInput
    ) -> LLMOutput:
        """low confidence일 때 uncertain_spans를 추가해 하위 시스템에 불확실성을 알린다."""
        if llm_input.confidence >= self.low_confidence_threshold:
            return output
        if output.uncertain_spans:
            return output  # 이미 LLM이 uncertain_spans를 채운 경우

        return LLMOutput(
            final_text=output.final_text,
            uncertain_spans=[{
                "text": output.final_text,
                "reason": f"low_confidence({llm_input.confidence:.2f})",
            }],
            retry_or_clarify=output.retry_or_clarify,
            normalization_notes=output.normalization_notes,
            raw_response=output.raw_response,
        )
