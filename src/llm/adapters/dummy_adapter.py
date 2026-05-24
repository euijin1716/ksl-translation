"""더미 LLM adapter.

실제 API 없이 smoke test를 가능하게 한다.
"""

from __future__ import annotations

from ..provider import LLMInput, LLMOutput, LLMProvider


class DummyLLMAdapter(LLMProvider):
    """테스트 및 오프라인 환경용 더미 LLM adapter.

    입력 초안을 그대로 반환한다.
    """

    def correct(self, llm_input: LLMInput) -> LLMOutput:
        return LLMOutput(
            final_text=llm_input.korean_draft,
            uncertain_spans=[],
            retry_or_clarify=False,
            normalization_notes="dummy_passthrough",
        )

    def health_check(self) -> bool:
        return True
