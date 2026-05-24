"""Claude (Anthropic) LLM adapter.

anthropic SDK가 설치되어 있어야 한다.
"""

from __future__ import annotations

import logging

from ..prompt_builder import SYSTEM_PROMPT, build_prompt
from ..provider import LLMInput, LLMOutput, LLMProvider
from ..response_parser import parse_response

logger = logging.getLogger(__name__)

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


class ClaudeAdapter(LLMProvider):
    """Claude API를 사용하는 LLM adapter.

    Args:
        model: Claude 모델 ID (기본: claude-sonnet-4-6)
        api_key: Anthropic API key (없으면 환경변수 ANTHROPIC_API_KEY 사용)
        max_tokens: 최대 출력 토큰 수
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. pip install anthropic")
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def correct(self, llm_input: LLMInput) -> LLMOutput:
        prompt = build_prompt(llm_input)
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            return parse_response(raw, fallback_text=llm_input.korean_draft)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return LLMOutput(
                final_text=llm_input.korean_draft,
                normalization_notes=f"api_error: {e}",
            )

    def health_check(self) -> bool:
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False
