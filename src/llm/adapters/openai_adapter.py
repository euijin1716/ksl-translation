"""OpenAI ChatCompletion LLM adapter.

openai SDK가 설치되어 있어야 한다.
API 오류 시 지수 백오프로 최대 max_retries회 재시도한다.
"""

from __future__ import annotations

import logging
import time

from ..prompt_builder import SYSTEM_PROMPT, build_prompt
from ..provider import LLMInput, LLMOutput, LLMProvider
from ..response_parser import parse_response

logger = logging.getLogger(__name__)

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

_RETRYABLE_STATUS = {429, 500, 502, 503}


class OpenAIAdapter(LLMProvider):
    """OpenAI ChatCompletion API를 사용하는 LLM adapter.

    Args:
        model: OpenAI 모델 ID (기본: gpt-4o)
        api_key: OpenAI API key (없으면 환경변수 OPENAI_API_KEY 사용)
        max_tokens: 최대 출력 토큰 수
        max_retries: API 오류 시 최대 재시도 횟수
        retry_base_delay: 재시도 초기 대기 시간(초), 지수 백오프 적용
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. pip install openai")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def correct(self, llm_input: LLMInput) -> LLMOutput:
        prompt = build_prompt(llm_input)
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = response.choices[0].message.content or ""
                return parse_response(raw, fallback_text=llm_input.korean_draft)

            except Exception as e:
                last_exc = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status not in _RETRYABLE_STATUS and not _is_transient(e):
                    break
                delay = self.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "OpenAI API error (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt + 1, self.max_retries, e, delay,
                )
                time.sleep(delay)

        logger.error("OpenAI API failed after %d attempts: %s", self.max_retries, last_exc)
        return LLMOutput(
            final_text=llm_input.korean_draft,
            normalization_notes=f"api_error: {last_exc}",
        )

    def health_check(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "network"))
