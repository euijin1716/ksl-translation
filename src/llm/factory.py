"""LLM provider factory.

base.yaml의 llm 섹션을 받아 올바른 adapter + ContextCorrector를 생성한다.
"""

from __future__ import annotations

import logging
from typing import Any

from .corrector import ContextCorrector
from .provider import LLMProvider

logger = logging.getLogger(__name__)

# provider 이름 → adapter 클래스 (lazy import로 선택적 의존성 처리)
_PROVIDER_MAP = {
    "claude": "src.llm.adapters.claude_adapter.ClaudeAdapter",
    "openai": "src.llm.adapters.openai_adapter.OpenAIAdapter",
    "dummy": "src.llm.adapters.dummy_adapter.DummyLLMAdapter",
}


def build_corrector(llm_config: dict[str, Any]) -> ContextCorrector:
    """llm config dict에서 ContextCorrector를 생성한다.

    Args:
        llm_config: base.yaml의 llm 섹션 dict.
            예: {"provider": "claude", "model": "claude-sonnet-4-6",
                 "max_tokens": 512, "max_prev_turns": 5,
                 "confidence_threshold": 0.4}

    Returns:
        구성된 ContextCorrector 인스턴스.
    """
    provider_name = llm_config.get("provider", "dummy").lower()
    provider = _build_provider(provider_name, llm_config)

    return ContextCorrector(
        provider=provider,
        max_prev_turns=int(llm_config.get("max_prev_turns", 5)),
        low_confidence_threshold=float(llm_config.get("confidence_threshold", 0.4)),
    )


def _build_provider(name: str, cfg: dict[str, Any]) -> LLMProvider:
    if name not in _PROVIDER_MAP:
        logger.warning("Unknown LLM provider '%s'; falling back to dummy.", name)
        name = "dummy"

    module_path, class_name = _PROVIDER_MAP[name].rsplit(".", 1)
    # 동적 import — 선택적 의존성(anthropic, openai)이 없어도 dummy는 동작
    try:
        import importlib
        module = importlib.import_module(module_path.replace("src.", "src/").replace("/", "."))
        # 실제 경로는 패키지 구조에 맞게 재조정
        module = importlib.import_module(_module_name(name))
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error("Failed to load provider '%s': %s. Falling back to dummy.", name, e)
        from .adapters.dummy_adapter import DummyLLMAdapter
        return DummyLLMAdapter()

    return _instantiate(cls, name, cfg)


def _module_name(provider: str) -> str:
    return {
        "claude": "src.llm.adapters.claude_adapter",
        "openai": "src.llm.adapters.openai_adapter",
        "dummy": "src.llm.adapters.dummy_adapter",
    }[provider]


def _instantiate(cls: type, name: str, cfg: dict[str, Any]) -> LLMProvider:
    """provider 종류별로 필요한 인자만 골라 인스턴스를 생성한다."""
    common = {
        "max_tokens": int(cfg.get("max_tokens", 512)),
        "max_retries": int(cfg.get("max_retries", 3)),
    }
    try:
        if name == "claude":
            return cls(
                model=cfg.get("model", "claude-sonnet-4-6"),
                api_key=cfg.get("api_key"),
                **common,
            )
        elif name == "openai":
            return cls(
                model=cfg.get("model", "gpt-4o"),
                api_key=cfg.get("api_key"),
                **common,
            )
        else:
            return cls()
    except Exception as e:
        logger.error("Failed to instantiate provider '%s': %s. Falling back to dummy.", name, e)
        from .adapters.dummy_adapter import DummyLLMAdapter
        return DummyLLMAdapter()
