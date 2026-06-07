"""오프라인 추론 파이프라인.

영상 입력 → 최종 한국어 문장 출력.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..data.gloss_vocab import GlossVocab
from ..data.signals import NMS_DETAIL_CLASSES, NMS_KEYS
from ..llm.corrector import ContextCorrector

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """추론 파이프라인 최종 출력 계약."""
    final_text: str
    draft_text: str
    confidence: float
    uncertain_spans: list[dict[str, Any]] = field(default_factory=list)
    retry_or_clarify: bool = False
    activity_state: str = "ended"          # "idle" | "ongoing" | "ended"
    gloss_hypotheses: list[str] = field(default_factory=list)
    nms_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class InferencePipeline:
    """KSL → 한국어 오프라인 추론 파이프라인.

    Args:
        model: KSLModel (eval 모드)
        corrector: ContextCorrector
        device: 추론 디바이스
        confidence_threshold: 이 이하면 retry_or_clarify 플래그
    """

    def __init__(
        self,
        model: nn.Module,
        gloss_vocab: GlossVocab,
        corrector: ContextCorrector | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.4,
        tokenizer: Any = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.corrector = corrector or ContextCorrector()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.tokenizer = tokenizer
        self.gloss_vocab = gloss_vocab

    @torch.no_grad()
    def infer(
        self,
        batch: dict[str, Any],
        domain: str = "unknown",
    ) -> InferenceResult:
        """단일 배치(1개 샘플)에 대한 추론을 수행한다.

        내부 중간 출력과 사용자 노출 출력을 분리한다.
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = self.model(
            pose=batch["pose"],
            left_hand=batch["left_hand"],
            right_hand=batch["right_hand"],
            face_blendshape=batch["face_blendshape"],
            face_key_subset=batch.get("face_key_subset"),
            presence_mask=batch.get("presence_mask"),
            left_hand_crop=batch.get("left_hand_crop"),
            right_hand_crop=batch.get("right_hand_crop"),
        )

        # ── 중간 출력 처리 ────────────────────────────────────────────────────
        gloss_pairs = self._decode_gloss(outputs.get("gloss_logits"))   # [(gloss, 신뢰도)]
        gloss_hypotheses = [g for g, _ in gloss_pairs]
        gloss_confidences = [c for _, c in gloss_pairs]
        nms_summary = self._decode_nms(outputs)
        confidence = self._estimate_confidence(outputs)
        draft_text = self._decode_draft(outputs, batch)
        activity_state = self._decode_activity(outputs.get("boundary_logits"))

        # ── gloss/NMS와 draft 강한 충돌 체크 ─────────────────────────────────
        retry_flag = confidence < self.confidence_threshold

        # ── LLM 문맥 보정 ─────────────────────────────────────────────────────
        llm_output = self.corrector.correct(
            korean_draft=draft_text,
            gloss_hypotheses=gloss_hypotheses,
            gloss_confidences=gloss_confidences,
            nms_summary=nms_summary,
            confidence=confidence,
            domain=domain,
            retry_or_clarify=retry_flag,
        )

        return InferenceResult(
            final_text=llm_output.final_text,
            draft_text=draft_text,
            confidence=confidence,
            uncertain_spans=llm_output.uncertain_spans,
            retry_or_clarify=llm_output.retry_or_clarify,
            activity_state=activity_state,
            gloss_hypotheses=gloss_hypotheses,
            nms_summary=nms_summary,
        )

    def _decode_gloss(self, gloss_logits: torch.Tensor | None) -> list[tuple[str, float]]:
        """CTC argmax 1-best gloss + 각 gloss의 신뢰도(emit 프레임 softmax 확률).

        Returns: [(gloss단어, 신뢰도 0~1), ...] (CTC 압축 후, 상위 5개).
        """
        if gloss_logits is None:
            return []
        probs = gloss_logits[0].softmax(dim=-1)          # [T, V]
        ids = probs.argmax(dim=-1).tolist()              # [T]
        # CTC: 연속 중복 제거, blank(0) 제거 + emit 프레임 신뢰도 수집
        prev, emitted = -1, []                           # [(id, conf)]
        for t, tok in enumerate(ids):
            if tok != prev and tok != 0:
                emitted.append((tok, float(probs[t, tok])))
            prev = tok
        emitted = emitted[:5]
        words = self.gloss_vocab.decode([tok for tok, _ in emitted])
        return [(w, round(c, 3)) for w, (_, c) in zip(words, emitted)]

    def _decode_nms(self, outputs: dict[str, Any]) -> dict[str, Any]:
        nms_logits = outputs.get("nms_logits")
        if nms_logits is None:
            summary: dict[str, Any] = {}
        else:
            probs = torch.sigmoid(nms_logits[0]).mean(dim=0)   # [nms_classes]
            summary = {k: round(probs[i].item(), 3) for i, k in enumerate(NMS_KEYS) if i < len(probs)}

        for group, classes in NMS_DETAIL_CLASSES.items():
            key = f"nms_{group}_logits"
            logits = outputs.get(key)
            if logits is None:
                continue
            probs = torch.softmax(logits[0], dim=-1).mean(dim=0)
            cls_idx = int(probs.argmax().item())
            summary[f"{group}_detail"] = {
                "label": classes[cls_idx],
                "confidence": round(probs[cls_idx].item(), 3),
            }

        return summary

    def _estimate_confidence(self, outputs: dict[str, Any]) -> float:
        """Intent logit의 max softmax 확률을 confidence proxy로 사용한다."""
        if "intent_logits" in outputs:
            probs = torch.softmax(outputs["intent_logits"][0], dim=-1)
            return round(probs.max().item(), 4)
        return 0.5

    def _decode_draft(self, outputs: dict[str, Any], batch: dict[str, Any]) -> str:
        if "draft_tokens" in outputs and self.tokenizer is not None:
            return self.tokenizer.decode(outputs["draft_tokens"][0].tolist(), skip_special_tokens=True)
        if "draft_tokens" in outputs:
            return f"[draft tokens: {outputs['draft_tokens'][0][:10].tolist()}...]"
        return "(draft not available)"

    def _decode_activity(self, boundary_logits: torch.Tensor | None) -> str:
        if boundary_logits is None:
            return "ended"
        last = boundary_logits[0, -1, :].argmax().item()
        return {0: "idle", 1: "ongoing", 2: "ended"}.get(last, "ended")
