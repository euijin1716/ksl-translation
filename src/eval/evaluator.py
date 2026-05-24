"""KSL 평가기.

4계층 평가: 추출 / 인식 / 번역 / 문맥 보정.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..data.signals import NMS_DETAIL_GROUPS
from .metrics import compute_accuracy, compute_bleu, compute_chrf, compute_f1, compute_wer

logger = logging.getLogger(__name__)


def _ctc_collapse(ids: list[int], blank_id: int = 0) -> list[int]:
    result: list[int] = []
    prev = None
    for idx in ids:
        if idx != blank_id and idx != prev:
            result.append(idx)
        prev = idx
    return result


def _decode_token_ids(tokenizer: Any, ids: list[int]) -> str:
    if tokenizer is None:
        return " ".join(str(i) for i in ids)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


@dataclass
class EvalResult:
    """평가 결과 컨테이너."""
    split: str
    num_samples: int
    draft_mode: str = "teacher"

    # 인식
    intent_accuracy: float = 0.0
    boundary_f1: float = 0.0
    gloss_wer: float = 0.0
    nms_f1: float = 0.0
    nms_detail_accuracy: dict[str, float] = field(default_factory=dict)

    # 번역
    bleu: float = 0.0
    chrf: float = 0.0

    # 도메인별 BLEU
    domain_bleu: dict[str, float] = field(default_factory=dict)
    domain_distribution: dict[str, int] = field(default_factory=dict)
    intent_label_distribution: dict[str, int] = field(default_factory=dict)
    boundary_label_distribution: dict[str, int] = field(default_factory=dict)
    metric_warnings: list[str] = field(default_factory=list)

    # 오류 샘플
    error_samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"[{self.split}] n={self.num_samples} draft={self.draft_mode} | "
            f"intent_acc={self.intent_accuracy:.3f} | "
            f"boundary_f1={self.boundary_f1:.3f} | "
            f"gloss_wer={self.gloss_wer:.3f} | "
            f"nms_f1={self.nms_f1:.3f} | "
            f"nms_detail={self.nms_detail_accuracy} | "
            f"BLEU={self.bleu:.2f} | chrF={self.chrf:.2f}"
        )


class KSLEvaluator:
    """모델 성능 평가기.

    Args:
        model: KSLModel (eval 모드)
        device: 평가 디바이스
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split: str = "test",
        tokenizer: Any = None,
        gloss_vocab: Any = None,
        draft_mode: str = "teacher",
    ) -> EvalResult:
        """DataLoader 전체에 대해 평가를 수행한다."""
        all_intent_preds, all_intent_labels = [], []
        all_boundary_preds, all_boundary_labels = [], []
        all_nms_preds, all_nms_labels = [], []
        nms_detail_preds: dict[str, list[int]] = {group: [] for group in NMS_DETAIL_GROUPS}
        nms_detail_labels: dict[str, list[int]] = {group: [] for group in NMS_DETAIL_GROUPS}
        all_hyp_texts, all_ref_texts = [], []
        all_gloss_hyp, all_gloss_ref = [], []
        domain_results: dict[str, list] = {}
        all_domains: list[str] = []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            all_domains.extend([str(d) for d in batch.get("domain", [])])

            seq_len = batch["seq_len"]
            B, T = batch["pose"].shape[:2]
            mask = torch.arange(T, device=self.device).unsqueeze(0) >= seq_len.unsqueeze(1)

            use_teacher = draft_mode == "teacher"
            outputs = self.model(
                pose=batch["pose"],
                left_hand=batch["left_hand"],
                right_hand=batch["right_hand"],
                face_blendshape=batch["face_blendshape"],
                face_key_subset=batch.get("face_key_subset"),
                presence_mask=batch.get("presence_mask"),
                tgt_tokens=batch.get("tgt_tokens") if use_teacher else None,
                tgt_padding=batch.get("tgt_padding") if use_teacher else None,
                src_key_padding_mask=mask,
            )

            # Intent accuracy
            if "intent_logits" in outputs:
                preds = outputs["intent_logits"].argmax(dim=-1).cpu().tolist()
                labels = batch["intent_label"].cpu().tolist()
                all_intent_preds.extend(preds)
                all_intent_labels.extend(labels)

            # Boundary F1
            if "boundary_logits" in outputs and "activity" in batch:
                valid_flat = (~mask).cpu().view(-1)
                bnd_preds = outputs["boundary_logits"].argmax(dim=-1).cpu().view(-1)[valid_flat].tolist()
                bnd_labels = batch["activity"].cpu().view(-1)[valid_flat].tolist()
                all_boundary_preds.extend(bnd_preds)
                all_boundary_labels.extend(bnd_labels)

            # NMS multi-label F1 over annotated fields only.
            if "nms_logits" in outputs and "nms_label" in batch and "nms_mask" in batch:
                nms_logits = outputs["nms_logits"]
                valid = (~mask).float()
                pooled = (nms_logits * valid.unsqueeze(-1)).sum(dim=1)
                pooled = pooled / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
                preds = (torch.sigmoid(pooled) >= 0.5).long().cpu()
                labels = batch["nms_label"].long().cpu()
                nms_mask = batch["nms_mask"].bool().cpu()
                all_nms_preds.extend(preds[nms_mask].tolist())
                all_nms_labels.extend(labels[nms_mask].tolist())

            # Fine NMS categorical accuracy over annotated groups.
            if "nms_detail_label" in batch and "nms_detail_mask" in batch:
                detail_labels = batch["nms_detail_label"].long().cpu()
                detail_mask = batch["nms_detail_mask"].bool().cpu()
                valid = (~mask).float()
                for group_idx, group in enumerate(NMS_DETAIL_GROUPS):
                    key = f"nms_{group}_logits"
                    if key not in outputs:
                        continue
                    logits = outputs[key]
                    pooled = (logits * valid.unsqueeze(-1)).sum(dim=1)
                    pooled = pooled / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
                    preds = pooled.argmax(dim=-1).cpu()
                    supervised = detail_mask[:, group_idx]
                    nms_detail_preds[group].extend(preds[supervised].tolist())
                    nms_detail_labels[group].extend(detail_labels[supervised, group_idx].tolist())

            # Gloss WER from CTC argmax decoding.
            if "gloss_logits" in outputs and "gloss_ids" in batch:
                pred_ids = outputs["gloss_logits"].argmax(dim=-1).cpu()
                for i, ref_ids in enumerate(batch["gloss_ids"]):
                    hyp_ids = _ctc_collapse(pred_ids[i, : int(seq_len[i].item())].tolist())
                    ref_list = ref_ids.cpu().tolist()
                    if gloss_vocab is not None:
                        all_gloss_hyp.append(gloss_vocab.decode(hyp_ids))
                        all_gloss_ref.append(gloss_vocab.decode(ref_list))
                    else:
                        all_gloss_hyp.append([str(x) for x in hyp_ids])
                        all_gloss_ref.append([str(x) for x in ref_list])

            # Draft metrics. Teacher mode uses teacher-forced token logits;
            # greedy mode uses actual autoregressive decoder output.
            if tokenizer is not None and batch.get("korean_text"):
                batch_domains = [str(d) for d in batch.get("domain", ["unknown"] * len(batch["korean_text"]))]
                if "draft_logits" in outputs:
                    pred = outputs["draft_logits"].argmax(dim=-1).cpu()
                    for ids, ref, domain in zip(pred.tolist(), batch["korean_text"], batch_domains):
                        hyp = _decode_token_ids(tokenizer, ids)
                        all_hyp_texts.append(hyp)
                        all_ref_texts.append(ref)
                        domain_results.setdefault(domain, []).append((hyp, ref))
                elif "draft_tokens" in outputs:
                    for ids, ref, domain in zip(outputs["draft_tokens"].cpu().tolist(), batch["korean_text"], batch_domains):
                        hyp = _decode_token_ids(tokenizer, ids)
                        all_hyp_texts.append(hyp)
                        all_ref_texts.append(ref)
                        domain_results.setdefault(domain, []).append((hyp, ref))

        result = EvalResult(split=split, num_samples=len(all_intent_labels), draft_mode=draft_mode)
        result.domain_distribution = dict(Counter(all_domains))
        result.intent_label_distribution = {
            str(k): v for k, v in Counter(all_intent_labels).items()
        }
        result.boundary_label_distribution = {
            str(k): v for k, v in Counter(all_boundary_labels).items()
        }
        if draft_mode == "teacher":
            result.metric_warnings.append(
                "draft metrics are teacher-forced; use draft_mode=greedy for real inference quality."
            )
        if len(result.domain_distribution) <= 1 and all_domains:
            result.metric_warnings.append(
                "single-domain split; intent_accuracy is not a discriminative metric."
            )
        if len(result.boundary_label_distribution) <= 1 and all_boundary_labels:
            result.metric_warnings.append(
                "single-class boundary labels; boundary_f1 is not reliable."
            )

        if all_intent_preds:
            result.intent_accuracy = compute_accuracy(all_intent_preds, all_intent_labels)

        if all_boundary_preds:
            f1_dict = compute_f1(all_boundary_preds, all_boundary_labels, num_classes=3)
            result.boundary_f1 = f1_dict["f1"]

        if all_nms_preds:
            f1_dict = compute_f1(all_nms_preds, all_nms_labels, num_classes=2, average="binary")
            result.nms_f1 = f1_dict["f1"]

        result.nms_detail_accuracy = {
            group: compute_accuracy(nms_detail_preds[group], nms_detail_labels[group])
            for group in NMS_DETAIL_GROUPS
            if nms_detail_preds[group]
        }

        if all_gloss_ref:
            result.gloss_wer = compute_wer(all_gloss_hyp, all_gloss_ref)

        if all_hyp_texts:
            bleu = compute_bleu(all_hyp_texts, all_ref_texts)
            result.bleu = bleu["bleu"]
            result.chrf = compute_chrf(all_hyp_texts, all_ref_texts)
            result.domain_bleu = {
                domain: compute_bleu([h for h, _ in pairs], [r for _, r in pairs])["bleu"]
                for domain, pairs in domain_results.items()
                if pairs
            }

        return result

    def save_result(self, result: EvalResult, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved eval result: {path}")
