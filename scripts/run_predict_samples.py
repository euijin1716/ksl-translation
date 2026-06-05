#!/usr/bin/env python3
"""Write human-readable predictions for a few manifest samples."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _model_runtime import build_model, load_checkpoint, load_config, load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/C/best.pt")
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--manifest", default="data/manifests/test.jsonl")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--gloss_vocab", default="data/manifests/gloss_vocab.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--draft_mode", choices=["teacher", "greedy", "both"], default="both")
    p.add_argument("--output", default="eval_results/stage_c_sample_predictions.jsonl")
    p.add_argument(
        "--enable_hand_visual",
        action="store_true",
        help="E2(hand visual) 스트림을 켜서 모델을 빌드한다. "
        "E2 포함(3-스트림)으로 학습된 체크포인트를 인스펙션할 때 사용.",
    )
    return p.parse_args()


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _ctc_decode(logits: torch.Tensor, seq_len: int, gloss_vocab) -> list[str]:
    ids = logits.argmax(dim=-1)[0, :seq_len].detach().cpu().tolist()
    collapsed: list[int] = []
    prev = None
    for idx in ids:
        if idx != 0 and idx != prev:
            collapsed.append(idx)
        prev = idx
    return gloss_vocab.decode(collapsed)


def _decode_draft_from_logits(logits: torch.Tensor, tokenizer) -> str:
    ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _decode_draft_tokens(tokens: torch.Tensor, tokenizer) -> str:
    ids = tokens[0].detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _decode_nms(outputs: dict[str, torch.Tensor], seq_len: int) -> dict[str, Any]:
    from src.data.signals import NMS_DETAIL_CLASSES, NMS_KEYS

    logits = outputs["nms_logits"]
    pooled = logits[0, :seq_len].sigmoid().mean(dim=0).detach().cpu()
    summary: dict[str, Any] = {
        key: {
            "prob": round(float(pooled[i]), 4),
            "pred": bool(float(pooled[i]) >= 0.5),
        }
        for i, key in enumerate(NMS_KEYS)
        if i < len(pooled)
    }
    for group, classes in NMS_DETAIL_CLASSES.items():
        key = f"nms_{group}_logits"
        if key not in outputs:
            continue
        detail_probs = outputs[key][0, :seq_len].softmax(dim=-1).mean(dim=0).detach().cpu()
        pred_idx = int(detail_probs.argmax().item())
        summary[f"{group}_detail"] = {
            "label": classes[pred_idx],
            "confidence": round(float(detail_probs[pred_idx]), 4),
        }
    return summary


@torch.no_grad()
def main():
    args = parse_args()

    from src.data.dummy import collate_fn
    from src.data.gloss_vocab import GlossVocab
    from src.data.keypoint_dataset import KeypointDataset

    cfg = load_config(args.config)
    if args.enable_hand_visual:
        cfg.setdefault("model", {})["enable_hand_visual"] = True
    tokenizer = load_tokenizer(cfg.get("tokenizer", {}))
    gloss_vocab = GlossVocab.load(args.gloss_vocab)
    model = build_model(cfg, tokenizer, gloss_vocab_size=len(gloss_vocab))
    ckpt = load_checkpoint(model, args.checkpoint, args.device)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint: %s | global_step=%s best_val_loss=%s",
        args.checkpoint,
        ckpt.get("global_step"),
        ckpt.get("best_val_loss"),
    )

    model_cfg = cfg.get("model", {})
    tokenizer_cfg = cfg.get("tokenizer", {})
    dataset = KeypointDataset(
        manifest_path=args.manifest,
        keypoint_root=cfg.get("data", {}).get("keypoint_root", "data/keypoints"),
        split_group=args.split,
        gloss_vocab=gloss_vocab,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.get("landmark", {}).get("max_seq_len", 512),
        max_text_len=tokenizer_cfg.get("max_length", 64),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end = min(args.start + args.num_samples, len(dataset))
    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(args.start, end):
            batch = collate_fn([dataset[idx]])
            batch = _to_device(batch, device)
            seq_len = int(batch["seq_len"][0].item())
            T = batch["pose"].shape[1]
            mask = torch.arange(T, device=device).unsqueeze(0) >= batch["seq_len"].unsqueeze(1)

            teacher_outputs = None
            greedy_outputs = None
            if args.draft_mode in ("teacher", "both"):
                teacher_outputs = model(
                    pose=batch["pose"],
                    left_hand=batch["left_hand"],
                    right_hand=batch["right_hand"],
                    face_blendshape=batch["face_blendshape"],
                    face_key_subset=batch.get("face_key_subset"),
                    presence_mask=batch.get("presence_mask"),
                    tgt_tokens=batch.get("tgt_tokens"),
                    tgt_padding=batch.get("tgt_padding"),
                    src_key_padding_mask=mask,
                )
            if args.draft_mode in ("greedy", "both"):
                greedy_outputs = model(
                    pose=batch["pose"],
                    left_hand=batch["left_hand"],
                    right_hand=batch["right_hand"],
                    face_blendshape=batch["face_blendshape"],
                    face_key_subset=batch.get("face_key_subset"),
                    presence_mask=batch.get("presence_mask"),
                    src_key_padding_mask=mask,
                )

            outputs = teacher_outputs or greedy_outputs
            record = {
                "index": idx,
                "sample_id": batch["sample_id"][0],
                "domain": batch["domain"][0],
                "reference_text": batch.get("korean_text", [""])[0],
                "reference_gloss": batch.get("gloss_tokens_raw", [[]])[0],
                "reference_nms": batch.get("nms_labels_raw", [{}])[0],
                "pred_gloss": _ctc_decode(outputs["gloss_logits"], seq_len, gloss_vocab),
                "pred_nms": _decode_nms(outputs, seq_len),
                "intent_pred_id": int(outputs["intent_logits"].argmax(dim=-1)[0].item()),
            }
            if teacher_outputs is not None and "draft_logits" in teacher_outputs:
                record["draft_teacher_forced"] = _decode_draft_from_logits(
                    teacher_outputs["draft_logits"], tokenizer
                )
            if greedy_outputs is not None and "draft_tokens" in greedy_outputs:
                record["draft_greedy"] = _decode_draft_tokens(
                    greedy_outputs["draft_tokens"], tokenizer
                )

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved sample predictions: %s (%s samples)", out_path, end - args.start)


if __name__ == "__main__":
    main()
