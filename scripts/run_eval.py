#!/usr/bin/env python3
"""Run evaluation on a real manifest-backed KSL split."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _model_runtime import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--manifest", default="data/manifests/test.jsonl")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--stage", default=None, help="Stage override; only C is supported")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gloss_vocab", default="data/manifests/gloss_vocab.json")
    p.add_argument("--output", default="eval_results/stage_c_test.json")
    p.add_argument(
        "--draft_mode",
        default=None,
        choices=["teacher", "greedy"],
        help="teacher is faster; greedy is closer to real inference but slower. Defaults to config eval.draft_mode or greedy.",
    )
    return p.parse_args()


def _merge_cfg(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _merge_cfg(result[k], v)
        else:
            result[k] = v
    return result


def _dataclass_from_dict(cls, cfg: dict | None, **overrides):
    cfg = dict(cfg or {})
    cfg.update(overrides)
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg.items() if k in allowed})


def load_tokenizer(tok_cfg: dict):
    from transformers import AutoTokenizer

    name = tok_cfg.get("name", "klue/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(name)
    logger.info(
        "Tokenizer loaded: %s | vocab_size=%s pad=%s bos=%s eos=%s",
        name,
        tokenizer.vocab_size,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id or tokenizer.cls_token_id,
        tokenizer.eos_token_id or tokenizer.sep_token_id,
    )
    return tokenizer


def build_decoder_config(decoder_cfg: dict, tokenizer):
    from src.models.decoder import DecoderConfig

    return DecoderConfig.from_tokenizer(
        tokenizer,
        d_model=decoder_cfg.get("d_model", 256),
        nhead=decoder_cfg.get("nhead", 4),
        num_layers=decoder_cfg.get("num_layers", 4),
        dim_feedforward=decoder_cfg.get("dim_feedforward", 512),
        dropout=decoder_cfg.get("dropout", 0.1),
        max_len=decoder_cfg.get("max_len", 128),
    )


def main():
    args = parse_args()

    base_cfg = {}
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    stage_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    cfg = _merge_cfg(base_cfg, stage_cfg)
    if args.stage and args.stage != "C":
        raise ValueError("Only Stage C is supported.")
    if args.stage:
        cfg["model"]["stage"] = args.stage
    if cfg["model"]["stage"] != "C":
        raise ValueError("Only Stage C evaluation is supported.")

    from torch.utils.data import DataLoader

    from src.data.dummy import collate_fn
    from src.data.gloss_vocab import GlossVocab
    from src.data.keypoint_dataset import KeypointDataset
    from src.eval.evaluator import KSLEvaluator
    from src.models.fusion import FusionConfig
    from src.models.heads import HeadsConfig
    from src.models.ksl_model import KSLModel, ModelConfig
    from src.models.streams.face_expr_encoder import FaceExprEncoderConfig
    from src.models.streams.hand_visual_encoder import HandVisualEncoderConfig
    from src.models.streams.landmark_encoder import LandmarkEncoderConfig

    tokenizer = load_tokenizer(cfg.get("tokenizer", {}))
    gloss_vocab = GlossVocab.load(args.gloss_vocab)
    model_cfg = cfg.get("model", {})
    landmark_cfg = model_cfg.get("landmark", {})
    tokenizer_cfg = cfg.get("tokenizer", {})

    dataset = KeypointDataset(
        manifest_path=args.manifest,
        keypoint_root=cfg.get("data", {}).get("keypoint_root", "data/keypoints"),
        crop_root=cfg.get("data", {}).get("crop_root", "data/crops"),
        split_group=args.split,
        gloss_vocab=gloss_vocab,
        tokenizer=tokenizer,
        max_seq_len=landmark_cfg.get("max_seq_len", 512),
        max_text_len=tokenizer_cfg.get("max_length", 64),
        load_hand_crops=cfg.get("data", {}).get("load_hand_crops", True),
        sampling_strategy=cfg.get("data", {}).get("sequence_sampling", "uniform"),
        boundary_mode=cfg.get("data", {}).get("boundary_mode", "annotation_or_motion"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    decoder_cfg = build_decoder_config(model_cfg.get("decoder", {}), tokenizer)
    heads_cfg = _dataclass_from_dict(
        HeadsConfig,
        model_cfg.get("heads", {}),
        gloss_vocab_size=len(gloss_vocab),
    )
    model_config = ModelConfig(
        stage=model_cfg.get("stage", "C"),
        landmark=_dataclass_from_dict(LandmarkEncoderConfig, model_cfg.get("landmark", {})),
        hand_visual=_dataclass_from_dict(HandVisualEncoderConfig, model_cfg.get("hand_visual", {})),
        face_expr=_dataclass_from_dict(FaceExprEncoderConfig, model_cfg.get("face_expr", {})),
        fusion=_dataclass_from_dict(FusionConfig, model_cfg.get("fusion", {})),
        heads=heads_cfg,
        decoder=decoder_cfg,
        enable_hand_visual=model_cfg.get("enable_hand_visual", False),
    )
    model = KSLModel(model_config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = load_checkpoint(model, ckpt_path, args.device)
    logger.info(
        "Loaded checkpoint: %s | global_step=%s best_val_loss=%s load_report=%s",
        ckpt_path,
        ckpt.get("global_step"),
        ckpt.get("best_val_loss"),
        ckpt.get("load_report"),
    )

    evaluator = KSLEvaluator(model, device=args.device)
    draft_mode = args.draft_mode or cfg.get("eval", {}).get("draft_mode", "greedy")
    result = evaluator.evaluate(
        loader,
        split=args.split,
        tokenizer=tokenizer,
        gloss_vocab=gloss_vocab,
        draft_mode=draft_mode,
    )
    logger.info(result.summary())
    evaluator.save_result(result, args.output)


if __name__ == "__main__":
    main()
