"""Shared runtime helpers for eval/prediction scripts."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml


def merge_cfg(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_cfg(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    base_cfg = {}
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    stage_cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return merge_cfg(base_cfg, stage_cfg)


def dataclass_from_dict(cls, cfg: dict | None, **overrides):
    cfg = dict(cfg or {})
    cfg.update(overrides)
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg.items() if k in allowed})


def load_tokenizer(tok_cfg: dict):
    from transformers import AutoTokenizer

    name = tok_cfg.get("name", "klue/roberta-base")
    return AutoTokenizer.from_pretrained(name)


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


def build_model(cfg: dict, tokenizer, gloss_vocab_size: int):
    from src.models.fusion import FusionConfig
    from src.models.heads import HeadsConfig
    from src.models.ksl_model import KSLModel, ModelConfig
    from src.models.streams.face_expr_encoder import FaceExprEncoderConfig
    from src.models.streams.hand_visual_encoder import HandVisualEncoderConfig
    from src.models.streams.landmark_encoder import LandmarkEncoderConfig

    model_cfg = cfg.get("model", {})
    if model_cfg.get("stage", "C") != "C":
        raise ValueError("Only Stage C is supported.")
    decoder_cfg = build_decoder_config(model_cfg.get("decoder", {}), tokenizer)
    heads_cfg = dataclass_from_dict(
        HeadsConfig,
        model_cfg.get("heads", {}),
        gloss_vocab_size=gloss_vocab_size,
    )
    model_config = ModelConfig(
        stage=model_cfg.get("stage", "C"),
        landmark=dataclass_from_dict(LandmarkEncoderConfig, model_cfg.get("landmark", {})),
        hand_visual=dataclass_from_dict(HandVisualEncoderConfig, model_cfg.get("hand_visual", {})),
        face_expr=dataclass_from_dict(FaceExprEncoderConfig, model_cfg.get("face_expr", {})),
        fusion=dataclass_from_dict(FusionConfig, model_cfg.get("fusion", {})),
        heads=heads_cfg,
        decoder=decoder_cfg,
        enable_hand_visual=model_cfg.get("enable_hand_visual", False),
    )
    return KSLModel(model_config)


def load_checkpoint(model, checkpoint: str | Path, device: str, strict: bool = False):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    model_state = model.state_dict()
    filtered = {}
    mismatched: list[str] = []

    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            mismatched.append(key)
            continue
        filtered[key] = value

    missing = [key for key in model_state if key not in filtered]
    unexpected = [key for key in state if key not in model_state]
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            "Checkpoint is not strictly compatible: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    model.load_state_dict(filtered, strict=False)
    ckpt["load_report"] = {
        "loaded": len(filtered),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }
    return ckpt
