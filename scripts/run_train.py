#!/usr/bin/env python3
"""학습 실행 스크립트.

사용법:
    python scripts/run_train.py --config configs/stage_c.yaml
    python scripts/run_train.py --config configs/stage_c.yaml --device cpu
"""

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--stage", default=None, help="Stage override; only C is supported")
    p.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    p.add_argument("--batch_size", type=int, default=None, help="Override train batch size")
    p.add_argument("--num_workers", type=int, default=8, help="Override DataLoader worker count")
    p.add_argument("--max_epochs", type=int, default=None, help="Override max epochs")
    p.add_argument("--amp", dest="use_amp", action="store_true", help="Enable CUDA mixed precision")
    p.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable CUDA mixed precision")
    p.set_defaults(use_amp=None)
    p.add_argument("--dummy", action="store_true",
                   help="Force dummy dataset (랜덤 텐서). manifest가 없을 때도 동작")
    p.add_argument("--manifest", default=None,
                   help="학습 manifest 경로. 지정하면 KeypointDataset 사용 "
                        "(예: data/manifests/train.jsonl)")
    p.add_argument("--gloss_vocab", default=None,
                   help="gloss vocab JSON 경로. 없으면 manifest에서 자동 구축")
    p.add_argument("--resume", default=None,
                   help="체크포인트 경로에서 학습 재개 (예: checkpoints/C/epoch_0050.pt)")
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
    """klue/roberta-base tokenizer를 로드한다.

    transformers가 없으면 None을 반환하고 경고를 출력한다.
    """
    try:
        from transformers import AutoTokenizer
        name = tok_cfg.get("name", "klue/roberta-base")
        tokenizer = AutoTokenizer.from_pretrained(name)
        logger.info(
            f"Tokenizer loaded: {name} | vocab_size={tokenizer.vocab_size} "
            f"pad={tokenizer.pad_token_id} bos={tokenizer.bos_token_id or tokenizer.cls_token_id} "
            f"eos={tokenizer.eos_token_id or tokenizer.sep_token_id}"
        )
        return tokenizer
    except ImportError:
        logger.warning(
            "transformers not installed. Tokenizer disabled. "
            "Install with: pip install transformers"
        )
        return None
    except Exception as e:
        logger.warning(f"Tokenizer load failed ({e}). Falling back to None.")
        return None


def build_decoder_config(decoder_cfg: dict, tokenizer):
    """DecoderConfig을 생성한다. tokenizer가 있으면 vocab_size와 특수토큰 ID를 거기서 가져온다."""
    from src.models.decoder import DecoderConfig

    if tokenizer is not None:
        return DecoderConfig.from_tokenizer(
            tokenizer,
            d_model=decoder_cfg.get("d_model", 256),
            nhead=decoder_cfg.get("nhead", 4),
            num_layers=decoder_cfg.get("num_layers", 4),
            dim_feedforward=decoder_cfg.get("dim_feedforward", 512),
            dropout=decoder_cfg.get("dropout", 0.1),
            max_len=decoder_cfg.get("max_len", 128),
        )

    # tokenizer 없이 config 값 직접 사용
    return DecoderConfig(
        d_model=decoder_cfg.get("d_model", 256),
        nhead=decoder_cfg.get("nhead", 4),
        num_layers=decoder_cfg.get("num_layers", 4),
        dim_feedforward=decoder_cfg.get("dim_feedforward", 512),
        dropout=decoder_cfg.get("dropout", 0.1),
        vocab_size=decoder_cfg.get("vocab_size", 32000),
        max_len=decoder_cfg.get("max_len", 128),
        pad_token_id=decoder_cfg.get("pad_token_id", 1),
        bos_token_id=decoder_cfg.get("bos_token_id", 0),
        eos_token_id=decoder_cfg.get("eos_token_id", 2),
    )


def main():
    args = parse_args()

    # Ada Lovelace(RTX 40xx)+ 에서 TF32 matmul과 cuDNN benchmark를 활성화한다.
    # TF32는 FP32 범위를 유지하면서 텐서코어를 사용해 matmul 속도를 높인다.
    # benchmark=True는 입력 크기가 일정할 때 최적 cuDNN 알고리즘을 자동 선택한다.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ── Config 로드 ────────────────────────────────────────────────────────
    base_cfg: dict = {}
    base_path = Path("configs/base.yaml")
    if base_path.exists():
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}

    with open(args.config) as f:
        stage_cfg = yaml.safe_load(f) or {}

    cfg = _merge_cfg(base_cfg, stage_cfg)

    if args.stage and args.stage != "C":
        raise ValueError("Only Stage C is supported.")
    if args.stage:
        cfg["model"]["stage"] = args.stage
        cfg["train"]["stage"] = args.stage
    if args.device:
        cfg["train"]["device"] = args.device
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.use_amp is not None:
        cfg["train"]["use_amp"] = args.use_amp

    stage = cfg["model"]["stage"]
    if stage != "C" or cfg["train"]["stage"] != "C":
        raise ValueError("Only Stage C training is supported.")
    logger.info(f"Starting training: Stage {stage}")

    # ── Tokenizer 로드 (config 준비 후) ────────────────────────────────────
    tok_cfg = cfg.get("tokenizer", {})
    tokenizer = load_tokenizer(tok_cfg)

    # ── Imports ────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    from src.data.dummy import DummyDataset, collate_fn
    from src.models.fusion import FusionConfig
    from src.models.heads import HeadsConfig
    from src.models.ksl_model import KSLModel, ModelConfig
    from src.models.streams.face_expr_encoder import FaceExprEncoderConfig
    from src.models.streams.hand_visual_encoder import HandVisualEncoderConfig
    from src.models.streams.landmark_encoder import LandmarkEncoderConfig
    from src.train.trainer import KSLTrainer, TrainerConfig

    # ── DecoderConfig: tokenizer에서 vocab_size와 특수토큰 ID 가져오기 ─────
    decoder_cfg = build_decoder_config(cfg["model"].get("decoder", {}), tokenizer)
    decoder_vocab_size = decoder_cfg.vocab_size

    # ── Dataset 선택: --manifest 있으면 KeypointDataset, 없거나 --dummy면 DummyDataset
    use_keypoint = (not args.dummy) and (args.manifest is not None) and Path(args.manifest).exists()

    if use_keypoint:
        from src.data.gloss_vocab import GlossVocab
        from src.data.keypoint_dataset import KeypointDataset

        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        landmark_cfg = model_cfg.get("landmark", {})
        tokenizer_cfg = cfg.get("tokenizer", {})
        keypoint_root = data_cfg.get("keypoint_root", "data/keypoints")
        crop_root = data_cfg.get("crop_root", "data/crops")
        manifest_dir = Path(args.manifest).parent
        max_seq_len = landmark_cfg.get("max_seq_len", 512)
        max_text_len = tokenizer_cfg.get("max_length", 64)

        # gloss vocab: 지정 파일 또는 train manifest에서 자동 구축
        if args.gloss_vocab and Path(args.gloss_vocab).exists():
            gloss_vocab = GlossVocab.load(args.gloss_vocab)
        else:
            train_manifest = manifest_dir / "train.jsonl"
            src_manifest = train_manifest if train_manifest.exists() else Path(args.manifest)
            gloss_vocab = GlossVocab.build_from_manifest(src_manifest)
            vocab_save_path = manifest_dir / "gloss_vocab.json"
            gloss_vocab.save(vocab_save_path)
            logger.info(f"GlossVocab built: {len(gloss_vocab)} tokens → {vocab_save_path}")

        enable_hand_visual = cfg.get("model", {}).get("enable_hand_visual", False)
        load_hand_crops = data_cfg.get("load_hand_crops", True) and enable_hand_visual
        if not enable_hand_visual and data_cfg.get("load_hand_crops", True):
            logger.info("enable_hand_visual=False → load_hand_crops 자동 비활성")

        dataset_kwargs = {
            "keypoint_root": keypoint_root,
            "crop_root": crop_root,
            "load_hand_crops": load_hand_crops,
            "sampling_strategy": data_cfg.get("sequence_sampling", "uniform"),
            "boundary_mode": data_cfg.get("boundary_mode", "annotation_or_motion"),
            "gloss_vocab": gloss_vocab,
            "tokenizer": tokenizer,
            "max_seq_len": max_seq_len,
            "max_text_len": max_text_len,
        }

        train_manifest = manifest_dir / "train.jsonl"
        valid_manifest = manifest_dir / "valid.jsonl"

        train_ds = KeypointDataset(
            manifest_path=train_manifest if train_manifest.exists() else Path(args.manifest),
            split_group="train",
            **dataset_kwargs,
        )
        val_ds = KeypointDataset(
            manifest_path=valid_manifest if valid_manifest.exists() else Path(args.manifest),
            split_group="valid",
            **dataset_kwargs,
        )
        logger.info(f"Using KeypointDataset: train={len(train_ds)}, valid={len(val_ds)}")
        logger.info("Train domain distribution: %s", dict(train_ds.domain_counts))
        logger.info("Valid domain distribution: %s", dict(val_ds.domain_counts))

    else:
        if args.manifest and not Path(args.manifest).exists():
            logger.warning(f"manifest not found: {args.manifest}. Falling back to DummyDataset.")
        train_ds = DummyDataset(
            split_group="train",
            num_samples=20,
            decoder_vocab_size=decoder_vocab_size,
            decoder_bos_id=decoder_cfg.bos_token_id,
            decoder_eos_id=decoder_cfg.eos_token_id,
            decoder_pad_id=decoder_cfg.pad_token_id,
        )
        val_ds = DummyDataset(
            split_group="valid",
            num_samples=20,
            decoder_vocab_size=decoder_vocab_size,
            decoder_bos_id=decoder_cfg.bos_token_id,
            decoder_eos_id=decoder_cfg.eos_token_id,
            decoder_pad_id=decoder_cfg.pad_token_id,
        )
        logger.info("Using DummyDataset (random tensors)")

    tc = cfg["train"]
    num_workers = int(tc.get("num_workers", 0))
    pin_memory = tc.get("pin_memory", tc.get("device") == "cuda")
    loader_kwargs = {
        "batch_size": tc["batch_size"],
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(tc.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(tc.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── Model ──────────────────────────────────────────────────────────────
    mc = cfg["model"]
    heads_cfg = _dataclass_from_dict(
        HeadsConfig,
        mc.get("heads", {}),
        gloss_vocab_size=(
            len(gloss_vocab)
            if use_keypoint
            else mc.get("heads", {}).get("gloss_vocab_size", 1001)
        ),
    )
    model_config = ModelConfig(
        stage=mc["stage"],
        landmark=_dataclass_from_dict(LandmarkEncoderConfig, mc.get("landmark", {})),
        hand_visual=_dataclass_from_dict(HandVisualEncoderConfig, mc.get("hand_visual", {})),
        face_expr=_dataclass_from_dict(FaceExprEncoderConfig, mc.get("face_expr", {})),
        fusion=_dataclass_from_dict(FusionConfig, mc.get("fusion", {})),
        heads=heads_cfg,
        decoder=decoder_cfg,
        enable_hand_visual=mc.get("enable_hand_visual", False),
    )
    logger.info(
        "Model config: stage=%s gloss_vocab_size=%s max_seq_len=%s",
        model_config.stage,
        model_config.heads.gloss_vocab_size,
        model_config.landmark.max_seq_len,
    )
    model = KSLModel(model_config)

    train_config = TrainerConfig(
        stage=tc["stage"],
        max_epochs=tc["max_epochs"],
        lr=tc["lr"],
        weight_decay=tc.get("weight_decay", 1e-4),
        grad_clip=tc.get("grad_clip", 1.0),
        batch_size=tc["batch_size"],
        num_workers=tc.get("num_workers", 0),
        device=tc["device"],
        checkpoint_dir=tc["checkpoint_dir"],
        log_dir=tc.get("log_dir", "logs"),
        save_every_n_epochs=tc.get("save_every_n_epochs", 5),
        val_every_n_epochs=tc.get("val_every_n_epochs", 1),
        early_stop_patience=tc.get("early_stop_patience", 0),
        early_stop_min_delta=tc.get("early_stop_min_delta", 0.0),
        monitor_metric=tc.get("monitor_metric", "val_loss"),
        monitor_mode=tc.get("monitor_mode", "min"),
        loss_weight_gloss=tc.get("loss_weight_gloss", 1.0),
        loss_weight_nms=tc.get("loss_weight_nms", 0.5),
        loss_weight_nms_detail=tc.get("loss_weight_nms_detail", 0.3),
        loss_weight_intent=tc.get("loss_weight_intent", 0.3),
        loss_weight_boundary=tc.get("loss_weight_boundary", 0.5),
        loss_weight_draft=tc.get("loss_weight_draft", 1.0),
        use_amp=tc.get("use_amp", tc.get("device") == "cuda"),
    )

    trainer = KSLTrainer(model, train_config, train_loader, val_loader)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
