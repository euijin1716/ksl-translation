#!/usr/bin/env python3
"""발표용 데모 스크립트: 수어 영상 한 개 → 한국어 번역 전체 파이프라인.

사용법:
    python scripts/run_demo.py --video path/to/sign_video.mp4
    python scripts/run_demo.py --video path/to/sign_video.mp4 --llm claude
    python scripts/run_demo.py --video path/to/sign_video.mp4 --no_llm   # LLM 없이 draft만
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _model_runtime import build_model, load_checkpoint, load_config, load_tokenizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── 설정 기본값 ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "checkpoint": "checkpoints/C/best.pt",
    "config":     "configs/stage_c.yaml",
    "gloss_vocab":"data/manifests/gloss_vocab.json",
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
    "max_seq_len": 128,
    "domain":     "help",
}

_POSE_JOINTS  = 25
_HAND_JOINTS  = 21
_FACE_BS_DIM  = 52
_FACE_KEY_DIM = 68


# ── 인자 파싱 ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="KSL 수어 영상 → 한국어 번역 데모")
    p.add_argument("--video",      required=True,  help="입력 수어 영상 경로 (.mp4 등)")
    p.add_argument("--checkpoint", default=DEFAULTS["checkpoint"])
    p.add_argument("--config",     default=DEFAULTS["config"])
    p.add_argument("--gloss_vocab",default=DEFAULTS["gloss_vocab"])
    p.add_argument("--device",     default=DEFAULTS["device"])
    p.add_argument("--domain",     default=DEFAULTS["domain"],
                   help="도메인: hospital / directions / order / help 등")
    p.add_argument("--llm",        default="claude", choices=["claude", "openai", "dummy"],
                   help="LLM provider (기본: claude)")
    p.add_argument("--no_llm",     action="store_true", help="LLM 보정 없이 모델 출력만 표시")
    p.add_argument("--keypoint_root", default="data/keypoints",
                   help="추출된 .npy 저장 루트 (없으면 영상에서 실시간 추출)")
    p.add_argument("--skip_extract", action="store_true",
                   help="이미 추출된 .npy가 있으면 추출 단계 생략")
    return p.parse_args()


# ── 영상에서 특징 추출 ────────────────────────────────────────────────────────
def extract_features(video_path: str, keypoint_root: str, sample_id: str) -> Path:
    """MediaPipe로 영상을 처리해 .npy 파일을 저장하고 디렉터리를 반환한다."""
    from src.data.schema import KSLSample
    from src.preprocess.pipelines.extraction_pipeline import ExtractionPipeline

    print("[1/4] MediaPipe 특징 추출 중...")
    t0 = time.time()

    cfg = {
        "keypoint_root": keypoint_root,
        "crop_root": "data/crops",
        "enable_crops": False,          # 데모에서는 crop 불필요
        "save_face_crop": False,
        "normalize_method": "shoulder_width",
        "skip_existing": True,
        "extractor_config": {
            "model_asset_path_hand": "data/mediapipe_models/hand_landmarker.task",
            "model_asset_path_face": "data/mediapipe_models/face_landmarker.task",
            "model_asset_path_pose": "data/mediapipe_models/pose_landmarker_full.task",
            "num_hands": 2,
            "face_blendshape": True,
            "pose_upper_body_only": True,
            "pose_joints": _POSE_JOINTS,
            "target_fps": 25.0,
        },
    }

    dummy_sample = KSLSample(
        sample_id=sample_id,
        dataset_name="demo",
        domain="help",
        video_path=video_path,
        korean_text="",
        gloss_tokens=[],
    )

    with ExtractionPipeline(cfg) as pipeline:
        result_sample = pipeline.process(dummy_sample)

    kp_dir = Path(keypoint_root) / sample_id
    elapsed = time.time() - t0
    print(f"    완료 ({elapsed:.1f}s) → {kp_dir}")
    return kp_dir


# ── .npy → 텐서 변환 ─────────────────────────────────────────────────────────
def load_keypoints_as_batch(kp_dir: Path, max_seq_len: int, device: torch.device) -> dict:
    """저장된 .npy 파일을 로드해 모델 입력 batch dict로 변환한다."""

    def _load(name: str, default_shape: tuple) -> np.ndarray:
        path = kp_dir / f"{name}.npy"
        if path.exists():
            return np.load(path).astype(np.float32)
        T = default_shape[0]
        return np.zeros((T,) + default_shape[1:], dtype=np.float32)

    # 기준 T를 pose에서 가져옴
    pose_path = kp_dir / "pose.npy"
    if not pose_path.exists():
        raise FileNotFoundError(f"pose.npy가 없습니다: {kp_dir}")

    pose_raw = np.load(pose_path).astype(np.float32)
    T_orig = pose_raw.shape[0]

    pose         = _load("pose",          (T_orig, _POSE_JOINTS, 3))
    left_hand    = _load("left_hand",     (T_orig, _HAND_JOINTS, 3))
    right_hand   = _load("right_hand",    (T_orig, _HAND_JOINTS, 3))
    face_bs      = _load("face_blendshape", (T_orig, _FACE_BS_DIM))
    face_key     = _load("face_key_subset", (T_orig, _FACE_KEY_DIM, 3))
    presence     = _load("presence_mask",   (T_orig, 4))

    # max_seq_len 맞춤: 자르거나 패딩
    def _pad_or_crop(arr: np.ndarray) -> np.ndarray:
        T = arr.shape[0]
        if T >= max_seq_len:
            return arr[:max_seq_len]
        pad_shape = (max_seq_len - T,) + arr.shape[1:]
        return np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)], axis=0)

    pose      = _pad_or_crop(pose)
    left_hand = _pad_or_crop(left_hand)
    right_hand= _pad_or_crop(right_hand)
    face_bs   = _pad_or_crop(face_bs)
    face_key  = _pad_or_crop(face_key)
    presence  = _pad_or_crop(presence)

    seq_len = min(T_orig, max_seq_len)
    T = max_seq_len

    # 배치 차원 추가 후 텐서 변환
    def _t(arr):
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    return {
        "pose":            _t(pose),
        "left_hand":       _t(left_hand),
        "right_hand":      _t(right_hand),
        "face_blendshape": _t(face_bs),
        "face_key_subset": _t(face_key),
        "presence_mask":   _t(presence),
        "seq_len":         torch.tensor([seq_len]),
    }


# ── 출력 포매터 ──────────────────────────────────────────────────────────────
def _bar(label: str, width: int = 60):
    print(f"\n{'━' * width}")
    print(f"  {label}")
    print(f"{'━' * width}")


def _conf_bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {val:.2f}"


def print_results(
    video_path: str,
    gloss: list[tuple[str, float]],
    nms: dict,
    boundary: str,
    confidence: float,
    draft_text: str,
    final_text: str | None,
    uncertain_spans: list,
    retry_or_clarify: bool,
    elapsed_infer: float,
    elapsed_llm: float | None,
):
    _bar("KSL → 한국어 번역 결과")
    print(f"  입력 영상 : {video_path}")

    print(f"\n  [Gloss 인식 (CTC)]")
    if gloss:
        for word, conf in gloss:
            print(f"    {word:15s}  {_conf_bar(conf, 12)}")
    else:
        print("    (인식된 gloss 없음)")

    print(f"\n  [비수지신호 (NMS)]")
    for key, val in nms.items():
        if "_detail" in key:
            label = val.get("label", "?")
            c     = val.get("confidence", 0)
            print(f"    {key:25s}  {label} ({c:.2f})")
        elif isinstance(val, dict):
            pred = "✓" if val.get("pred") else "✗"
            prob = val.get("prob", 0)
            print(f"    {key:25s}  {pred}  prob={prob:.3f}")

    print(f"\n  [수어 활동 구간]  {boundary}")
    print(f"  [신뢰도]          {_conf_bar(confidence)}")

    print(f"\n  [모델 초안]")
    print(f"    {draft_text}")

    if final_text is not None:
        print(f"\n  [LLM 보정 최종]")
        print(f"    {final_text}")
        if uncertain_spans:
            print(f"    ⚠ 불확실 구간: {[s.get('text','?') for s in uncertain_spans]}")
        if retry_or_clarify:
            print(f"    ⚠ 재표현 요청 권장")

    print(f"\n  [처리 시간]")
    print(f"    모델 추론 : {elapsed_infer*1000:.0f} ms")
    if elapsed_llm is not None:
        print(f"    LLM 보정  : {elapsed_llm*1000:.0f} ms")
    print()


# ── LLM provider 생성 ─────────────────────────────────────────────────────────
def build_corrector(provider_name: str):
    from src.llm.corrector import ContextCorrector

    if provider_name == "dummy":
        return ContextCorrector()

    if provider_name == "claude":
        import os
        from src.llm.adapters.claude_adapter import ClaudeAdapter
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  ⚠ ANTHROPIC_API_KEY 환경변수가 없습니다. dummy로 대체합니다.")
            return ContextCorrector()
        return ContextCorrector(provider=ClaudeAdapter(api_key=api_key))

    if provider_name == "openai":
        import os
        from src.llm.adapters.openai_adapter import OpenAIAdapter
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  ⚠ OPENAI_API_KEY 환경변수가 없습니다. dummy로 대체합니다.")
            return ContextCorrector()
        return ContextCorrector(provider=OpenAIAdapter(api_key=api_key))

    return ContextCorrector()


# ── 모델 추론 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, batch, device):
    T = batch["pose"].shape[1]
    seq_len = int(batch["seq_len"][0].item())
    mask = torch.arange(T, device=device).unsqueeze(0) >= batch["seq_len"].unsqueeze(1).to(device)

    outputs = model(
        pose=           batch["pose"],
        left_hand=      batch["left_hand"],
        right_hand=     batch["right_hand"],
        face_blendshape=batch["face_blendshape"],
        face_key_subset=batch["face_key_subset"],
        presence_mask=  batch["presence_mask"],
        src_key_padding_mask=mask,
    )
    return outputs, seq_len, mask


def decode_gloss(gloss_logits, seq_len, gloss_vocab) -> list[tuple[str, float]]:
    from src.data.gloss_vocab import GlossVocab
    probs = gloss_logits[0, :seq_len].softmax(dim=-1)
    ids   = probs.argmax(dim=-1).tolist()
    prev, emitted = -1, []
    for t, tok in enumerate(ids):
        if tok != prev and tok != 0:
            emitted.append((tok, float(probs[t, tok])))
        prev = tok
    emitted = emitted[:5]
    words = gloss_vocab.decode([tok for tok, _ in emitted])
    return [(w, round(c, 3)) for w, (_, c) in zip(words, emitted)]


def decode_nms(outputs, seq_len) -> dict:
    from src.data.signals import NMS_DETAIL_CLASSES, NMS_KEYS
    logits = outputs["nms_logits"]
    pooled = logits[0, :seq_len].sigmoid().mean(dim=0).cpu()
    summary = {
        k: {"prob": round(float(pooled[i]), 3), "pred": float(pooled[i]) >= 0.5}
        for i, k in enumerate(NMS_KEYS) if i < len(pooled)
    }
    for group, classes in NMS_DETAIL_CLASSES.items():
        key = f"nms_{group}_logits"
        if key not in outputs:
            continue
        p = outputs[key][0, :seq_len].softmax(dim=-1).mean(dim=0).cpu()
        idx = int(p.argmax())
        summary[f"{group}_detail"] = {"label": classes[idx], "confidence": round(float(p[idx]), 3)}
    return summary


def decode_boundary(boundary_logits) -> str:
    last = boundary_logits[0, -1, :].argmax().item()
    return {0: "idle (수어 없음)", 1: "ongoing (수어 중)", 2: "ended (수어 종료)"}.get(last, "unknown")


def decode_draft(draft_tokens, tokenizer) -> str:
    if tokenizer is None:
        return f"[토큰: {draft_tokens[0][:8].tolist()}...]"
    return tokenizer.decode(draft_tokens[0].tolist(), skip_special_tokens=True).strip()


def estimate_confidence(outputs) -> float:
    if "intent_logits" in outputs:
        return round(float(torch.softmax(outputs["intent_logits"][0], dim=-1).max()), 4)
    return 0.5


# ── 메인 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    video_path = args.video
    device     = torch.device(args.device)

    if not Path(video_path).exists():
        print(f"오류: 영상 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    sample_id = f"demo_{Path(video_path).stem}"

    # ── Step 1: 특징 추출 ─────────────────────────────────────────────────
    kp_dir = Path(args.keypoint_root) / sample_id
    if args.skip_extract and (kp_dir / "pose.npy").exists():
        print(f"[1/4] 추출 건너뜀 (이미 존재: {kp_dir})")
    else:
        kp_dir = extract_features(video_path, args.keypoint_root, sample_id)

    # ── Step 2: 모델 로딩 ─────────────────────────────────────────────────
    print("[2/4] 모델 로딩 중...")
    from src.data.gloss_vocab import GlossVocab
    cfg        = load_config(args.config)
    tokenizer  = load_tokenizer(cfg.get("tokenizer", {}))
    gloss_vocab= GlossVocab.load(args.gloss_vocab)
    model      = build_model(cfg, tokenizer, gloss_vocab_size=len(gloss_vocab))
    load_checkpoint(model, args.checkpoint, args.device)
    model.to(device).eval()
    print(f"    체크포인트: {args.checkpoint}")

    # ── Step 3: 모델 추론 ─────────────────────────────────────────────────
    print("[3/4] 모델 추론 중...")
    max_seq_len = cfg.get("model", {}).get("landmark", {}).get("max_seq_len", 128)
    batch   = load_keypoints_as_batch(kp_dir, max_seq_len, device)
    t_infer = time.perf_counter()
    outputs, seq_len, _ = run_inference(model, batch, device)
    elapsed_infer = time.perf_counter() - t_infer

    gloss      = decode_gloss(outputs["gloss_logits"], seq_len, gloss_vocab)
    nms        = decode_nms(outputs, seq_len)
    boundary   = decode_boundary(outputs["boundary_logits"])
    confidence = estimate_confidence(outputs)
    draft_text = decode_draft(outputs.get("draft_tokens"), tokenizer)

    # ── Step 4: LLM 보정 ─────────────────────────────────────────────────
    final_text    = None
    uncertain     = []
    retry         = False
    elapsed_llm   = None

    if not args.no_llm:
        print("[4/4] LLM 보정 중...")
        corrector = build_corrector(args.llm)
        gloss_words = [g for g, _ in gloss]
        gloss_confs = [c for _, c in gloss]
        t_llm = time.perf_counter()
        llm_out = corrector.correct(
            korean_draft=draft_text,
            gloss_hypotheses=gloss_words,
            gloss_confidences=gloss_confs,
            nms_summary=nms,
            confidence=confidence,
            domain=args.domain,
        )
        elapsed_llm = time.perf_counter() - t_llm
        final_text  = llm_out.final_text
        uncertain   = llm_out.uncertain_spans
        retry       = llm_out.retry_or_clarify
    else:
        print("[4/4] LLM 보정 생략 (--no_llm)")

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    print_results(
        video_path    = video_path,
        gloss         = gloss,
        nms           = nms,
        boundary      = boundary,
        confidence    = confidence,
        draft_text    = draft_text,
        final_text    = final_text,
        uncertain_spans=uncertain,
        retry_or_clarify=retry,
        elapsed_infer = elapsed_infer,
        elapsed_llm   = elapsed_llm,
    )


if __name__ == "__main__":
    main()
