#!/usr/bin/env python3
"""상세 추론 추적 스크립트.

test 셋에서 N개 샘플을 선별해 파이프라인 각 단계를 자세하게 로그로 출력한다.

  MediaPipe 키포인트 통계
  → CTC 프레임별 gloss 타임라인
  → NMS 신호 막대
  → Intent 클래스 확률
  → Boundary/Activity 분포
  → 번역 예측 vs 정답 비교
  → 샘플별 / 전체 집계 지표

사용 예:
  python scripts/run_eval_trace.py \\
    --checkpoint checkpoints/C/best.pt \\
    --config configs/stage_c.yaml \\
    --manifest data/manifests/test.jsonl \\
    --gloss_vocab data/manifests/gloss_vocab.json \\
    --num_samples 20 \\
    --seed 42 \\
    --log_file trace_logs/eval_trace.log
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _model_runtime import load_checkpoint

# ── Rich (선택) ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    _RICH = True
except ImportError:
    _RICH = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_W = 78  # 출력 너비

_DOMAIN_LIST = ["hospital", "directions", "order", "reservation", "public", "help", "unknown"]
_DOMAIN_KO = {
    "hospital": "병원",
    "directions": "길안내",
    "order": "주문/결제",
    "reservation": "예약/확인",
    "public": "공공민원",
    "help": "도움요청",
    "unknown": "알수없음",
}

_PRESENCE_NAMES = ["pose", "left_hand", "right_hand", "face"]

# ── 포매터 헬퍼 ──────────────────────────────────────────────────────────────

def _bar(prob: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    n = round(prob * width)
    return fill * n + empty * (width - n)


def _sep(char: str = "═") -> str:
    return char * _W


def _box_line(text: str) -> str:
    inner = f" {text}"
    return f"│{inner:<{_W - 2}}│"


def _inline_diff(hyp: str, ref: str) -> str:
    """문자 단위 diff를 한 줄로 표현 (+추가/-삭제/공백=일치)."""
    ops = list(difflib.SequenceMatcher(None, ref, hyp).get_opcodes())
    out_hyp, out_ref = [], []
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            out_hyp.append(ref[i1:i2])
            out_ref.append(ref[i1:i2])
        elif op == "replace":
            out_hyp.append(f"[+{hyp[j1:j2]}]")
            out_ref.append(f"[-{ref[i1:i2]}]")
        elif op == "insert":
            out_hyp.append(f"[+{hyp[j1:j2]}]")
        elif op == "delete":
            out_ref.append(f"[-{ref[i1:i2]}]")
    return "hyp: " + "".join(out_hyp) + "\n  ref: " + "".join(out_ref)


def _wer_single(hyp: list[str], ref: list[str]) -> float:
    """레벤슈타인 거리 기반 단어 오류율 (단일 샘플)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m] / len(ref)


def _bleu1_single(hyp: str, ref: str) -> float:
    """간이 unigram BLEU (sacrebleu 없을 때 fallback, 참고용)."""
    if not hyp or not ref:
        return 0.0
    hyp_chars = list(hyp.replace(" ", ""))
    ref_chars = list(ref.replace(" ", ""))
    ref_counter = Counter(ref_chars)
    matches = sum(min(cnt, ref_counter.get(c, 0)) for c, cnt in Counter(hyp_chars).items())
    brevity = min(1.0, math.exp(1 - len(ref_chars) / max(len(hyp_chars), 1)))
    return brevity * (matches / max(len(hyp_chars), 1))


def _compute_sample_metrics(hyp: str, ref: str) -> dict[str, float]:
    try:
        from sacrebleu.metrics import BLEU, CHRF
        bleu = BLEU(effective_order=True).corpus_score([hyp], [[ref]]).score
        chrf = CHRF().corpus_score([hyp], [[ref]]).score
    except Exception:
        bleu = _bleu1_single(hyp, ref) * 100.0
        chrf = 0.0
    return {"bleu": round(bleu, 2), "chrf": round(chrf, 2)}


# ── 출력 블록 함수 ────────────────────────────────────────────────────────────

def _print_header(idx: int, total: int, sample_id: str, domain: str, split: str) -> None:
    print()
    print(_sep("═"))
    tag = f"SAMPLE {idx}/{total}  │  id: {sample_id}  │  domain: {domain} ({_DOMAIN_KO.get(domain, '')})"
    print(f"  {tag}")
    print(f"  split: {split}")
    print(_sep("═"))


def _print_metadata(batch: dict[str, Any]) -> None:
    print("\n[METADATA]")
    sid = batch.get("sample_id", ["?"])
    sid = sid[0] if isinstance(sid, (list, tuple)) else sid
    domain = batch.get("domain", ["?"])
    domain = domain[0] if isinstance(domain, (list, tuple)) else domain
    korean = batch.get("korean_text", [""])
    korean = korean[0] if isinstance(korean, (list, tuple)) else korean
    gloss_raw = batch.get("gloss_tokens_raw", [[]])
    gloss_raw = gloss_raw[0] if isinstance(gloss_raw, (list, tuple)) and gloss_raw else []
    seq_len = int(batch["seq_len"][0].item()) if isinstance(batch["seq_len"], torch.Tensor) else int(batch["seq_len"][0])
    print(f"  sample_id : {sid}")
    print(f"  domain    : {domain}")
    print(f"  seq_len   : {seq_len} frames")
    print(f"  GT gloss  : {gloss_raw if gloss_raw else '(없음)'}")
    print(f"  GT korean : {korean if korean else '(없음)'}")


def _print_keypoint_stats(batch: dict[str, Any]) -> None:
    print("\n[MEDIAPIPE KEYPOINTS]")
    presence = batch.get("presence_mask")
    seq_len = int(batch["seq_len"][0].item())
    header = f"  {'modality':<14} {'detected':>10}  {'bar':<22} {'shape'}"
    print(header)
    print("  " + "-" * (_W - 4))

    # presence_mask shape: (B, T, 4)
    if presence is not None and isinstance(presence, torch.Tensor):
        pm = presence[0, :seq_len].bool()  # (T, 4)
        for ch, name in enumerate(_PRESENCE_NAMES):
            if ch < pm.shape[-1]:
                detected = int(pm[:, ch].sum().item())
            else:
                detected = seq_len
            pct = detected / max(seq_len, 1)
            bar = _bar(pct, width=22)
            print(f"  {name:<14} {detected:>4}/{seq_len:<4}  {bar}  {pct * 100:.1f}%")
    else:
        print("  (presence_mask 없음 — 키포인트 파일에서 로드되지 않음)")

    # 텐서 shape 출력
    print()
    shapes = {
        "pose": batch.get("pose"),
        "left_hand": batch.get("left_hand"),
        "right_hand": batch.get("right_hand"),
        "face_blendshape": batch.get("face_blendshape"),
        "face_key_subset": batch.get("face_key_subset"),
    }
    for name, t in shapes.items():
        if t is not None and isinstance(t, torch.Tensor):
            print(f"  {name:<20} shape={tuple(t[0].shape)}")


def _print_ctc_timeline(
    gloss_logits: torch.Tensor | None,
    seq_len: int,
    gloss_vocab: Any,
    n_buckets: int = 12,
) -> None:
    print("\n[CTC GLOSS TIMELINE]")
    if gloss_logits is None:
        print("  (gloss_logits 없음)")
        return

    T = seq_len
    probs = gloss_logits[0, :T].softmax(dim=-1)   # (T, V)
    argmax_ids = probs.argmax(dim=-1).tolist()      # (T,)
    argmax_probs = probs.max(dim=-1).values.tolist()  # (T,)

    # 프레임을 버킷으로 나눔
    bucket_size = max(1, T // n_buckets)
    print(f"  T={T} frames, bucket_size≈{bucket_size}")
    print(f"  {'frames':<12}  {'top token':<16}  {'conf':>5}  bar")
    print("  " + "-" * (_W - 4))

    for b in range(0, T, bucket_size):
        end = min(b + bucket_size, T)
        bucket_ids = argmax_ids[b:end]
        bucket_probs = argmax_probs[b:end]

        # 버킷 내 가장 빈번한 non-blank 토큰
        counter = Counter(i for i in bucket_ids if i != 0)
        if counter:
            top_id = counter.most_common(1)[0][0]
            # 해당 토큰이 예측된 프레임의 평균 확률
            avg_conf = sum(p for i, p in zip(bucket_ids, bucket_probs) if i == top_id) / counter[top_id]
            label_list = gloss_vocab.decode([top_id]) if gloss_vocab else [str(top_id)]
            label = label_list[0] if label_list else str(top_id)
        else:
            label = "[blank]"
            avg_conf = sum(bucket_probs) / max(len(bucket_probs), 1)

        bar = _bar(avg_conf, width=20)
        frame_range = f"f{b + 1:04d}-{end:04d}"
        print(f"  {frame_range:<12}  {label:<16}  {avg_conf:>5.3f}  {bar}")

    # CTC collapse
    prev, emitted = -1, []
    for t, tok in enumerate(argmax_ids):
        if tok != prev and tok != 0:
            emitted.append((tok, float(probs[t, tok])))
        prev = tok
    emitted = emitted[:10]

    if gloss_vocab:
        words = gloss_vocab.decode([tok for tok, _ in emitted])
        pairs = [(w, round(c, 3)) for w, (_, c) in zip(words, emitted)]
    else:
        pairs = [(str(tok), round(c, 3)) for tok, c in emitted]

    print()
    print(f"  CTC collapsed ({len(pairs)} tokens):")
    for w, c in pairs:
        print(f"    {w:<18}  conf={c:.3f}  {_bar(c, width=20)}")


def _print_gloss_comparison(
    hyp_pairs: list[tuple[str, float]],
    ref_glosses: list[str],
) -> None:
    print("\n[GLOSS 비교]")
    hyp_words = [w for w, _ in hyp_pairs]
    wer = _wer_single(hyp_words, ref_glosses)
    match = "✓ EXACT MATCH" if wer == 0.0 else f"WER={wer:.3f}"
    print(f"  Hyp : {hyp_words}")
    print(f"  Ref : {ref_glosses}")
    print(f"  → {match}")


def _print_nms_signals(outputs: dict, seq_len: int) -> None:
    from src.data.signals import NMS_KEYS, NMS_DETAIL_CLASSES, NMS_DETAIL_GROUPS

    print("\n[NMS 신호]  (active 프레임 평균)")
    nms_logits = outputs.get("nms_logits")
    if nms_logits is None:
        print("  (nms_logits 없음)")
    else:
        probs = torch.sigmoid(nms_logits[0, :seq_len]).mean(dim=0)   # (nms_classes,)
        print(f"  {'signal':<20}  {'prob':>5}  bar")
        print("  " + "-" * 50)
        for i, key in enumerate(NMS_KEYS):
            if i >= len(probs):
                break
            p = float(probs[i].item())
            flag = "◀ HIGH" if p >= 0.5 else ""
            print(f"  {key:<20}  {p:>5.3f}  {_bar(p, width=20)} {flag}")

    print()
    for group, classes in NMS_DETAIL_CLASSES.items():
        key = f"nms_{group}_logits"
        logits = outputs.get(key)
        if logits is None:
            continue
        p = torch.softmax(logits[0, :seq_len], dim=-1).mean(dim=0)  # (n_classes,)
        cls_idx = int(p.argmax().item())
        label = classes[cls_idx]
        conf = float(p[cls_idx].item())
        all_str = "  ".join(f"{c}:{float(p[j]):.2f}" for j, c in enumerate(classes))
        print(f"  [{group}]  pred={label}({conf:.3f})   all=[ {all_str} ]")


def _print_intent(outputs: dict, gt_intent: int | None) -> None:
    print("\n[INTENT 예측]")
    logits = outputs.get("intent_logits")
    if logits is None:
        print("  (intent_logits 없음)")
        return
    probs = torch.softmax(logits[0], dim=-1)
    pred_idx = int(probs.argmax().item())
    print(f"  {'domain':<14}  {'prob':>5}  bar")
    print("  " + "-" * 50)
    for i, domain in enumerate(_DOMAIN_LIST):
        p = float(probs[i].item())
        marker = ""
        if i == pred_idx:
            marker += "← PRED"
        if gt_intent is not None and i == gt_intent:
            marker += " (GT)"
            if i == pred_idx:
                marker += " ✓"
            else:
                marker += " ✗"
        print(f"  {domain:<14}  {p:>5.3f}  {_bar(p, width=18)} {marker}")


def _print_boundary(outputs: dict, batch: dict, seq_len: int) -> None:
    print("\n[BOUNDARY / ACTIVITY]")
    logits = outputs.get("boundary_logits")
    if logits is None:
        print("  (boundary_logits 없음)")
        return
    preds = logits[0, :seq_len].argmax(dim=-1).tolist()
    counter = Counter(preds)
    labels = {0: "idle", 1: "signing", 2: "boundary"}
    print(f"  {'class':<14}  {'frames':>6}  {'pct':>5}  bar")
    print("  " + "-" * 50)
    for cls, name in labels.items():
        n = counter.get(cls, 0)
        pct = n / max(seq_len, 1)
        print(f"  {name}({cls})      {n:>6}  {pct * 100:>4.1f}%  {_bar(pct, width=18)}")

    # ground truth activity
    if "activity" in batch:
        gt = batch["activity"][0, :seq_len].tolist()
        gt_counter = Counter(gt)
        print(f"\n  GT activity: {dict(gt_counter)}")
        try:
            from src.eval.metrics import compute_f1
            f1_dict = compute_f1(preds, gt, num_classes=3)
            print(f"  Boundary F1 (this sample, macro): {f1_dict['f1']:.3f}")
        except Exception:
            pass

    last_state = {0: "idle", 1: "ongoing", 2: "ended"}.get(preds[-1] if preds else 0, "ended")
    print(f"  final state: {last_state}")


def _print_translation(
    draft: str,
    ref: str,
    seq_len: int,
) -> None:
    print("\n[번역 비교]")
    print(f"  DRAFT     : {draft or '(없음)'}")
    print(f"  REFERENCE : {ref or '(없음)'}")

    if draft and ref:
        if draft.strip() == ref.strip():
            print("              ↑ EXACT MATCH ✓")
        else:
            print()
            diff = _inline_diff(draft, ref)
            print(f"  diff  : {diff}")
        metrics = _compute_sample_metrics(draft, ref)
        print()
        print(f"  BLEU (this sample): {metrics['bleu']:.2f}")
        print(f"  chrF (this sample): {metrics['chrf']:.2f}")
    return


def _print_sample_summary(idx: int, hyp: str, ref: str, gloss_wer: float | None) -> dict:
    metrics = _compute_sample_metrics(hyp, ref)
    print("\n" + "─" * _W)
    print(f"  Sample {idx} summary → BLEU={metrics['bleu']:.2f}  chrF={metrics['chrf']:.2f}"
          + (f"  gloss_WER={gloss_wer:.3f}" if gloss_wer is not None else ""))
    print("─" * _W)
    return metrics


def _print_aggregate(results: list[dict], eval_result: Any | None = None) -> None:
    print()
    print(_sep("═"))
    print(f"  AGGREGATE RESULTS  ({len(results)} samples)")
    print(_sep("═"))

    bleus = [r["bleu"] for r in results if "bleu" in r]
    chrfs = [r["chrf"] for r in results if "chrf" in r]
    wers  = [r["gloss_wer"] for r in results if r.get("gloss_wer") is not None]

    def _avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    print(f"  BLEU        : {_avg(bleus):.2f}")
    print(f"  chrF        : {_avg(chrfs):.2f}")
    if wers:
        print(f"  Gloss WER   : {_avg(wers):.3f}")

    if eval_result is not None:
        print(f"  Boundary F1 : {eval_result.boundary_f1:.3f}")
        print(f"  NMS F1      : {eval_result.nms_f1:.3f}")
        print(f"  Intent Acc  : {eval_result.intent_accuracy:.3f}")
        print(f"  ROUGE-L     : {eval_result.rouge_l:.3f}")
        print(f"  TTFT avg    : {eval_result.ttft_ms:.0f} ms")

    print()
    # Best/worst by BLEU
    indexed = sorted(enumerate(results), key=lambda x: x[1].get("bleu", 0))
    worst3 = indexed[:3]
    best3  = indexed[-3:][::-1]
    print("  [BEST 3]")
    for i, r in best3:
        print(f"    [{i + 1:02d}] BLEU={r.get('bleu', 0):.2f}  hyp='{r.get('hyp', '')[:50]}'")
    print("  [WORST 3]")
    for i, r in worst3:
        print(f"    [{i + 1:02d}] BLEU={r.get('bleu', 0):.2f}  ref='{r.get('ref', '')[:50]}'")

    print()
    print("  [DOMAIN DISTRIBUTION]")
    domains = [r.get("domain", "?") for r in results]
    for d, cnt in Counter(domains).most_common():
        sub = [r for r in results if r.get("domain") == d]
        avg_b = _avg([r["bleu"] for r in sub if "bleu" in r])
        print(f"    {d:<14} n={cnt}  avg_BLEU={avg_b:.2f}")

    print(_sep("═"))


# ── 모델 로딩 (run_eval.py 와 동일) ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="파이프라인 상세 추적 — 샘플별 키포인트→gloss→번역 전 과정 출력"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--manifest", default="data/manifests/test.jsonl")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gloss_vocab", default="data/manifests/gloss_vocab.json")
    p.add_argument("--num_samples", type=int, default=20, help="추적할 샘플 수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_ids", nargs="*", help="특정 sample_id 지정 (없으면 무작위)")
    p.add_argument("--log_file", default=None, help="로그를 파일에도 저장")
    p.add_argument(
        "--draft_mode",
        default="greedy",
        choices=["teacher", "greedy"],
        help="greedy=실제 추론, teacher=교사강제(빠름)",
    )
    p.add_argument(
        "--llm",
        default="none",
        choices=["none", "claude", "openai", "dummy"],
    )
    return p.parse_args()


def _dataclass_from_dict(cls, cfg: dict | None, **overrides):
    cfg = dict(cfg or {})
    cfg.update(overrides)
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg.items() if k in allowed})


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 로그 파일 설정 ──────────────────────────────────────────────────────
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        # stdout 출력도 파일로 미러링하기 위해 별도 트리거
        logger.addHandler(file_handler)
        print(f"# 로그 파일: {log_path}", flush=True)

    # ── 설정 로드 ───────────────────────────────────────────────────────────
    base_cfg: dict = {}
    if Path("configs/base.yaml").exists():
        base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8")) or {}
    stage_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    from _model_runtime import merge_cfg as _merge_cfg
    cfg = _merge_cfg(base_cfg, stage_cfg)

    # ── import ──────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from src.data.dummy import collate_fn
    from src.data.gloss_vocab import GlossVocab
    from src.data.keypoint_dataset import KeypointDataset
    from src.eval.evaluator import KSLEvaluator
    from src.models.decoder import DecoderConfig
    from src.models.fusion import FusionConfig
    from src.models.heads import HeadsConfig
    from src.models.ksl_model import KSLModel, ModelConfig
    from src.models.streams.face_expr_encoder import FaceExprEncoderConfig
    from src.models.streams.hand_visual_encoder import HandVisualEncoderConfig
    from src.models.streams.landmark_encoder import LandmarkEncoderConfig
    from transformers import AutoTokenizer

    tok_name = cfg.get("tokenizer", {}).get("name", "klue/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    gloss_vocab = GlossVocab.load(args.gloss_vocab)

    model_cfg = cfg.get("model", {})
    landmark_cfg = model_cfg.get("landmark", {})
    tokenizer_cfg = cfg.get("tokenizer", {})

    decoder_cfg = DecoderConfig.from_tokenizer(
        tokenizer,
        d_model=model_cfg.get("decoder", {}).get("d_model", 256),
        nhead=model_cfg.get("decoder", {}).get("nhead", 4),
        num_layers=model_cfg.get("decoder", {}).get("num_layers", 4),
        dim_feedforward=model_cfg.get("decoder", {}).get("dim_feedforward", 512),
        dropout=model_cfg.get("decoder", {}).get("dropout", 0.1),
        max_len=model_cfg.get("decoder", {}).get("max_len", 128),
    )
    heads_cfg = _dataclass_from_dict(
        HeadsConfig, model_cfg.get("heads", {}), gloss_vocab_size=len(gloss_vocab)
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

    ckpt = load_checkpoint(model, Path(args.checkpoint), args.device)
    logger.info(
        "Checkpoint loaded: step=%s  best_val_loss=%s",
        ckpt.get("global_step"),
        ckpt.get("best_val_loss"),
    )

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
    logger.info("Dataset size: %d", len(dataset))

    # ── 샘플 선별 ──────────────────────────────────────────────────────────
    n = min(args.num_samples, len(dataset))
    if args.sample_ids:
        id_set = set(args.sample_ids)
        indices = [i for i, s in enumerate(dataset.samples) if s.sample_id in id_set]
        if not indices:
            logger.error("지정한 sample_ids가 데이터셋에 없습니다: %s", args.sample_ids)
            sys.exit(1)
    else:
        indices = random.sample(range(len(dataset)), n)
    indices = sorted(indices)
    logger.info("추적할 샘플 %d개: indices=%s", len(indices), indices[:10])

    # ── LLM (선택) ──────────────────────────────────────────────────────────
    corrector = None
    if args.llm != "none":
        from src.llm.factory import build_corrector
        llm_cfg = dict(cfg.get("llm", {}))
        llm_cfg["provider"] = args.llm
        corrector = build_corrector(llm_cfg)

    model.eval()
    device = torch.device(args.device)
    model.to(device)

    # ── KSLEvaluator (전체 집계용) ──────────────────────────────────────────
    evaluator = KSLEvaluator(model, device=args.device, corrector=corrector)
    # 선별된 샘플만으로 서브셋 DataLoader 구성
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    # ── 샘플별 추적 루프 ───────────────────────────────────────────────────
    sample_results: list[dict] = []
    use_teacher = args.draft_mode == "teacher"

    print()
    print(_sep("═"))
    print(f"  KSL 추론 추적  |  {len(indices)}개 샘플  |  모델: {Path(args.checkpoint).name}")
    print(f"  config: {args.config}  |  draft_mode: {args.draft_mode}  |  device: {args.device}")
    print(_sep("═"))

    with torch.no_grad():
        for sample_num, (idx, batch) in enumerate(zip(indices, loader), start=1):
            sid = batch.get("sample_id", ["?"])
            sid = sid[0] if isinstance(sid, (list, tuple)) else sid
            domain = batch.get("domain", ["?"])
            domain = domain[0] if isinstance(domain, (list, tuple)) else domain
            split_grp = batch.get("split_group", ["?"])
            split_grp = split_grp[0] if isinstance(split_grp, (list, tuple)) else split_grp

            _print_header(sample_num, len(indices), sid, domain, split_grp)
            _print_metadata(batch)

            # 배치를 device로
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            seq_len = int(batch_d["seq_len"][0].item())
            T = batch_d["pose"].shape[1]
            mask = torch.arange(T, device=device).unsqueeze(0) >= batch_d["seq_len"].unsqueeze(1)

            t_start = time.perf_counter()
            outputs = model(
                pose=batch_d["pose"],
                left_hand=batch_d["left_hand"],
                right_hand=batch_d["right_hand"],
                face_blendshape=batch_d["face_blendshape"],
                face_key_subset=batch_d.get("face_key_subset"),
                presence_mask=batch_d.get("presence_mask"),
                tgt_tokens=batch_d.get("tgt_tokens") if use_teacher else None,
                tgt_padding=batch_d.get("tgt_padding") if use_teacher else None,
                src_key_padding_mask=mask,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            # ─ 키포인트 통계 (MediaPipe 출력)
            _print_keypoint_stats(batch_d)

            # ─ CTC 타임라인
            _print_ctc_timeline(outputs.get("gloss_logits"), seq_len, gloss_vocab)

            # ─ Gloss 비교
            from src.eval.evaluator import _ctc_collapse
            gloss_logits = outputs.get("gloss_logits")
            if gloss_logits is not None:
                pred_ids = gloss_logits[0, :seq_len].argmax(dim=-1)
                hyp_gloss_ids = _ctc_collapse(pred_ids.tolist())
                hyp_gloss = gloss_vocab.decode(hyp_gloss_ids)
            else:
                hyp_gloss = []
            ref_gloss = batch.get("gloss_tokens_raw", [[]])[0] if isinstance(
                batch.get("gloss_tokens_raw"), (list, tuple)) else []
            # hyp_pairs for display
            if gloss_logits is not None:
                probs_g = gloss_logits[0, :seq_len].softmax(dim=-1)
                ids_g = probs_g.argmax(dim=-1).tolist()
                prev_g, emitted_g = -1, []
                for t_g, tok_g in enumerate(ids_g):
                    if tok_g != prev_g and tok_g != 0:
                        emitted_g.append((tok_g, float(probs_g[t_g, tok_g])))
                    prev_g = tok_g
                emitted_g = emitted_g[:10]
                w_g = gloss_vocab.decode([tok for tok, _ in emitted_g])
                hyp_pairs = [(w, round(c, 3)) for w, (_, c) in zip(w_g, emitted_g)]
            else:
                hyp_pairs = []
            _print_gloss_comparison(hyp_pairs, ref_gloss)

            # ─ NMS 신호
            _print_nms_signals(outputs, seq_len)

            # ─ Intent
            gt_intent_idx = int(batch_d["intent_label"][0].item()) if "intent_label" in batch_d else None
            _print_intent(outputs, gt_intent_idx)

            # ─ Boundary
            _print_boundary(outputs, batch_d, seq_len)

            # ─ 번역
            korean_ref = batch.get("korean_text", [""])
            korean_ref = korean_ref[0] if isinstance(korean_ref, (list, tuple)) else korean_ref

            draft_text = ""
            if "draft_logits" in outputs and tokenizer is not None:
                ids_d = outputs["draft_logits"][0].argmax(dim=-1).tolist()
                draft_text = tokenizer.decode(ids_d, skip_special_tokens=True).strip()
            elif "draft_tokens" in outputs and tokenizer is not None:
                draft_text = tokenizer.decode(
                    outputs["draft_tokens"][0].tolist(), skip_special_tokens=True
                ).strip()

            llm_text = ""
            if corrector is not None and draft_text:
                nms_sum = {}
                nms_logits_s = outputs.get("nms_logits")
                if nms_logits_s is not None:
                    from src.data.signals import NMS_KEYS
                    probs_n = torch.sigmoid(nms_logits_s[0, :seq_len]).mean(dim=0)
                    nms_sum = {k: round(float(probs_n[i]), 3) for i, k in enumerate(NMS_KEYS) if i < len(probs_n)}
                conf_llm = 0.5
                if "intent_logits" in outputs:
                    conf_llm = float(torch.softmax(outputs["intent_logits"][0], dim=-1).max().item())
                llm_out = corrector.correct(
                    korean_draft=draft_text,
                    gloss_hypotheses=[w for w, _ in hyp_pairs],
                    gloss_confidences=[c for _, c in hyp_pairs],
                    nms_summary=nms_sum,
                    confidence=conf_llm,
                    domain=domain,
                )
                llm_text = llm_out.final_text
                print(f"\n  LLM 보정: {llm_text}")

            _print_translation(draft_text, korean_ref, seq_len)

            print(f"\n  추론 시간: {elapsed_ms:.1f} ms  |  T={seq_len} frames")

            # ─ 샘플 결과 저장
            gloss_wer = _wer_single(hyp_gloss, ref_gloss) if ref_gloss else None
            s_metrics = _compute_sample_metrics(draft_text, korean_ref)
            sample_results.append({
                "idx": idx,
                "sample_id": sid,
                "domain": domain,
                "hyp": draft_text,
                "ref": korean_ref,
                "llm": llm_text,
                "gloss_wer": gloss_wer,
                "bleu": s_metrics["bleu"],
                "chrf": s_metrics["chrf"],
                "elapsed_ms": round(elapsed_ms, 1),
            })
            _print_sample_summary(sample_num, draft_text, korean_ref, gloss_wer)

    # ── 전체 집계 ─────────────────────────────────────────────────────────
    # evaluator로 전체 집계 지표 계산 (이미 선별 샘플 기준)
    eval_result = evaluator.evaluate(
        loader,
        split=args.split,
        tokenizer=tokenizer,
        gloss_vocab=gloss_vocab,
        draft_mode=args.draft_mode,
    )
    _print_aggregate(sample_results, eval_result)

    # ── JSON 저장 ─────────────────────────────────────────────────────────
    out_dir = Path("trace_logs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"trace_{args.split}_{args.num_samples}samples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": str(args.checkpoint),
                "num_samples": len(indices),
                "draft_mode": args.draft_mode,
                "samples": sample_results,
                "aggregate": eval_result.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  JSON 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
