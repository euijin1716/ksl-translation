#!/usr/bin/env python3
"""raw 영상 삭제 전 GO/NO-GO 검증 게이트 (read-only).

전처리(run_preprocess) 산출물이 완전·정확한지 확인한 뒤에만 raw 영상을 삭제한다.
영상·MediaPipe 불필요 — 매니페스트 + keypoints(+crops)만 읽는다.

가장 단순한 실행:
    python scripts/verify_before_delete.py data/manifests_chunks/chunk_002/all.jsonl

경로·임계값은 configs/base.yaml에서 자동 로드, crops는 자동 감지(있으면 검사).
종료 코드: GO=0, NO-GO=1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# 기대 shape
_SHAPES = {
    "pose": (25, 3),
    "left_hand": (21, 3),
    "right_hand": (21, 3),
    "face_blendshape": (52,),
    "face_key_subset": (68, 3),
}
_EXPLOSION_ABS = 1e4   # face_key 정상 ≈ O(10). 이 이상이면 #2 폭발 의심.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="raw 삭제 전 keypoint 산출물 검증 게이트")
    p.add_argument("manifest", help="검증할 매니페스트 (예: data/manifests_chunks/chunk_002/all.jsonl)")
    p.add_argument("--config", default="configs/base.yaml", help="경로·임계값 출처 (기본 base.yaml)")
    p.add_argument("--keypoint-root", default=None, help="기본: config data.keypoint_root")
    p.add_argument("--crop-root", default=None, help="기본: config data.crop_root")
    p.add_argument("--face-presence-min", type=float, default=0.7, help="전체 face presence 평균 하한 (기본 0.7)")
    p.add_argument("--max-dead-face-ratio", type=float, default=0.2, help="dead-face 허용 비율 상한 (기본 0.2)")
    p.add_argument("--sample", type=int, default=300, help="배열 정밀 점검 표본 수 (기본 300)")
    p.add_argument("--full", action="store_true", help="배열 정밀 점검을 전수로")
    p.add_argument("--require-crops", action="store_true", help="crop 없으면 NO-GO (기본: 없으면 생략)")
    p.add_argument("--raw-root", default=None, help="(선택) 공식 _F 키포인트 XML 존재 점검용 raw 루트")
    return p.parse_args()


def load_config(path: str) -> dict:
    if yaml is None or not Path(path).exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_manifest(path: str) -> list[dict]:
    recs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def view_suffix(video_path: str | None) -> str:
    base = re.split(r"[\\/]", video_path or "")[-1]
    m = re.search(r"_([A-Za-z])\.(mp4|avi|mov)$", base, re.I)
    return m.group(1).upper() if m else "none"


def safe_load_npy(path: Path):
    try:
        return np.load(path)
    except Exception as e:
        return e


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    pre_cfg = cfg.get("preprocess", {}) if isinstance(cfg, dict) else {}

    keypoint_root = Path(args.keypoint_root or data_cfg.get("keypoint_root", "data/keypoints"))
    crop_root = Path(args.crop_root or data_cfg.get("crop_root", "data/crops"))

    if not Path(args.manifest).exists():
        print(f"NO-GO: manifest 없음: {args.manifest}")
        return 1
    recs = load_manifest(args.manifest)
    n = len(recs)
    if n == 0:
        print("NO-GO: 매니페스트에 샘플이 0개")
        return 1

    block: list[str] = []   # NO-GO 사유
    warn: list[str] = []
    info: list[str] = []

    # ── 전수(저비용) 점검: 매니페스트 필드 + pose.npy 존재 ──────────────────────
    missing_pose: list[str] = []
    null_kp: list[str] = []
    extr_err: list[str] = []
    lowface_flag = 0
    view_dist: Counter = Counter()
    unknown_signer = 0
    no_gloss = 0
    has_nms_cnt = 0
    numframes0 = 0
    spans_empty = 0

    for r in recs:
        sid = r.get("sample_id", "?")
        kp_dir = keypoint_root / sid
        if not (kp_dir / "pose.npy").exists():
            missing_pose.append(sid)
        if not r.get("keypoint_path"):
            null_kp.append(sid)
        flags = r.get("quality_flags") or []
        if any(str(f).startswith("extraction_error") for f in flags):
            extr_err.append(sid)
        if any(str(f).startswith("low_face_presence") for f in flags):
            lowface_flag += 1
        view_dist[view_suffix(r.get("video_path"))] += 1
        if str(r.get("signer_id") or "UNKNOWN").upper() == "UNKNOWN":
            unknown_signer += 1
        if not r.get("gloss_tokens"):
            no_gloss += 1
        md = r.get("metadata") or {}
        if r.get("nms_labels") or md.get("has_nms"):
            has_nms_cnt += 1
        if not r.get("num_frames"):
            numframes0 += 1
        if not md.get("annotation_spans"):
            spans_empty += 1

    # ── 표본(np.load) 점검: presence / meta / 배열 ──────────────────────────────
    idxs = list(range(n)) if args.full else list(range(min(args.sample, n)))
    face_rates: list[float] = []
    pose_low: list[str] = []
    hands_dead: list[str] = []
    shape_bad: list[str] = []
    nan_bad: list[str] = []
    explode_bad: list[str] = []
    degenerate: list[str] = []
    align_bad: list[str] = []
    fps_mismatch: list[str] = []
    fps1_bad: list[str] = []
    nms_usable = 0
    sampled = 0
    crop_seen = 0
    crop_bad: list[str] = []

    for i in idxs:
        r = recs[i]
        sid = r.get("sample_id", "?")
        kp_dir = keypoint_root / sid
        if not (kp_dir / "pose.npy").exists():
            continue
        sampled += 1
        # presence
        pm = safe_load_npy(kp_dir / "presence_mask.npy")
        face_rate = None
        if isinstance(pm, np.ndarray) and pm.ndim == 2 and pm.shape[1] >= 4:
            T_p = pm.shape[0]
            face_rate = float(pm[:, 3].mean())
            face_rates.append(face_rate)
            if float(pm[:, 0].mean()) < 0.5:
                pose_low.append(sid)
            if max(float(pm[:, 1].mean()), float(pm[:, 2].mean())) < 0.1:
                hands_dead.append(sid)
        else:
            T_p = None
        # NMS 사용성: 라벨 있고 + 얼굴 입력 살아있음
        md = r.get("metadata") or {}
        if (r.get("nms_labels") or md.get("has_nms")) and (face_rate is not None and face_rate >= 0.1):
            nms_usable += 1
        # meta: fps / 정렬
        meta = {}
        mp = kp_dir / "meta.json"
        if mp.exists():
            try:
                meta = json.load(open(mp, encoding="utf-8"))
            except Exception:
                meta = {}
        ofps, pfps, fsk = meta.get("original_fps"), meta.get("processed_fps"), meta.get("frame_skip")
        if ofps and fsk and pfps and abs(float(pfps) - float(ofps) / float(fsk)) > 1e-3:
            fps1_bad.append(sid)
        if ofps and r.get("fps") and abs(float(ofps) - float(r["fps"])) > 1e-3:
            fps_mismatch.append(sid)
        pfi = meta.get("processed_frame_indices")
        if pfi is not None and T_p is not None and len(pfi) != T_p:
            align_bad.append(sid)
        # 배열 정밀 점검
        Ts = []
        for name, suf in _SHAPES.items():
            arr = safe_load_npy(kp_dir / f"{name}.npy")
            if isinstance(arr, Exception):
                shape_bad.append(f"{sid}:{name}(load err)")
                continue
            if arr.shape[1:] != suf:
                shape_bad.append(f"{sid}:{name}{arr.shape}")
            Ts.append(arr.shape[0])
            if not np.isfinite(arr).all():
                nan_bad.append(f"{sid}:{name}")
            if name == "face_key_subset" and np.abs(arr).max(initial=0.0) > _EXPLOSION_ABS:
                explode_bad.append(f"{sid}(max={np.abs(arr).max():.1e})")
            if name in ("pose", "left_hand", "right_hand") and arr.size and float(arr.std()) < 1e-9:
                degenerate.append(f"{sid}:{name}")
        if Ts and len(set(Ts)) > 1:
            align_bad.append(f"{sid}:T불일치{set(Ts)}")
        if Ts and min(Ts) < 5:
            shape_bad.append(f"{sid}:T<5({min(Ts)})")
        # crop (자동 감지)
        ci = crop_root / sid / "crop_index.json"
        if ci.exists():
            crop_seen += 1
            try:
                frames = json.load(open(ci, encoding="utf-8")).get("frames", [])
                if not frames:
                    crop_bad.append(f"{sid}(빈 crop_index)")
            except Exception:
                crop_bad.append(f"{sid}(crop_index load err)")

    face_mean = float(np.mean(face_rates)) if face_rates else 0.0
    face_med = float(np.median(face_rates)) if face_rates else 0.0
    dead_face_ratio = float(np.mean([r < 0.1 for r in face_rates])) if face_rates else 1.0

    # ── 판정 ────────────────────────────────────────────────────────────────────
    if missing_pose:
        block.append(f"pose.npy 누락 {len(missing_pose)}개 (예: {missing_pose[:5]})")
    if null_kp:
        block.append(f"keypoint_path null {len(null_kp)}개")
    if extr_err:
        block.append(f"extraction_error 플래그 {len(extr_err)}개 (예: {extr_err[:5]})")
    if face_rates and face_mean < args.face_presence_min:
        block.append(f"face presence 평균 {face_mean:.3f} < {args.face_presence_min} (얼굴 복구 실패)")
    if face_rates and dead_face_ratio > args.max_dead_face_ratio:
        block.append(f"dead-face 비율 {dead_face_ratio:.1%} > {args.max_dead_face_ratio:.0%}")
    if shape_bad:
        block.append(f"shape/T 이상 {len(shape_bad)}건 (예: {shape_bad[:5]})")
    if nan_bad:
        block.append(f"NaN/Inf {len(nan_bad)}건 (예: {nan_bad[:5]})")
    if explode_bad:
        block.append(f"face_key 폭발 의심 {len(explode_bad)}건 (예: {explode_bad[:5]})")
    if align_bad:
        block.append(f"프레임 정렬 불일치 {len(align_bad)}건 (예: {align_bad[:5]})")
    if args.require_crops and crop_seen < sampled:
        block.append(f"crop 누락: 표본 {sampled}중 {crop_seen}만 crop_index 보유")
    if crop_bad:
        block.append(f"crop 손상 {len(crop_bad)}건 (예: {crop_bad[:5]})")

    if pose_low:
        warn.append(f"pose presence 낮은 샘플 {len(pose_low)}개")
    if hands_dead:
        warn.append(f"양손 모두 dead {len(hands_dead)}개")
    if degenerate:
        warn.append(f"상수(degenerate) 스트림 {len(degenerate)}건 (예: {degenerate[:5]})")
    if fps_mismatch:
        warn.append(f"manifest.fps ≠ meta.original_fps {len(fps_mismatch)}건")
    if fps1_bad:
        warn.append(f"processed_fps ≠ original/frame_skip {len(fps1_bad)}건 (#1 미적용?)")
    if unknown_signer:
        warn.append(f"signer_id=UNKNOWN {unknown_signer}개 (split 영향)")
    if no_gloss:
        warn.append(f"gloss 없음 {no_gloss}개")
    nonF = sum(v for k, v in view_dist.items() if k not in ("F", "none"))
    if nonF:
        warn.append(f"비-_F 뷰 {nonF}개 (face 결손과 상관) — view dist {dict(view_dist)}")

    info.append(f"NMS 사용성(라벨+얼굴 살아있음): 표본 {sampled}중 {nms_usable} ({(nms_usable/sampled*100) if sampled else 0:.0f}%)")
    info.append(f"low_face_presence 플래그(매니페스트 전수): {lowface_flag}개")
    info.append(f"manifest num_frames==0: {numframes0}/{n} (실 프레임수는 keypoint meta.json)")
    info.append(f"annotation_spans 빈값: {spans_empty}/{n} (boundary 타이밍 미추출 — JSON 보존 시 복구 가능)")
    if crop_seen == 0 and not args.require_crops:
        info.append("crops 없음 → 점검 생략(의도된 --skip_crops로 간주)")

    # 공식 _F XML 존재 (opt-in, bounded)
    if args.raw_root:
        try:
            cnt = 0
            for _ in Path(args.raw_root).rglob("*_F.xml"):
                cnt += 1
                if cnt >= 50:
                    break
            info.append(f"공식 _F 키포인트 XML: {'>=50개 존재' if cnt >= 50 else f'{cnt}개'} (보존 검토)")
        except Exception as e:
            info.append(f"_F XML 점검 실패: {e}")

    # ── 리포트 ──────────────────────────────────────────────────────────────────
    print("=" * 64)
    print("verify_before_delete — raw 영상 삭제 전 게이트")
    print("=" * 64)
    print(f"manifest        : {args.manifest}")
    print(f"keypoint_root   : {keypoint_root}")
    print(f"samples         : {n}  | 배열정밀점검 표본: {sampled}{' (전수)' if args.full else ''}")
    print(f"view dist       : {dict(view_dist)}")
    print(f"face presence   : 평균 {face_mean:.3f} / 중앙 {face_med:.3f} / dead-face {dead_face_ratio:.1%}")
    print("-" * 64)
    if block:
        print("🔴 BLOCK:")
        for b in block:
            print("   - " + b)
    if warn:
        print("🟡 WARN:")
        for w in warn:
            print("   - " + w)
    if info:
        print("ℹ️  INFO:")
        for x in info:
            print("   - " + x)
    print("=" * 64)
    if block:
        print("결과: ❌ NO-GO — 위 BLOCK 해결 전 raw 영상 삭제 금지.")
        return 1
    print("결과: ✅ GO — keypoint 산출물 검증 통과. (WARN 확인 후) raw 영상 삭제 가능.")
    print("      ※ B: 라벨 JSON(라벨링데이터)은 삭제하지 말 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
