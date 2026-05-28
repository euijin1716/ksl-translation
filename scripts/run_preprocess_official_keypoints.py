# ============================================================================
# [미채택 / 고려 흔적] AIHub 공식 keypoint(JSON landmarks) 패킹 전처리
# ----------------------------------------------------------------------------
# 디스크 제약 배치용 "MediaPipe 없이 공식 landmark를 패킹" 대체 경로로 한때 고려.
# → 채택하지 않음.
# 최종 결정(2026-05-28): MediaPipe Holistic 사용으로 확정.
#   이유: 이 경로는 face_blendshape=0 → E3(표정/비수지) 신호가 죽음.
#         프로젝트는 비수지 신호가 필수라 Holistic의 실제 blendshape가 필요.
# 아래는 당시 구현 전체를 참고용으로 주석 보존 (실행되지 않음).
# ============================================================================

# #!/usr/bin/env python3
# """Pack AIHub official landmark JSON into the project's keypoint format.
# 
# This is intended for disk-constrained batch processing:
# build a manifest for the currently downloaded raw chunk, pack the official
# JSON landmarks into .npy files, verify, then delete only that raw chunk.
# """
# 
# from __future__ import annotations
# 
# import argparse
# import json
# import logging
# import sys
# from pathlib import Path
# from typing import Any
# 
# import numpy as np
# 
# sys.path.insert(0, str(Path(__file__).parent.parent))
# 
# from src.data.manifest import read_manifest, write_manifest
# from src.data.schema import KSLSample
# from src.preprocess.extractors.base_extractor import ExtractionResult
# from src.preprocess.normalizers.coordinate_normalizer import (
#     apply_shoulder_transform,
#     normalize_landmarks,
#     shoulder_transform_params,
# )
# from src.preprocess.packers.landmark_packer import LandmarkPacker
# from src.preprocess.validators.shape_validator import validate_extraction_result
# 
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)
# 
# 
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Pack official AIHub JSON landmarks into npy keypoints.")
#     p.add_argument("--manifest", required=True, help="Chunk manifest JSONL to update in place.")
#     p.add_argument("--raw_root", default="data/raw", help="Root that source_annotation_path is relative to.")
#     p.add_argument("--output_root", default="data/keypoints_official", help="Where npy keypoints are written.")
#     p.add_argument("--confidence_threshold", type=float, default=0.05)
#     p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
#     p.add_argument("--progress_every", type=int, default=100)
#     return p.parse_args()
# 
# 
# def _reshape(frames: Any, joints: int, name: str) -> np.ndarray:
#     arr = np.asarray(frames or [], dtype=np.float32)
#     if arr.ndim != 2 or arr.shape[1] != joints * 3:
#         raise ValueError(f"{name} expected [T,{joints * 3}], got {arr.shape}")
#     return arr.reshape(arr.shape[0], joints, 3)
# 
# 
# def _landmark_field(landmarks: dict[str, Any], base: str) -> Any:
#     return landmarks.get(f"{base}_2d") or landmarks.get(f"{base}_3d")
# 
# 
# def _presence_and_quality(arr: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
#     conf = arr[:, :, 2]
#     present = (conf > threshold).any(axis=1)
#     quality = np.clip(conf.mean(axis=1), 0.0, 1.0).astype(np.float32)
#     return present, quality
# 
# 
# def _annotation_spans(raw: dict[str, Any], fps: float) -> list[dict[str, Any]]:
#     sign_script = raw.get("sign_script")
#     if not isinstance(sign_script, dict):
#         return []
# 
#     spans: list[dict[str, Any]] = []
#     for key in (
#         "sign_gestures_both",
#         "sign_gestures_strong",
#         "sign_gestures_weak",
#         "sign_gestures_right",
#         "sign_gestures_left",
#     ):
#         events = sign_script.get(key) or []
#         if not isinstance(events, list):
#             continue
#         for item in events:
#             if not isinstance(item, dict):
#                 continue
#             gloss = item.get("gloss_id") or item.get("gloss") or item.get("word") or ""
#             start = item.get("start_frame")
#             end = item.get("end_frame")
#             if start is None and item.get("start") is not None:
#                 start = float(item["start"]) * fps
#             if end is None and item.get("end") is not None:
#                 end = float(item["end"]) * fps
#             if start is None or end is None:
#                 continue
#             spans.append(
#                 {
#                     "gloss": str(gloss).strip(),
#                     "start_frame": int(round(float(start))),
#                     "end_frame": int(round(float(end))),
#                     "stream": key,
#                 }
#             )
#     spans.sort(key=lambda x: (x["start_frame"], x["end_frame"], x["stream"]))
#     return spans
# 
# 
# def _load_result(label_path: Path, fps: float, threshold: float) -> tuple[ExtractionResult, list[dict[str, Any]]]:
#     with open(label_path, encoding="utf-8") as f:
#         raw = json.load(f)
#     landmarks = raw.get("landmarks")
#     if not isinstance(landmarks, dict):
#         raise ValueError(f"landmarks missing in {label_path}")
# 
#     pose_raw = _reshape(_landmark_field(landmarks, "pose_keypoints"), 25, "pose_keypoints")
#     left_raw = _reshape(_landmark_field(landmarks, "hand_left_keypoints"), 21, "hand_left_keypoints")
#     right_raw = _reshape(_landmark_field(landmarks, "hand_right_keypoints"), 21, "hand_right_keypoints")
#     face70_raw = _reshape(_landmark_field(landmarks, "face_keypoints"), 70, "face_keypoints")
#     face_raw = face70_raw[:, :68, :]
# 
#     min_t = min(pose_raw.shape[0], left_raw.shape[0], right_raw.shape[0], face_raw.shape[0])
#     if min_t < 5:
#         raise ValueError(f"too few frames after alignment: {min_t}")
#     pose_raw = pose_raw[:min_t]
#     left_raw = left_raw[:min_t]
#     right_raw = right_raw[:min_t]
#     face_raw = face_raw[:min_t]
# 
#     pose_present, pose_quality = _presence_and_quality(pose_raw, threshold)
#     left_present, left_quality = _presence_and_quality(left_raw, threshold)
#     right_present, right_quality = _presence_and_quality(right_raw, threshold)
#     face_present, face_quality = _presence_and_quality(face_raw, threshold)
# 
#     center, width = shoulder_transform_params(pose_raw)
#     pose = apply_shoulder_transform(pose_raw, center, width)
#     face = apply_shoulder_transform(face_raw, center, width)
#     left, _ = normalize_landmarks(left_raw, method="bbox")
#     right, _ = normalize_landmarks(right_raw, method="bbox")
# 
#     presence = np.stack([pose_present, left_present, right_present, face_present], axis=1)
#     quality = np.stack([pose_quality, left_quality, right_quality, face_quality], axis=1)
#     face_blendshape = np.zeros((min_t, 52), dtype=np.float32)
# 
#     result = ExtractionResult(
#         pose=pose,
#         left_hand=left,
#         right_hand=right,
#         face_blendshape=face_blendshape,
#         face_key_subset=face,
#         presence_mask=presence.astype(bool),
#         quality_mask=quality.astype(np.float32),
#         meta={
#             "source": "aihub_official_landmarks_json",
#             "source_annotation_path": str(label_path),
#             "original_fps": float(fps),
#             "processed_fps": float(fps),
#             "frame_skip": 1,
#             "num_frames": int(min_t),
#             "processed_frame_indices": list(range(min_t)),
#         },
#     )
#     return result, _annotation_spans(raw, float(fps))
# 
# 
# def _resolve_label(raw_root: Path, sample: KSLSample) -> Path:
#     if sample.source_annotation_path:
#         direct = raw_root / sample.source_annotation_path
#         if direct.exists():
#             return direct
#     label_name = (sample.metadata or {}).get("label_file")
#     if label_name:
#         matches = list(raw_root.rglob(str(label_name)))
#         if matches:
#             return matches[0]
#     raise FileNotFoundError(f"source annotation not found for {sample.sample_id}")
# 
# 
# def main() -> int:
#     args = parse_args()
#     raw_root = Path(args.raw_root)
#     output_root = Path(args.output_root)
#     packer = LandmarkPacker(output_root)
# 
#     samples = list(read_manifest(args.manifest))
#     processed: list[KSLSample] = []
#     failed: list[str] = []
# 
#     for idx, sample in enumerate(samples, start=1):
#         out_dir = output_root / sample.sample_id
#         if args.skip_existing and (out_dir / "pose.npy").exists():
#             sample.keypoint_path = str(out_dir.relative_to(output_root.parent))
#             processed.append(sample)
#         else:
#             try:
#                 label_path = _resolve_label(raw_root, sample)
#                 result, spans = _load_result(label_path, sample.fps or 30.0, args.confidence_threshold)
#                 report = validate_extraction_result(sample.sample_id, result)
#                 flags = [f for f in sample.quality_flags if not f.startswith("low_face_presence")]
#                 flags = [f for f in flags if not f.startswith("extraction_error")]
#                 if not report.passed:
#                     flags.extend(f"extraction_error:{e}" for e in report.errors)
#                 sample.quality_flags = flags
#                 if spans:
#                     sample.metadata["annotation_spans"] = spans
#                 kp_dir = packer.pack(sample.sample_id, result)
#                 sample.keypoint_path = str(kp_dir.relative_to(output_root.parent))
#                 sample.crop_index_path = None
#                 processed.append(sample)
#             except Exception as exc:
#                 logger.error("[%s] official landmark packing failed: %s", sample.sample_id, exc)
#                 failed.append(sample.sample_id)
#                 processed.append(sample)
# 
#         if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(samples)):
#             logger.info("Progress: %s/%s failed=%s", idx, len(samples), len(failed))
# 
#     write_manifest(sorted(processed, key=lambda s: s.sample_id), args.manifest)
#     manifest_dir = Path(args.manifest).parent
#     for split_group in ("train", "valid", "test"):
#         split_path = manifest_dir / f"{split_group}.jsonl"
#         if split_path.exists():
#             write_manifest([s for s in processed if s.split_group == split_group], split_path)
# 
#     logger.info("Done: %s samples, failed=%s, output_root=%s", len(processed), len(failed), output_root)
#     if failed:
#         logger.warning("Failed sample ids: %s", failed[:20])
#         return 1
#     return 0
# 
# 
# if __name__ == "__main__":
#     raise SystemExit(main())
