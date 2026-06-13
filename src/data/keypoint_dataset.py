"""Manifest-backed keypoint dataset for Stage C training/evaluation."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .gloss_vocab import GlossVocab
from .manifest import read_manifest
from .schema import KSLSample
from .signals import encode_nms_detail_labels, encode_nms_labels

logger = logging.getLogger(__name__)

_POSE_JOINTS = 25
_HAND_JOINTS = 21
_FACE_BS_DIM = 52
_FACE_KEY_DIM = 68
_HAND_CROP_H = 112
_HAND_CROP_W = 112

_DOMAIN_LIST = ["hospital", "directions", "order", "reservation", "public", "help", "unknown"]


class KeypointDataset(Dataset):
    """Read preprocessed numpy keypoints and optional ROI crops from a manifest.

    Activity labels use three classes: 0=idle, 1=signing, 2=boundary. If
    manifest metadata contains ``annotation_spans`` with frame ranges, labels
    are derived from those spans. Otherwise they fall back to hand presence and
    motion, which is still more informative than a single all-active class.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        keypoint_root: str | Path = "data/keypoints",
        split_group: str | None = None,
        gloss_vocab: GlossVocab | None = None,
        tokenizer: Any = None,
        max_seq_len: int = 512,
        max_text_len: int = 64,
        skip_missing: bool = True,
        crop_root: str | Path = "data/crops",
        load_hand_crops: bool = True,
        sampling_strategy: str = "uniform",
        boundary_mode: str = "annotation_or_motion",
    ) -> None:
        self.keypoint_root = Path(keypoint_root)
        self.crop_root = Path(crop_root)
        self.gloss_vocab = gloss_vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.load_hand_crops = load_hand_crops
        self.sampling_strategy = sampling_strategy
        self.boundary_mode = boundary_mode

        if tokenizer is not None:
            self.pad_id = tokenizer.pad_token_id or 1
            self.bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
            self.eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
        else:
            self.pad_id, self.bos_id, self.eos_id = 1, 0, 2

        all_samples = list(read_manifest(manifest_path))
        if split_group is not None:
            all_samples = [s for s in all_samples if s.split_group == split_group]

        self.samples: list[KSLSample] = []
        skipped = 0
        for sample in all_samples:
            kp_dir = self._resolve_keypoint_dir(sample)
            if kp_dir is None or not (kp_dir / "pose.npy").exists():
                if skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(
                    f"[{sample.sample_id}] keypoint_path missing or pose.npy not found: {kp_dir}"
                )
            self.samples.append(sample)

        self.domain_counts = Counter(s.domain for s in self.samples)
        self.intent_source_counts = Counter(s.intent_source for s in self.samples)
        if skipped:
            logger.warning(
                "KeypointDataset: %s samples skipped because keypoints were missing.",
                skipped,
            )
        logger.info(
            "KeypointDataset loaded: %s samples%s",
            len(self.samples),
            f" (split={split_group})" if split_group else "",
        )
        if len(self.domain_counts) <= 1 and self.samples:
            logger.warning(
                "KeypointDataset split has a single domain distribution: %s. "
                "Intent accuracy is not discriminative for this split.",
                dict(self.domain_counts),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        kp_dir = self._resolve_keypoint_dir(sample)
        if kp_dir is None:
            raise FileNotFoundError(f"[{sample.sample_id}] keypoint_path is missing")

        pose_arr = self._load_npy(kp_dir, "pose", (_POSE_JOINTS, 3))
        left_hand_arr = self._load_npy(kp_dir, "left_hand", (_HAND_JOINTS, 3))
        right_hand_arr = self._load_npy(kp_dir, "right_hand", (_HAND_JOINTS, 3))
        face_bs_arr = self._load_npy(kp_dir, "face_blendshape", (_FACE_BS_DIM,))
        face_key_arr = self._load_npy(kp_dir, "face_key_subset", (_FACE_KEY_DIM, 3))
        presence_arr = self._load_npy(kp_dir, "presence_mask", (4,), dtype=bool)

        frame_idx = self._make_frame_indices(pose_arr.shape[0])
        T = int(frame_idx.shape[0])

        pose = torch.from_numpy(self._select_frames(pose_arr, frame_idx).astype(np.float32))
        left_hand = torch.from_numpy(self._select_frames(left_hand_arr, frame_idx).astype(np.float32))
        right_hand = torch.from_numpy(self._select_frames(right_hand_arr, frame_idx).astype(np.float32))
        face_bs = torch.from_numpy(self._select_frames(face_bs_arr, frame_idx).astype(np.float32))
        face_key = torch.from_numpy(self._select_frames(face_key_arr, frame_idx).astype(np.float32))
        presence_np = self._select_frames(presence_arr, frame_idx).astype(bool)
        presence = torch.from_numpy(presence_np)

        if self.gloss_vocab is not None and sample.gloss_tokens:
            gloss_ids = torch.tensor(self.gloss_vocab.encode(sample.gloss_tokens), dtype=torch.long)
        else:
            gloss_ids = torch.tensor([GlossVocab.BLANK_ID], dtype=torch.long)

        domain = sample.domain if sample.domain in _DOMAIN_LIST else "unknown"
        intent_label = torch.tensor(_DOMAIN_LIST.index(domain), dtype=torch.long)

        nms_label, nms_mask = encode_nms_labels(sample.nms_labels)
        nms_detail_label, nms_detail_mask = encode_nms_detail_labels(sample.nms_labels)

        activity = torch.from_numpy(
            self._build_activity_labels(sample, presence_arr, left_hand_arr, right_hand_arr, frame_idx)
        ).long()

        tgt_tokens, draft_labels = self._encode_korean(sample.korean_text)

        result: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "split_group": sample.split_group,
            "korean_text": sample.korean_text,
            "gloss_tokens_raw": sample.gloss_tokens or [],
            "nms_labels_raw": sample.nms_labels.to_dict() if sample.nms_labels is not None else {},
            "pose": pose,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "face_blendshape": face_bs,
            "face_key_subset": face_key,
            "presence_mask": presence,
            "source_frame_idx": torch.from_numpy(frame_idx.astype(np.int64)),
            "gloss_ids": gloss_ids,
            "nms_label": nms_label,
            "nms_mask": nms_mask,
            "nms_detail_label": nms_detail_label,
            "nms_detail_mask": nms_detail_mask,
            "intent_label": intent_label,
            "activity": activity,
            "seq_len": torch.tensor(T),
            "tgt_tokens": tgt_tokens,
            "draft_labels": draft_labels,
        }
        if self.load_hand_crops:
            result.update(self._load_hand_crops(sample, frame_idx))
        return result

    def _resolve_keypoint_dir(self, sample: KSLSample) -> Path | None:
        if sample.keypoint_path is None:
            return None
        kp = _manifest_path(sample.keypoint_path)
        if kp.is_absolute():
            return kp
        full = self.keypoint_root.parent / kp
        if full.exists():
            return full
        return self.keypoint_root / kp

    def _load_npy(
        self,
        kp_dir: Path,
        name: str,
        frame_shape: tuple,
        dtype: type = np.float32,
    ) -> np.ndarray:
        path = kp_dir / f"{name}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.ndim == 1 and len(frame_shape) == 1:
                arr = arr.reshape(-1, *frame_shape) if arr.shape != frame_shape else arr[np.newaxis]
            if dtype == bool:
                return arr.astype(bool)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        logger.debug("Missing %s.npy for %s, using zeros.", name, kp_dir.name)
        if dtype == bool:
            return np.zeros((1, *frame_shape), dtype=bool)
        return np.zeros((1, *frame_shape), dtype=np.float32)

    @staticmethod
    def _select_frames(arr: np.ndarray, frame_idx: np.ndarray) -> np.ndarray:
        if arr.shape[0] == 0:
            raise ValueError("Cannot select frames from an empty array.")
        clipped = np.clip(frame_idx, 0, arr.shape[0] - 1)
        return arr[clipped]

    def _make_frame_indices(self, num_frames: int) -> np.ndarray:
        num_frames = max(int(num_frames), 1)
        if num_frames <= self.max_seq_len:
            return np.arange(num_frames, dtype=np.int64)

        strategy = str(self.sampling_strategy or "uniform").lower()
        if strategy in {"head", "front", "truncate"}:
            return np.arange(self.max_seq_len, dtype=np.int64)
        if strategy == "center":
            start = max((num_frames - self.max_seq_len) // 2, 0)
            return np.arange(start, start + self.max_seq_len, dtype=np.int64)
        return np.linspace(0, num_frames - 1, self.max_seq_len).round().astype(np.int64)

    def _build_activity_labels(
        self,
        sample: KSLSample,
        presence_arr: np.ndarray,
        left_hand_arr: np.ndarray,
        right_hand_arr: np.ndarray,
        frame_idx: np.ndarray,
    ) -> np.ndarray:
        ann_labels = self._activity_from_annotation(sample, frame_idx)
        if ann_labels is not None:
            return ann_labels

        mode = str(self.boundary_mode or "annotation_or_motion").lower()
        if mode == "annotation":
            labels = np.ones(len(frame_idx), dtype=np.int64)
            if labels.size:
                labels[0] = 2
                labels[-1] = 2
            return labels
        return self._activity_from_motion(presence_arr, left_hand_arr, right_hand_arr, frame_idx)

    def _activity_from_annotation(self, sample: KSLSample, frame_idx: np.ndarray) -> np.ndarray | None:
        spans = sample.metadata.get("annotation_spans") if isinstance(sample.metadata, dict) else None
        if not isinstance(spans, list) or not spans:
            return None

        meta = self._load_keypoint_meta(sample)
        original_fps = float(meta.get("original_fps") or sample.fps or 25.0)
        processed_fps = float(meta.get("processed_fps") or original_fps)
        sampled_original = frame_idx.astype(np.float32) * (original_fps / max(processed_fps, 1e-6))

        labels = np.zeros(len(frame_idx), dtype=np.int64)
        boundary_positions: set[int] = set()
        for span in spans:
            if not isinstance(span, dict):
                continue
            start = self._to_float(span.get("start_frame"))
            end = self._to_float(span.get("end_frame"))
            if start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            active = (sampled_original >= start) & (sampled_original <= end)
            labels[active] = 1
            if len(frame_idx) > 0:
                boundary_positions.add(int(np.argmin(np.abs(sampled_original - start))))
                boundary_positions.add(int(np.argmin(np.abs(sampled_original - end))))

        for pos in boundary_positions:
            labels[pos] = 2
        return labels if np.any(labels != 0) else None

    def _activity_from_motion(
        self,
        presence_arr: np.ndarray,
        left_hand_arr: np.ndarray,
        right_hand_arr: np.ndarray,
        frame_idx: np.ndarray,
    ) -> np.ndarray:
        if len(frame_idx) == 0:
            return np.zeros(0, dtype=np.int64)

        presence_sampled = self._select_frames(presence_arr, frame_idx)
        active = presence_sampled[:, 1] | presence_sampled[:, 2]
        if not active.any():
            active = np.ones(len(frame_idx), dtype=bool)

        labels = np.where(active, 1, 0).astype(np.int64)
        active_idx = np.flatnonzero(active)
        if active_idx.size:
            labels[active_idx[0]] = 2
            labels[active_idx[-1]] = 2

        motion = self._hand_motion(left_hand_arr, right_hand_arr, frame_idx)
        if motion.size >= 3 and float(motion.max()) > 0.0:
            threshold = float(np.quantile(motion, 0.9))
            for pos in np.flatnonzero(motion >= threshold):
                if active[pos]:
                    labels[pos] = 2
        return labels

    def _hand_motion(
        self,
        left_hand_arr: np.ndarray,
        right_hand_arr: np.ndarray,
        frame_idx: np.ndarray,
    ) -> np.ndarray:
        left = self._select_frames(left_hand_arr, frame_idx).reshape(len(frame_idx), -1)
        right = self._select_frames(right_hand_arr, frame_idx).reshape(len(frame_idx), -1)
        combined = np.concatenate([left, right], axis=1)
        if len(frame_idx) <= 1:
            return np.zeros(len(frame_idx), dtype=np.float32)
        motion = np.linalg.norm(np.diff(combined, axis=0), axis=1)
        return np.concatenate([[0.0], motion]).astype(np.float32)

    def _load_keypoint_meta(self, sample: KSLSample) -> dict[str, Any]:
        kp_dir = self._resolve_keypoint_dir(sample)
        if kp_dir is None:
            return {}
        path = kp_dir / "meta.json"
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_hand_crops(self, sample: KSLSample, frame_idx: np.ndarray) -> dict[str, torch.Tensor]:
        index_path = self._resolve_crop_index_path(sample)
        if index_path is None or not index_path.exists():
            return {}
        try:
            with open(index_path, encoding="utf-8") as f:
                crop_index = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("[%s] crop_index load failed: %s", sample.sample_id, index_path)
            return {}

        frames = crop_index.get("frames", [])
        if not isinstance(frames, list):
            return {}
        by_processed_idx = {
            int(item.get("source_frame_idx", item.get("frame_idx"))): item
            for item in frames
            if isinstance(item, dict) and ("source_frame_idx" in item or "frame_idx" in item)
        }
        if not by_processed_idx:
            return {}

        left, right = [], []
        crop_root = index_path.parent.parent
        for idx in frame_idx.tolist():
            item = by_processed_idx.get(int(idx), {})
            left.append(self._read_crop(crop_root, item.get("left_hand")))
            right.append(self._read_crop(crop_root, item.get("right_hand")))
        return {
            "left_hand_crop": torch.from_numpy(np.stack(left).astype(np.float32)),
            "right_hand_crop": torch.from_numpy(np.stack(right).astype(np.float32)),
        }

    def _resolve_crop_index_path(self, sample: KSLSample) -> Path | None:
        if sample.crop_index_path is None:
            return None
        path = _manifest_path(sample.crop_index_path)
        if path.is_absolute():
            return path
        full = self.crop_root.parent / path
        if full.exists():
            return full
        return self.crop_root / path

    def _read_crop(self, crop_root: Path, rel_path: Any) -> np.ndarray:
        zero = np.zeros((3, _HAND_CROP_H, _HAND_CROP_W), dtype=np.float32)
        if not rel_path:
            return zero
        try:
            import cv2
        except ImportError:
            return zero

        path = crop_root / str(rel_path)
        img = cv2.imread(str(path))
        if img is None:
            return zero
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (_HAND_CROP_W, _HAND_CROP_H))
        return np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _encode_korean(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None or not text:
            bos = torch.tensor([self.bos_id], dtype=torch.long)
            eos = torch.tensor([self.eos_id], dtype=torch.long)
            return bos, eos

        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            truncation=True,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]
        if len(ids) == 0:
            ids = [self.tokenizer.unk_token_id or 3]

        tgt_tokens = torch.tensor([self.bos_id] + ids, dtype=torch.long)
        draft_labels = torch.tensor(ids + [self.eos_id], dtype=torch.long)
        return tgt_tokens, draft_labels


def _manifest_path(value: str | Path) -> Path:
    """Convert manifest paths written on Windows into POSIX-friendly paths."""
    return Path(str(value).replace("\\", "/"))
