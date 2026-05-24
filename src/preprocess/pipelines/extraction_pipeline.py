"""End-to-end 추출 파이프라인.

단일 샘플을 받아 추출 → 검증 → 저장 → crop index 업데이트까지 처리한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ...data.schema import KSLSample
from ..croppers.roi_cropper import ROICropper
from ..extractors.mediapipe_extractor import MediaPipeExtractor
from ..normalizers.coordinate_normalizer import normalize_landmarks
from ..packers.landmark_packer import LandmarkPacker
from ..validators.shape_validator import validate_extraction_result

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """수어 영상 → 특징 텐서 파이프라인.

    Args:
        config: 파이프라인 설정 dict. 주요 키:
            - keypoint_root: npy 저장 루트
            - crop_root: crop 저장 루트
            - save_face_crop: face crop 저장 여부 (기본 False)
            - normalize_method: "shoulder_width" | "bbox" | "none"
            - extractor_config: MediaPipeExtractor 설정
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.keypoint_root = Path(config.get("keypoint_root", "data/keypoints"))
        self.crop_root = Path(config.get("crop_root", "data/crops"))
        self.save_face_crop = config.get("save_face_crop", False)
        self.enable_crops = config.get("enable_crops", True)
        self.crop_missing_from_saved_bbox = config.get("crop_missing_from_saved_bbox", True)
        self.normalize_method = config.get("normalize_method", "shoulder_width")
        self.skip_existing = config.get("skip_existing", True)
        self.progress_every = int(config.get("progress_every", 25))
        self.video_roots = [
            Path(p) for p in config.get(
                "video_roots",
                ["data/raw/aihub_sign", "data/aihub_sign", "."],
            )
        ]

        self.extractor = MediaPipeExtractor(config.get("extractor_config"))
        self.packer = LandmarkPacker(self.keypoint_root)
        self.cropper = ROICropper(
            self.crop_root,
            save_face_crop=self.save_face_crop,
        )

    def process(self, sample: KSLSample) -> KSLSample:
        """단일 샘플을 처리하고 keypoint_path/crop_index_path가 업데이트된 샘플을 반환한다.

        실패해도 조용히 버리지 않고 quality_flags에 기록한다.
        """
        existing_dir = self.keypoint_root / sample.sample_id
        if self.skip_existing and (existing_dir / "pose.npy").exists():
            sample.keypoint_path = str(existing_dir.relative_to(self.keypoint_root.parent))
            existing_crop_index = self.crop_root / sample.sample_id / "crop_index.json"
            if existing_crop_index.exists():
                sample.crop_index_path = str(existing_crop_index.relative_to(self.crop_root.parent))
            elif self.enable_crops and self.crop_missing_from_saved_bbox:
                self._crop_from_saved_bbox(sample, existing_dir)
            return sample

        video_path = self._resolve_video_path(sample.video_path)
        result = self.extractor.extract(str(video_path))

        # 좌표 정규화
        if result.pose is not None:
            result.pose, _ = normalize_landmarks(result.pose, self.normalize_method)
        if result.left_hand is not None:
            result.left_hand, _ = normalize_landmarks(result.left_hand, method="bbox")
        if result.right_hand is not None:
            result.right_hand, _ = normalize_landmarks(result.right_hand, method="bbox")

        # 검증
        report = validate_extraction_result(sample.sample_id, result)
        if not report.passed:
            logger.warning(
                f"[{sample.sample_id}] Extraction validation failed: {report.errors}"
            )
            sample.quality_flags.extend([f"extraction_error:{e}" for e in report.errors])

        # 저장
        kp_dir = self.packer.pack(sample.sample_id, result)
        sample.keypoint_path = str(kp_dir.relative_to(self.keypoint_root.parent))

        if self.enable_crops:
            self._crop_from_result(sample, str(video_path), result)

        return sample

    def _crop_from_result(self, sample: KSLSample, video_path: str, result: Any) -> None:
        if result.left_hand_bbox is None and result.right_hand_bbox is None:
            return
        self.cropper.crop_video(
            sample.sample_id,
            video_path,
            result.left_hand_bbox,
            result.right_hand_bbox,
            result.face_bbox,
            frame_indices=result.meta.get("processed_frame_indices"),
        )
        sample.crop_index_path = str(
            (self.crop_root / sample.sample_id / "crop_index.json").relative_to(self.crop_root.parent)
        )

    def _crop_from_saved_bbox(self, sample: KSLSample, existing_dir: Path) -> None:
        left_bbox = self._load_optional_npy(existing_dir / "left_hand_bbox.npy")
        right_bbox = self._load_optional_npy(existing_dir / "right_hand_bbox.npy")
        if left_bbox is None and right_bbox is None:
            return
        face_bbox = self._load_optional_npy(existing_dir / "face_bbox.npy")
        frame_indices = None
        meta_path = existing_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    frame_indices = json.load(f).get("processed_frame_indices")
            except (OSError, json.JSONDecodeError):
                frame_indices = None
        video_path = self._resolve_video_path(sample.video_path)
        self.cropper.crop_video(
            sample.sample_id,
            str(video_path),
            left_bbox,
            right_bbox,
            face_bbox,
            frame_indices=frame_indices,
        )
        sample.crop_index_path = str(
            (self.crop_root / sample.sample_id / "crop_index.json").relative_to(self.crop_root.parent)
        )

    @staticmethod
    def _load_optional_npy(path: Path) -> Any:
        if not path.exists():
            return None
        import numpy as np

        return np.load(path)

    def _resolve_video_path(self, video_path: str) -> Path:
        """manifest의 상대 video_path를 실제 파일 경로로 해석한다."""
        path = Path(video_path)
        if path.exists():
            return path
        if path.is_absolute():
            return path
        for root in self.video_roots:
            candidate = root / path
            if candidate.exists():
                return candidate
        return path

    def process_batch(self, samples: list[KSLSample]) -> tuple[list[KSLSample], list[str]]:
        """여러 샘플을 순차 처리한다.

        Returns:
            (성공 샘플 목록, 실패 sample_id 목록)
        """
        processed, failed = [], []
        total = len(samples)
        for i, sample in enumerate(samples, start=1):
            try:
                processed.append(self.process(sample))
            except Exception as e:
                logger.error(f"[{sample.sample_id}] Pipeline error: {e}")
                failed.append(sample.sample_id)
            if self.progress_every > 0 and (i % self.progress_every == 0 or i == total):
                logger.info(
                    f"Preprocess progress: {i}/{total} "
                    f"(ok={len(processed)}, failed={len(failed)})"
                )
        return processed, failed

    def close(self) -> None:
        self.extractor.close()

    def __enter__(self) -> "ExtractionPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
