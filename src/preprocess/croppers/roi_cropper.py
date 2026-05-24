"""ROI crop 저장 모듈.

crop과 landmark는 별도 파일에 저장한다.
face crop은 save_face_crop 옵션이 활성화된 경우만 저장한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# OpenCV는 선택적 의존성
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class ROICropper:
    """비디오 프레임에서 손/얼굴 ROI를 잘라 저장한다.

    Args:
        output_root: crop 저장 루트 디렉터리
        crop_size: (height, width) 리사이즈 크기
        save_face_crop: True이면 face crop 저장 (기본 False, A-005)
    """

    def __init__(
        self,
        output_root: str | Path,
        crop_size: tuple[int, int] = (112, 112),
        save_face_crop: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.crop_size = crop_size
        self.save_face_crop = save_face_crop

    def crop_video(
        self,
        sample_id: str,
        video_path: str,
        left_hand_bbox: np.ndarray | None,
        right_hand_bbox: np.ndarray | None,
        face_bbox: np.ndarray | None,
        frame_indices: list[int] | np.ndarray | None = None,
    ) -> dict:
        """비디오에서 각 프레임의 ROI를 저장하고 crop index를 반환한다.

        Args:
            sample_id: 샘플 식별자
            video_path: 원본 비디오 경로
            *_bbox: [T, 4] 형태의 bbox (x, y, w, h) 또는 None

        Returns:
            crop index dict {"left_hand_dir": ..., "face_dir": ...}
        """
        if not _CV2_AVAILABLE:
            return self._dummy_crop_index(sample_id)

        out = self.output_root / sample_id / "crops"
        (out / "left_hand").mkdir(parents=True, exist_ok=True)
        (out / "right_hand").mkdir(parents=True, exist_ok=True)
        if self.save_face_crop:
            (out / "face").mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        crop_index: dict = {"frames": []}
        frame_map = None
        if frame_indices is not None:
            frame_map = {int(src_idx): i for i, src_idx in enumerate(frame_indices)}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            source_idx = frame_map.get(frame_idx) if frame_map is not None else frame_idx
            if source_idx is None:
                frame_idx += 1
                continue

            frame_info: dict = {"frame_idx": frame_idx, "source_frame_idx": int(source_idx)}

            if left_hand_bbox is not None and source_idx < len(left_hand_bbox):
                crop = self._crop_frame(frame, left_hand_bbox[source_idx])
                if crop is not None:
                    path = out / "left_hand" / f"{source_idx:05d}.jpg"
                    cv2.imwrite(str(path), crop)
                    frame_info["left_hand"] = str(path.relative_to(self.output_root))

            if right_hand_bbox is not None and source_idx < len(right_hand_bbox):
                crop = self._crop_frame(frame, right_hand_bbox[source_idx])
                if crop is not None:
                    path = out / "right_hand" / f"{source_idx:05d}.jpg"
                    cv2.imwrite(str(path), crop)
                    frame_info["right_hand"] = str(path.relative_to(self.output_root))

            if self.save_face_crop and face_bbox is not None and source_idx < len(face_bbox):
                crop = self._crop_frame(frame, face_bbox[source_idx])
                if crop is not None:
                    path = out / "face" / f"{source_idx:05d}.jpg"
                    cv2.imwrite(str(path), crop)
                    frame_info["face"] = str(path.relative_to(self.output_root))

            crop_index["frames"].append(frame_info)
            frame_idx += 1

        cap.release()

        index_path = self.output_root / sample_id / "crop_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(crop_index, f)

        return crop_index

    def _crop_frame(self, frame: "np.ndarray", bbox: np.ndarray) -> "np.ndarray | None":
        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            return None
        H, W = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (self.crop_size[1], self.crop_size[0]))

    def _dummy_crop_index(self, sample_id: str) -> dict:
        return {"sample_id": sample_id, "dummy": True, "frames": []}
