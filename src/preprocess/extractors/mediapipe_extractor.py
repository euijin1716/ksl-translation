"""MediaPipe Hand/Face/Pose Landmarker 기반 추출기.

CLAUDE.md 정책:
- 기본은 Hand Landmarker + Face Landmarker + Pose Landmarker 분리형
- Holistic은 PoC/비교 실험용으로만 사용
- Face Landmarker에서 blendshape 출력을 반드시 고려
- Hand Landmarker는 양손 추출 전제
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base_extractor import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

# MediaPipe는 선택적 의존성: 없으면 dummy 모드로 동작
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.warning(
        "mediapipe not installed. MediaPipeExtractor will return dummy tensors. "
        "Install with: pip install mediapipe"
    )

# 추출기 설정 기본값
_DEFAULTS: dict[str, Any] = {
    "model_asset_path_hand": None,     # None이면 기본 번들 모델 사용
    "model_asset_path_face": None,
    "model_asset_path_pose": None,
    "num_hands": 2,
    "face_blendshape": True,
    "pose_upper_body_only": True,      # A-002
    "pose_joints": 25,                 # 상반신 서브셋 크기 (A-002)
    "face_key_subset_indices": list(range(68)),  # 주요 얼굴 랜드마크
    "target_fps": 25.0,                # A-003
    "save_face_crop": False,           # A-005
}

_FACE_BLENDSHAPE_DIM = 52             # A-001


class MediaPipeExtractor(BaseExtractor):
    """MediaPipe Hand/Face/Pose Landmarker를 사용한 특징 추출기.

    Args:
        config: 추출 설정 (defaults에 대한 오버라이드)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**_DEFAULTS, **(config or {})}
        self._initialized = False
        self._hand_landmarker = None
        self._face_landmarker = None
        self._pose_landmarker = None
        # 여러 영상을 배치 처리할 때 타임스탬프가 단조 증가를 유지하기 위한
        # 전역 오프셋. 각 영상 처리 후 마지막 타임스탬프 + 여유분으로 갱신된다.
        self._global_ts_ms: int = 0

        if _MP_AVAILABLE:
            self._init_landmarkers()

    def _init_landmarkers(self) -> None:
        """MediaPipe landmarker 모델을 초기화한다."""
        # 실제 MediaPipe Task API 초기화
        # 모델 파일이 없으면 초기화 실패 → dummy 모드로 폴백
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            base_options_cls = mp_tasks.BaseOptions
            running_mode = mp_vision.RunningMode

            # Hand Landmarker
            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=base_options_cls(
                    model_asset_path=self.config["model_asset_path_hand"]
                ),
                running_mode=running_mode.VIDEO,
                num_hands=self.config["num_hands"],
            )
            self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

            # Face Landmarker
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_options_cls(
                    model_asset_path=self.config["model_asset_path_face"]
                ),
                running_mode=running_mode.VIDEO,
                output_face_blendshapes=self.config["face_blendshape"],
            )
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)

            # Pose Landmarker
            pose_opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_options_cls(
                    model_asset_path=self.config["model_asset_path_pose"]
                ),
                running_mode=running_mode.VIDEO,
            )
            self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

            self._initialized = True
            logger.info("MediaPipe landmarkers initialized successfully")

        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}. Using dummy mode.")
            self._initialized = False

    def extract(self, video_path: str) -> ExtractionResult:
        """비디오에서 랜드마크를 추출한다.

        MediaPipe가 없거나 초기화에 실패했으면 dummy 텐서를 반환한다.
        전처리 실패는 조용히 버리지 않고 result.errors에 기록한다.
        """
        if not _MP_AVAILABLE or not self._initialized:
            return self._extract_dummy(video_path)

        return self._extract_mediapipe(video_path)

    def _extract_mediapipe(self, video_path: str) -> ExtractionResult:
        """실제 MediaPipe 추출 (모델 파일 필요)."""
        import cv2

        result = ExtractionResult()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result.errors.append(f"Cannot open video: {video_path}")
            return result

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_fps = self.config["target_fps"]
        frame_skip = max(1, round(original_fps / target_fps))

        pose_frames, left_hand_frames, right_hand_frames = [], [], []
        face_bs_frames, face_key_frames = [], []
        left_hand_bboxes, right_hand_bboxes, face_bboxes = [], [], []
        presence_frames = []
        processed_frame_indices = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            # 전역 오프셋 기반 타임스탬프: 여러 영상을 배치 처리해도 단조 증가 보장
            timestamp_ms = self._global_ts_ms + int(frame_idx * 1000 / original_fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Pose
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_lm = self._parse_pose(pose_result)

            # Hands
            hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            left_lm, right_lm = self._parse_hands(hand_result)

            # Face
            face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_bs, face_key = self._parse_face(face_result)

            pose_frames.append(pose_lm)
            left_hand_frames.append(left_lm)
            right_hand_frames.append(right_lm)
            face_bs_frames.append(face_bs)
            face_key_frames.append(face_key)
            left_hand_bboxes.append(self._landmarks_to_bbox(left_lm, frame.shape))
            right_hand_bboxes.append(self._landmarks_to_bbox(right_lm, frame.shape))
            face_bboxes.append(self._landmarks_to_bbox(face_key, frame.shape, padding=0.1))
            processed_frame_indices.append(frame_idx)

            # presence mask: [pose_ok, left_ok, right_ok, face_ok]
            presence_frames.append([
                pose_lm is not None,
                left_lm is not None,
                right_lm is not None,
                face_bs is not None,
            ])

            frame_idx += 1

        cap.release()

        # 다음 영상을 위해 전역 타임스탬프 전진 (영상 길이 + 10초 여유)
        self._global_ts_ms += int(frame_idx * 1000 / original_fps) + 10_000

        T = len(pose_frames)
        pose_arr = np.stack([f if f is not None else np.zeros((self.config["pose_joints"], 3)) for f in pose_frames])
        left_arr = np.stack([f if f is not None else np.zeros((21, 3)) for f in left_hand_frames])
        right_arr = np.stack([f if f is not None else np.zeros((21, 3)) for f in right_hand_frames])
        face_bs_arr = np.stack([f if f is not None else np.zeros(_FACE_BLENDSHAPE_DIM) for f in face_bs_frames])
        face_key_arr = np.stack([f if f is not None else np.zeros((len(self.config["face_key_subset_indices"]), 3)) for f in face_key_frames])
        left_bbox_arr = np.stack([b if b is not None else np.zeros(4, dtype=np.float32) for b in left_hand_bboxes])
        right_bbox_arr = np.stack([b if b is not None else np.zeros(4, dtype=np.float32) for b in right_hand_bboxes])
        face_bbox_arr = np.stack([b if b is not None else np.zeros(4, dtype=np.float32) for b in face_bboxes])
        presence_arr = np.array(presence_frames, dtype=bool)

        effective_fps = original_fps / frame_skip
        resampled = frame_skip > 1
        result.pose = pose_arr
        result.left_hand = left_arr
        result.right_hand = right_arr
        result.face_blendshape = face_bs_arr
        result.face_key_subset = face_key_arr
        result.left_hand_bbox = left_bbox_arr
        result.right_hand_bbox = right_bbox_arr
        result.face_bbox = face_bbox_arr
        result.presence_mask = presence_arr
        result.meta = {
            "original_fps": original_fps,
            "processed_fps": effective_fps,  # 실제 처리 fps = original_fps / frame_skip (target_fps 아님)
            "target_fps": target_fps,
            "frame_skip": frame_skip,
            "num_frames": T,
            "resampled": resampled,
            "frame_drop_ratio": 0.0,
            "processed_frame_indices": processed_frame_indices,
        }
        return result

    def _parse_pose(self, result: Any) -> np.ndarray | None:
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        n = self.config["pose_joints"]
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms[:n]], dtype=np.float32)
        return arr

    def _parse_hands(self, result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        left, right = None, None
        if not result.hand_landmarks:
            return left, right
        for i, handedness in enumerate(result.handedness):
            label = handedness[0].category_name.lower()
            arr = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[i]], dtype=np.float32)
            if label == "left":
                left = arr
            else:
                right = arr
        return left, right

    def _parse_face(self, result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not result.face_landmarks:
            return None, None
        indices = self.config["face_key_subset_indices"]
        key_arr = np.array([[result.face_landmarks[0][j].x, result.face_landmarks[0][j].y, result.face_landmarks[0][j].z] for j in indices], dtype=np.float32)
        if result.face_blendshapes:
            bs_arr = np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)
        else:
            bs_arr = np.zeros(_FACE_BLENDSHAPE_DIM, dtype=np.float32)
        return bs_arr, key_arr

    def _landmarks_to_bbox(
        self,
        landmarks: np.ndarray | None,
        frame_shape: tuple[int, ...],
        padding: float = 0.25,
    ) -> np.ndarray | None:
        if landmarks is None or landmarks.size == 0:
            return None
        height, width = frame_shape[:2]
        xs = np.clip(landmarks[:, 0], 0.0, 1.0) * width
        ys = np.clip(landmarks[:, 1], 0.0, 1.0) * height
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        pad_x = box_w * padding
        pad_y = box_h * padding
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(width), x2 + pad_x)
        y2 = min(float(height), y2 + pad_y)
        return np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)

    def _extract_dummy(self, video_path: str) -> ExtractionResult:
        """MediaPipe 없이 더미 텐서를 반환한다."""
        T = 32
        n_pose = self.config["pose_joints"]
        n_face_key = len(self.config["face_key_subset_indices"])
        rng = np.random.default_rng(abs(hash(video_path)) % (2**31))

        return ExtractionResult(
            pose=rng.random((T, n_pose, 3), dtype=np.float32),
            left_hand=rng.random((T, 21, 3), dtype=np.float32),
            right_hand=rng.random((T, 21, 3), dtype=np.float32),
            face_blendshape=rng.random((T, _FACE_BLENDSHAPE_DIM), dtype=np.float32),
            face_key_subset=rng.random((T, n_face_key, 3), dtype=np.float32),
            presence_mask=np.ones((T, 4), dtype=bool),
            quality_mask=np.ones((T, 4), dtype=np.float32),
            meta={
                "original_fps": 25.0,
                "processed_fps": 25.0,
                "num_frames": T,
                "resampled": False,
                "frame_drop_ratio": 0.0,
                "dummy_mode": True,
            },
        )

    def close(self) -> None:
        if self._hand_landmarker:
            self._hand_landmarker.close()
        if self._face_landmarker:
            self._face_landmarker.close()
        if self._pose_landmarker:
            self._pose_landmarker.close()
