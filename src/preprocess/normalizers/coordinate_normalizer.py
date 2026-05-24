"""랜드마크 좌표 정규화.

사람마다 비교 가능한 좌표 공간으로 변환한다.
정규화 이력은 메타데이터에 기록해야 한다.
"""

from __future__ import annotations

import numpy as np


def normalize_landmarks(
    landmarks: np.ndarray,
    method: str = "shoulder_width",
    reference_indices: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """랜드마크 좌표를 정규화한다.

    Args:
        landmarks: [T, J, 3] 형태의 랜드마크 배열
        method: "shoulder_width" | "bbox" | "none"
        reference_indices: shoulder_width 기준 관절 인덱스 (왼쪽, 오른쪽)

    Returns:
        (정규화된 [T, J, 3] 배열, 정규화 메타데이터 dict)
    """
    if method == "none":
        return landmarks.copy(), {"method": "none"}

    T, J, D = landmarks.shape
    normalized = landmarks.copy()
    meta: dict = {"method": method}

    if method == "shoulder_width":
        # pose 기준 어깨 너비로 스케일, 어깨 중심으로 이동
        # 기본 어깨 인덱스: 11(왼쪽), 12(오른쪽) (MediaPipe Pose)
        li, ri = reference_indices or (11, 12)
        if J > max(li, ri):
            left_shoulder = normalized[:, li, :2]   # [T, 2]
            right_shoulder = normalized[:, ri, :2]  # [T, 2]
            center = (left_shoulder + right_shoulder) / 2.0
            width = np.linalg.norm(right_shoulder - left_shoulder, axis=-1, keepdims=True)  # [T, 1]
            width = np.maximum(width, 1e-6)

            # XY 이동: 어깨 중심을 원점으로
            normalized[:, :, :2] -= center[:, np.newaxis, :]
            # XY 스케일: 어깨 너비로 나눔
            normalized[:, :, :2] /= width[:, np.newaxis, :]
            meta["scale_by"] = "shoulder_width"
            meta["center_by"] = "shoulder_midpoint"
        else:
            meta["warning"] = f"reference_indices ({li},{ri}) out of range for J={J}"

    elif method == "bbox":
        # 전체 랜드마크 bounding box로 정규화
        xy = normalized[:, :, :2]
        mins = xy.min(axis=1, keepdims=True)  # [T, 1, 2]
        maxs = xy.max(axis=1, keepdims=True)  # [T, 1, 2]
        scale = np.maximum(maxs - mins, 1e-6)
        normalized[:, :, :2] = (xy - mins) / scale
        meta["scale_by"] = "bbox"

    return normalized.astype(np.float32), meta
