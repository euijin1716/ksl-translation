"""랜드마크 좌표 정규화.

사람마다 비교 가능한 좌표 공간으로 변환한다.
정규화 이력은 메타데이터에 기록해야 한다.
"""

from __future__ import annotations

import numpy as np

# 어깨 너비가 이 값 미만이면 pose 미검출/어깨 겹침(퇴화)으로 보고, ÷~0 폭발 대신 0 처리한다.
# 실제 어깨 너비(정규화 이미지좌표)는 ~0.1~0.4라 충분히 분리된다.
_MIN_SHOULDER_WIDTH = 1e-3


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
            # 어깨 너비 퇴화(pose 미검출) 프레임은 ÷~0 폭발 대신 XY를 0으로
            normalized[width[:, 0] < _MIN_SHOULDER_WIDTH, :, :2] = 0.0
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


def shoulder_transform_params(
    pose: np.ndarray,
    reference_indices: tuple[int, int] = (11, 12),
) -> tuple[np.ndarray, np.ndarray]:
    """pose 어깨로부터 프레임별 (center[T,2], width[T,1]) 변환 파라미터를 구한다.

    `normalize_landmarks(method="shoulder_width")`와 동일한 기준을 산출하므로,
    pose와 face_key 등 여러 스트림을 **같은 몸 기준 좌표계**로 정규화할 때 공유한다.
    """
    li, ri = reference_indices
    left = pose[:, li, :2]
    right = pose[:, ri, :2]
    center = (left + right) / 2.0
    width = np.linalg.norm(right - left, axis=-1, keepdims=True)
    width = np.maximum(width, 1e-6)
    return center, width


def apply_shoulder_transform(
    landmarks: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    """미리 구한 어깨 기준 변환(center/width)을 landmarks의 XY에 적용한다.

    Args:
        landmarks: [T, J, 3]
        center:    [T, 2]  (shoulder_transform_params 산출)
        width:     [T, 1]
    Z는 건드리지 않는다 (shoulder_width 정규화와 동일).
    """
    out = landmarks.copy()
    out[:, :, :2] -= center[:, np.newaxis, :]
    out[:, :, :2] /= width[:, np.newaxis, :]
    # 어깨 너비 퇴화(pose 미검출) 프레임은 ÷~0 폭발 대신 XY를 0으로
    out[width[:, 0] < _MIN_SHOULDER_WIDTH, :, :2] = 0.0
    return out.astype(np.float32)
