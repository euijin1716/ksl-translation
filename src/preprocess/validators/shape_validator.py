"""추출 결과의 shape 및 품질 검증."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..extractors.base_extractor import ExtractionResult


@dataclass
class ValidationReport:
    sample_id: str
    passed: bool
    errors: list[str]
    warnings: list[str]
    stats: dict


def validate_extraction_result(
    sample_id: str,
    result: ExtractionResult,
    expected_pose_joints: int = 25,
    expected_hand_joints: int = 21,
    expected_blendshape: int = 52,
    min_frames: int = 5,
) -> ValidationReport:
    """추출 결과의 shape, 값 범위, 품질을 검증한다."""
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    def _check(name: str, arr: np.ndarray | None, expected_shape_suffix: tuple) -> None:
        if arr is None:
            warnings.append(f"{name} is None (not extracted)")
            return
        for i, (actual, expected) in enumerate(zip(arr.shape[1:], expected_shape_suffix)):
            if expected is not None and actual != expected:
                errors.append(f"{name} shape mismatch at dim {i+1}: expected {expected}, got {actual}")

    T = None
    if result.pose is not None:
        T = result.pose.shape[0]
        _check("pose", result.pose, (expected_pose_joints, 3))

    _check("left_hand", result.left_hand, (expected_hand_joints, 3))
    _check("right_hand", result.right_hand, (expected_hand_joints, 3))
    _check("face_blendshape", result.face_blendshape, (expected_blendshape,))

    if T is not None and T < min_frames:
        errors.append(f"Too few frames: {T} < {min_frames}")

    # 품질 통계
    if result.presence_mask is not None:
        rates = result.presence_mask.mean(axis=0)
        stats["presence_rates"] = rates.tolist()
        for i, rate in enumerate(rates):
            if rate < 0.5:
                warnings.append(f"Stream {i} has low presence rate: {rate:.2f}")

    errors.extend(result.errors)

    return ValidationReport(
        sample_id=sample_id,
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
