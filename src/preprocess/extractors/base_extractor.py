"""추출기 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExtractionResult:
    """단일 비디오에 대한 추출 결과.

    None 필드는 해당 스트림을 추출하지 않았거나 실패했음을 의미한다.
    """
    # ── 랜드마크 ──────────────────────────────────────────────────────────────
    pose: np.ndarray | None = None              # [T, 25, 3]
    pose_world: np.ndarray | None = None        # [T, 25, 3] (선택)
    left_hand: np.ndarray | None = None         # [T, 21, 3]
    right_hand: np.ndarray | None = None        # [T, 21, 3]
    left_hand_world: np.ndarray | None = None   # [T, 21, 3] (선택)
    right_hand_world: np.ndarray | None = None  # [T, 21, 3] (선택)
    face_blendshape: np.ndarray | None = None   # [T, 52]
    face_key_subset: np.ndarray | None = None   # [T, N, 3]

    # ── 마스크 ────────────────────────────────────────────────────────────────
    presence_mask: np.ndarray | None = None     # [T, num_streams] bool
    quality_mask: np.ndarray | None = None      # [T, num_streams] float

    # ── crop 프레임 인덱스 ────────────────────────────────────────────────────
    # 실제 이미지는 croppers가 별도로 저장
    left_hand_bbox: np.ndarray | None = None    # [T, 4] (x, y, w, h)
    right_hand_bbox: np.ndarray | None = None   # [T, 4]
    face_bbox: np.ndarray | None = None         # [T, 4]

    # ── 메타데이터 ────────────────────────────────────────────────────────────
    meta: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """모든 추출기가 구현해야 하는 인터페이스."""

    @abstractmethod
    def extract(self, video_path: str) -> ExtractionResult:
        """비디오 파일에서 랜드마크 및 특징을 추출한다."""
        ...

    @abstractmethod
    def close(self) -> None:
        """리소스를 해제한다."""
        ...

    def __enter__(self) -> "BaseExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
