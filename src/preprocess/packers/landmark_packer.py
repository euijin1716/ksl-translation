"""ExtractionResult → NPZ 파일로 패킹하는 모듈."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..extractors.base_extractor import ExtractionResult


class LandmarkPacker:
    """추출 결과를 개별 npy 파일로 저장한다.

    디렉터리 구조:
        {output_root}/{sample_id}/pose.npy
                                 /left_hand.npy
                                 ...
                                 /meta.json
    """

    def __init__(self, output_root: str | Path, save_world: bool = False) -> None:
        self.output_root = Path(output_root)
        self.save_world = save_world

    def pack(self, sample_id: str, result: ExtractionResult) -> Path:
        """추출 결과를 저장하고 저장 디렉터리 경로를 반환한다."""
        out_dir = self.output_root / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _save(name: str, arr: np.ndarray | None) -> None:
            if arr is not None:
                np.save(out_dir / f"{name}.npy", arr)

        _save("pose", result.pose)
        _save("left_hand", result.left_hand)
        _save("right_hand", result.right_hand)
        _save("face_blendshape", result.face_blendshape)
        _save("face_key_subset", result.face_key_subset)
        _save("presence_mask", result.presence_mask)
        _save("quality_mask", result.quality_mask)
        _save("left_hand_bbox", result.left_hand_bbox)
        _save("right_hand_bbox", result.right_hand_bbox)
        _save("face_bbox", result.face_bbox)

        if self.save_world:
            _save("pose_world", result.pose_world)
            _save("left_hand_world", result.left_hand_world)
            _save("right_hand_world", result.right_hand_world)

        meta = dict(result.meta)
        meta["sample_id"] = sample_id
        meta["errors"] = result.errors
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return out_dir

    def load(self, sample_id: str) -> dict[str, np.ndarray]:
        """저장된 npy 파일을 로드해 딕셔너리로 반환한다."""
        out_dir = self.output_root / sample_id
        result: dict[str, np.ndarray] = {}
        for npy_file in out_dir.glob("*.npy"):
            result[npy_file.stem] = np.load(npy_file)
        return result
