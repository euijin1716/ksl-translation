"""공통 데이터 스키마 정의.

모든 데이터셋 adapter는 KSLSample 로 변환해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DOMAINS = frozenset({"hospital", "directions", "order", "reservation", "public", "help", "unknown"})
SPLIT_GROUPS = frozenset({"train", "valid", "test"})
INTENT_SOURCES = frozenset({"gold", "auto_estimated"})


@dataclass
class NMSLabels:
    """비수지신호(Non-Manual Signal) 라벨.

    None은 해당 항목이 어노테이션 되지 않았음을 의미한다.
    """
    eyebrow_raise: bool | None = None
    eyebrow_furrow: bool | None = None
    eye_wide: bool | None = None
    eye_squint: bool | None = None
    nose_wrinkle: bool | None = None
    mouth_open: bool | None = None
    mouth_shape: str | None = None      # 예: "아", "이", "우"
    cheek_puff: bool | None = None
    head_nod: bool | None = None
    head_shake: bool | None = None
    head_tilt: bool | None = None
    gaze_direction: str | None = None   # "forward", "left", "right", "down", "up"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NMSLabels":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class KSLSample:
    """공통 샘플 스키마.

    모든 필수 필드는 기본값이 없다.
    선택 필드는 None 기본값을 가진다.
    """
    # ── 필수 식별자 ─────────────────────────────────────────────────────────
    sample_id: str
    dataset_name: str
    domain: str                          # DOMAINS 내 값
    scenario_id: str
    turn_id: int
    utterance_id: str
    signer_id: str
    split_group: str                     # SPLIT_GROUPS 내 값

    # ── 비디오 정보 ──────────────────────────────────────────────────────────
    video_path: str                      # 프로젝트 root 기준 상대경로
    fps: float
    num_frames: int

    # ── 라벨 ─────────────────────────────────────────────────────────────────
    korean_text: str
    gloss_tokens: list[str] | None
    nms_labels: NMSLabels | None
    intent: str | None
    intent_source: str                   # INTENT_SOURCES 내 값

    # ── 품질/가용성 플래그 ───────────────────────────────────────────────────
    quality_flags: list[str] = field(default_factory=list)
    has_face: bool = True
    has_hands: bool = True

    # ── 원본 메타데이터 ──────────────────────────────────────────────────────
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── 선택 필드 ────────────────────────────────────────────────────────────
    prev_turn_ids: list[str] | None = None
    dialogue_id: str | None = None
    source_annotation_path: str | None = None
    keypoint_path: str | None = None
    crop_index_path: str | None = None

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got '{self.domain}'")
        if self.split_group not in SPLIT_GROUPS:
            raise ValueError(f"split_group must be one of {SPLIT_GROUPS}, got '{self.split_group}'")
        if self.intent_source not in INTENT_SOURCES:
            raise ValueError(f"intent_source must be one of {INTENT_SOURCES}, got '{self.intent_source}'")

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        if self.nms_labels is not None:
            d["nms_labels"] = self.nms_labels.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KSLSample":
        d = d.copy()
        if isinstance(d.get("nms_labels"), dict):
            d["nms_labels"] = NMSLabels.from_dict(d["nms_labels"])
        return cls(**d)


@dataclass
class SplitManifest:
    """Signer-independent split 결과 기록."""

    version: str
    dataset_name: str
    split_seed: int
    signer_split: dict[str, list[str]]   # {"train": [...], "valid": [...], "test": [...]}
    sample_counts: dict[str, int]

    def validate(self) -> None:
        """train/valid/test signer 교집합이 없는지 검사한다."""
        train_set = set(self.signer_split.get("train", []))
        valid_set = set(self.signer_split.get("valid", []))
        test_set = set(self.signer_split.get("test", []))
        overlap_tv = train_set & valid_set
        overlap_tt = train_set & test_set
        overlap_vt = valid_set & test_set
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Signer leakage detected: train∩valid={overlap_tv}, "
                f"train∩test={overlap_tt}, valid∩test={overlap_vt}"
            )
