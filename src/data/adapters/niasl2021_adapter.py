"""NIASL2021 데이터셋 adapter.

출처:
    KoSign Sign Language Translation Project: Introducing The NIASL2021 Dataset
    SLTAT 2022 (ACL Anthology: https://aclanthology.org/2022.sltat-1.9/)

데이터 신청:
    국립국어원 언어정보나눔터 (https://kli.korean.go.kr/)
    또는 논문 저자(NIASL / 국립국어원) 직접 문의

== 예상 폴더 구조 (A-011 가정) ==

Mode A: 분할 매니페스트 방식 (권장)
    <root>/
    ├── train.json
    ├── dev.json
    ├── test.json
    └── videos/
        └── {sample_id}.mp4

Mode B: 파일별 방식
    <root>/
    ├── annotations/
    │   ├── train/
    │   │   └── {sample_id}.json
    │   ├── dev/
    │   └── test/
    └── videos/
        └── {sample_id}.mp4

Mode C: CSV 방식
    <root>/
    ├── train.csv
    ├── dev.csv
    ├── test.csv
    └── videos/

== 어노테이션 JSON 포맷 ==

분할 매니페스트 (train.json) — 리스트:
[
  {
    "id":          "NIASL2021_T_000001",   # 또는 "name", "sample_id"
    "signer":      "P01",                  # 또는 "signer_id"
    "korean_text": "서울 날씨가 맑겠습니다",  # 또는 "text", "korean"
    "gloss":       "서울 날씨 맑다",         # 공백 분리 또는 리스트
    "nms": {                               # 없으면 None
      "head_movement": "nod",
      "eyebrow": "raise"
    },
    "domain":      "weather",              # 없으면 auto-estimated
    "fps":         25,
    "num_frames":  125,
    "start_frame": 0,
    "end_frame":   125
  }
]

파일별 방식: 위와 동일하되 단일 dict.
CSV: 열 이름은 JSON 키 이름과 동일.

== NMS 필드 매핑 ==
NIASL2021 NMS 8종:
  head_movement, eyebrow, cheek, mouth_shape,
  mouth_open, eye, nose, gaze
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from ..schema import DOMAINS, KSLSample, NMSLabels
from .base import BaseAdapter

logger = logging.getLogger(__name__)

# ── 필드명 후보 목록 (앞쪽이 우선) ─────────────────────────────────────────────
_FIELD_CANDIDATES: dict[str, list[str]] = {
    "id":          ["id", "name", "sample_id", "video_id", "uid"],
    "signer":      ["signer", "signer_id", "speaker", "subject", "person"],
    "korean_text": ["korean_text", "text", "korean", "translation", "sentence"],
    "gloss":       ["gloss", "sign", "sign_text", "gloss_sequence", "gloss_text"],
    "domain":      ["domain", "category", "topic", "class"],
    "fps":         ["fps", "frame_rate", "FrameRate"],
    "num_frames":  ["num_frames", "total_frames", "length", "TotalFrame"],
    "start_frame": ["start_frame", "start", "StartFrame"],
    "end_frame":   ["end_frame", "end", "EndFrame"],
    "nms":         ["nms", "facial_expression", "nonmanual", "NMS"],
    "video":       ["video", "video_path", "video_file", "file", "path"],
}

# NIASL2021 도메인 → 프로젝트 도메인 매핑
_DOMAIN_MAP: dict[str, str] = {
    "weather": "directions",    # 날씨예보 → 길안내(가장 근접)
    "emergency": "help",        # 긴급재난 → 도움요청
    "alert": "help",
    "forecast": "directions",
    "daily": "unknown",
    "general": "unknown",
}

# NMS 필드명 → NMSLabels 매핑
_NMS_FIELD_MAP: dict[str, str] = {
    "head_movement": "head_nod",       # "nod" → True / "shake" → head_shake
    "head_nod":      "head_nod",
    "head_shake":    "head_shake",
    "head_tilt":     "head_tilt",
    "eyebrow":       "eyebrow_raise",  # "raise"/"furrow" 추가 분기
    "eyebrow_raise": "eyebrow_raise",
    "eyebrow_furrow":"eyebrow_furrow",
    "eye":           "eye_wide",
    "eye_wide":      "eye_wide",
    "eye_squint":    "eye_squint",
    "mouth_open":    "mouth_open",
    "mouth_shape":   "mouth_shape",    # str value
    "mouth":         "mouth_open",
    "cheek":         "cheek_puff",
    "cheek_puff":    "cheek_puff",
    "nose":          "nose_wrinkle",
    "nose_wrinkle":  "nose_wrinkle",
    "gaze":          "gaze_direction", # str value
    "gaze_direction":"gaze_direction",
}


def _first(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _parse_nms(raw: Any) -> NMSLabels | None:
    """NIASL2021 NMS 딕셔너리 → NMSLabels 변환."""
    if not raw or not isinstance(raw, dict):
        return None
    kwargs: dict[str, Any] = {}
    for src_key, raw_val in raw.items():
        tgt = _NMS_FIELD_MAP.get(src_key.lower())
        if tgt is None:
            continue
        if tgt in ("mouth_shape", "gaze_direction"):
            kwargs[tgt] = str(raw_val) if raw_val else None
        elif isinstance(raw_val, bool):
            kwargs[tgt] = raw_val
        elif isinstance(raw_val, str):
            val_lower = raw_val.lower()
            if src_key.lower() in ("head_movement",):
                if "nod" in val_lower:
                    kwargs["head_nod"] = True
                elif "shake" in val_lower:
                    kwargs["head_shake"] = True
                elif "tilt" in val_lower:
                    kwargs["head_tilt"] = True
            elif src_key.lower() in ("eyebrow",):
                if "raise" in val_lower:
                    kwargs["eyebrow_raise"] = True
                elif "furrow" in val_lower or "frown" in val_lower:
                    kwargs["eyebrow_furrow"] = True
            elif src_key.lower() in ("mouth",):
                if "open" in val_lower:
                    kwargs["mouth_open"] = True
            else:
                positive = val_lower not in ("none", "neutral", "false", "no", "0")
                kwargs[tgt] = positive if positive else None
    try:
        return NMSLabels(**{k: v for k, v in kwargs.items()
                            if k in NMSLabels.__dataclass_fields__})
    except Exception:
        return None


def _map_domain(raw_domain: str | None) -> tuple[str, str]:
    """원본 도메인 → (프로젝트 도메인, intent_source) 반환."""
    if raw_domain is None:
        return "unknown", "auto_estimated"
    key = raw_domain.strip().lower()
    if key in DOMAINS:
        return key, "gold"
    mapped = _DOMAIN_MAP.get(key, "unknown")
    return mapped, "auto_estimated"


def _parse_gloss(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(g).strip() for g in raw if str(g).strip()]
    if isinstance(raw, str):
        tokens = raw.strip().split()
        return tokens if tokens else None
    return None


def _record_to_sample(
    rec: dict[str, Any],
    root: Path,
    dataset_name: str,
    split_label: str,
    idx: int,
) -> KSLSample | None:
    """단일 레코드 딕셔너리 → KSLSample 변환. 필수 필드 누락 시 None 반환."""
    sample_id = _first(rec, _FIELD_CANDIDATES["id"])
    if sample_id is None:
        sample_id = f"{dataset_name}_{split_label}_{idx:06d}"

    signer_raw = _first(rec, _FIELD_CANDIDATES["signer"])
    if signer_raw is None:
        logger.warning(f"[{dataset_name}] signer 필드 없음: {sample_id}")
        signer_raw = "UNKNOWN"
    signer_id = str(signer_raw).strip()

    korean_text = _first(rec, _FIELD_CANDIDATES["korean_text"])
    if not korean_text:
        logger.debug(f"[{dataset_name}] korean_text 없음, 건너뜀: {sample_id}")
        return None
    korean_text = str(korean_text).strip()

    # 비디오 경로 (상대경로로 변환)
    video_raw = _first(rec, _FIELD_CANDIDATES["video"])
    if video_raw:
        vpath = Path(str(video_raw))
        if not vpath.is_absolute():
            video_path = str(vpath)
        else:
            try:
                video_path = str(vpath.relative_to(root))
            except ValueError:
                video_path = vpath.name
    else:
        # 비디오 경로 없음 → ID 기반 추정
        video_path = f"videos/{sample_id}.mp4"

    fps_raw = _first(rec, _FIELD_CANDIDATES["fps"])
    fps = float(fps_raw) if fps_raw is not None else 25.0

    num_frames_raw = _first(rec, _FIELD_CANDIDATES["num_frames"])
    if num_frames_raw is not None:
        num_frames = int(num_frames_raw)
    else:
        end = _first(rec, _FIELD_CANDIDATES["end_frame"])
        start = _first(rec, _FIELD_CANDIDATES["start_frame"], 0)
        if end is not None:
            num_frames = int(end) - int(start)
        else:
            num_frames = 0
            logger.debug(f"[{dataset_name}] num_frames 추정 불가: {sample_id}")

    raw_domain = _first(rec, _FIELD_CANDIDATES["domain"])
    domain, intent_source = _map_domain(str(raw_domain) if raw_domain else None)

    gloss_tokens = _parse_gloss(_first(rec, _FIELD_CANDIDATES["gloss"]))
    nms_labels = _parse_nms(_first(rec, _FIELD_CANDIDATES["nms"]))

    start_frame = _first(rec, _FIELD_CANDIDATES["start_frame"], 0)
    end_frame = _first(rec, _FIELD_CANDIDATES["end_frame"])
    try:
        span_start = int(float(start_frame or 0))
    except (TypeError, ValueError):
        span_start = 0
    try:
        span_end = int(float(end_frame if end_frame is not None else num_frames))
    except (TypeError, ValueError):
        span_end = int(num_frames)

    return KSLSample(
        sample_id=str(sample_id),
        dataset_name=dataset_name,
        domain=domain,
        scenario_id=f"niasl_{raw_domain or 'unknown'}",
        turn_id=0,
        utterance_id=str(sample_id),
        signer_id=signer_id,
        split_group="train",        # split generator가 덮어씀
        video_path=video_path,
        fps=fps,
        num_frames=num_frames,
        korean_text=korean_text,
        gloss_tokens=gloss_tokens,
        nms_labels=nms_labels,
        intent=domain,
        intent_source=intent_source,
        quality_flags=[] if num_frames > 0 else ["missing_num_frames"],
        has_face=True,
        has_hands=True,
        metadata={
            "original_id": str(sample_id),
            "original_domain": raw_domain,
            "split_label": split_label,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "annotation_spans": [
                {
                    "gloss": " ".join(gloss_tokens or []),
                    "start_frame": span_start,
                    "end_frame": span_end,
                }
            ],
        },
        source_annotation_path=None,
    )


class NIASL2021Adapter(BaseAdapter):
    """NIASL2021 한국수어 번역 데이터셋 adapter.

    Args:
        root: 데이터셋 루트 디렉터리
        splits: 로드할 split 목록. 기본값 ["train", "dev", "test"]
        config: 선택적 설정 딕셔너리.
            - field_map: {내부필드명: 실제필드명} 재정의
            - domain_map: {원본도메인: 프로젝트도메인} 재정의
            - video_dir: 비디오 서브디렉터리 이름 (기본 "videos")

    데이터가 없을 때:
        iter_samples()가 빈 이터레이터를 반환하고 경고를 로그에 남긴다.
        먼저 docs/data_guide.md를 참고해 데이터를 준비한다.
    """

    SPLIT_ALIASES: dict[str, list[str]] = {
        "train": ["train", "training", "Train"],
        "dev":   ["dev", "valid", "val", "Dev", "Valid"],
        "test":  ["test", "Test", "eval"],
    }

    def __init__(
        self,
        root: str | Path,
        splits: list[str] | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(root, config)
        self.splits = splits or ["train", "dev", "test"]

    @property
    def dataset_name(self) -> str:
        return "niasl2021"

    def iter_samples(self) -> Iterator[KSLSample]:
        if not self.root.exists():
            logger.warning(
                f"[NIASL2021] 루트 디렉터리 없음: {self.root}\n"
                "데이터 신청: https://kli.korean.go.kr/\n"
                "예상 구조: docs/data_guide.md 참조"
            )
            return

        mode = self._detect_mode()
        logger.info(f"[NIASL2021] 감지된 포맷 모드: {mode}")

        count = 0
        for split in self.splits:
            for rec, split_label in self._load_split(mode, split):
                sample = _record_to_sample(rec, self.root, self.dataset_name, split_label, count)
                if sample is not None:
                    count += 1
                    yield sample

        if count == 0:
            logger.warning(f"[NIASL2021] 로드된 샘플이 없음. 포맷을 확인하세요: docs/data_guide.md")
        else:
            logger.info(f"[NIASL2021] {count}개 샘플 로드 완료")

    # ── 내부 메서드 ────────────────────────────────────────────────────────────

    def _detect_mode(self) -> str:
        """데이터 구조를 자동 감지한다."""
        # Mode A: 분할 매니페스트 JSON
        for split in self.splits:
            for alias in self.SPLIT_ALIASES.get(split, [split]):
                if (self.root / f"{alias}.json").exists():
                    return "manifest_json"
        # Mode C: CSV
        for split in self.splits:
            for alias in self.SPLIT_ALIASES.get(split, [split]):
                if (self.root / f"{alias}.csv").exists():
                    return "manifest_csv"
        # Mode B: 파일별 JSON
        annot_dir = self.root / "annotations"
        if annot_dir.exists():
            return "per_file"
        # fallback: 재귀 탐색
        return "per_file"

    def _load_split(
        self, mode: str, split: str
    ) -> Iterator[tuple[dict[str, Any], str]]:
        aliases = self.SPLIT_ALIASES.get(split, [split])

        if mode == "manifest_json":
            for alias in aliases:
                path = self.root / f"{alias}.json"
                if not path.exists():
                    continue
                raw = self._load_json(path)
                if raw is None:
                    continue
                records = raw if isinstance(raw, list) else raw.get("data", raw.get("samples", []))
                for rec in records:
                    yield rec, split
                return
            logger.warning(f"[NIASL2021] {split} 매니페스트 없음 ({aliases})")

        elif mode == "manifest_csv":
            for alias in aliases:
                path = self.root / f"{alias}.csv"
                if not path.exists():
                    continue
                yield from ((row, split) for row in self._load_csv(path))
                return

        else:  # per_file
            for alias in aliases:
                dirs = [
                    self.root / "annotations" / alias,
                    self.root / alias,
                ]
                for d in dirs:
                    if not d.exists():
                        continue
                    for jpath in sorted(d.rglob("*.json")):
                        raw = self._load_json(jpath)
                        if raw and isinstance(raw, dict):
                            yield raw, split
                    return

    def _load_json(self, path: Path) -> Any:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[NIASL2021] JSON 로드 실패: {path} — {e}")
            return None

    def _load_csv(self, path: Path) -> list[dict[str, Any]]:
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        except Exception as e:
            logger.error(f"[NIASL2021] CSV 로드 실패: {path} — {e}")
            return []
