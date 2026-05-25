"""AI Hub 데이터셋 공통 파서 유틸리티.

두 AI Hub adapter(수어 영상, 재난안전)가 공유하는 JSON 파싱 로직을 제공한다.
adapter 간 의존성을 피하기 위해 이 파일에만 위치시킨다.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VIDEO_SUFFIX_PRIORITY = {
    "f": 0,  # 정면(front): 얼굴/손이 가장 잘 보임. 데이터셋 공식 키포인트 주석도 _F 기준.
    "d": 1,  # 이전 기본값이었으나 사선/하향 뷰라 MediaPipe face 검출이 대부분 실패.
    "c": 2,
    "l": 3,
    "r": 4,
    "u": 5,
}

# ── AI Hub 데이터셋의 알려진 필드명 변형 ──────────────────────────────────────
# 각 리스트의 첫 번째 값이 우선순위가 높다.
_DATA_INFO_MAP: dict[str, list[str]] = {
    "video_name": ["VideoName", "video_name", "FileName", "file_name", "Name"],
    "fps":        ["FrameRate", "FPS", "fps", "frame_rate", "FramePerSecond"],
    "total_frame":["TotalFrame", "total_frame", "FrameCount", "NumFrames", "num_frames"],
    "signer_id":  ["SignerID", "signer_id", "SpeakerID", "PersonID", "SubjectID", "subject_id"],
    "korean_text":["KoreanText", "korean_text", "SentenceText", "Text", "Sentence", "text"],
    "category":   ["Category", "category", "Domain", "domain", "Class", "WordCategory"],
    "gender":     ["Gender", "gender"],
    "width":      ["VideoWidth", "Width", "width"],
    "height":     ["VideoHeight", "Height", "height"],
}

_ANNOTATION_MAP: dict[str, list[str]] = {
    "gloss":       ["SignGloss", "sign_gloss", "Gloss", "gloss", "Word", "word", "Sign"],
    "start_frame": ["StartFrame", "start_frame", "Start", "start"],
    "end_frame":   ["EndFrame",   "end_frame",   "End",   "end"],
    "both_hands":  ["BothHands", "both_hands", "TwoHands", "two_hands"],
    "strong_hand": ["StrongHand", "strong_hand", "DominantHand"],
    "weak_hand":   ["WeakHand",   "weak_hand",   "NonDominantHand"],
}


def _first(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """d에서 keys 순서로 첫 번째로 존재하는 값을 반환한다."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def parse_data_info(raw: dict[str, Any]) -> dict[str, Any]:
    """AI Hub JSON의 DataInfo 또는 최상위 딕셔너리에서 메타 필드를 추출한다."""
    # DataInfo 섹션이 있으면 그 안에서, 없으면 raw 전체에서 찾는다
    src = raw.get("DataInfo") or raw.get("data_info") or raw
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    result: dict[str, Any] = {}
    for field, candidates in _DATA_INFO_MAP.items():
        result[field] = _first(src, candidates)

    # 실제 재난/수어 말뭉치 JSON은 DataInfo 대신 metadata + korean_text +
    # sign_script 구조를 사용한다.
    result["video_name"] = result["video_name"] or metadata.get("id")
    result["fps"] = result["fps"] or metadata.get("video_fps")
    result["korean_text"] = result["korean_text"] or raw.get("korean_text")
    signer = metadata.get("signer")
    translator = metadata.get("translator")
    if result["signer_id"] is None and isinstance(signer, dict):
        result["signer_id"] = signer.get("id")
    if result["signer_id"] is None and isinstance(translator, dict):
        result["signer_id"] = translator.get("id")
    return result


def parse_annotations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """AI Hub JSON의 Annotation 배열을 추출해 정규화된 리스트로 반환한다."""
    annot_raw = (
        raw.get("Annotation")
        or raw.get("annotation")
        or raw.get("Annotations")
        or raw.get("annotations")
        or []
    )
    if not annot_raw:
        annot_raw = _parse_sign_script_annotations(raw)
    if isinstance(annot_raw, dict):
        annot_raw = [annot_raw]

    result = []
    for item in annot_raw:
        parsed: dict[str, Any] = {}
        for field, candidates in _ANNOTATION_MAP.items():
            parsed[field] = _first(item, candidates)
        # 원본도 보존
        parsed["_raw"] = item
        result.append(parsed)
    return result


def _parse_sign_script_annotations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """실제 AI Hub sign_script 구조에서 gloss 이벤트를 추출한다."""
    sign_script = raw.get("sign_script")
    if not isinstance(sign_script, dict):
        return []

    result: list[dict[str, Any]] = []
    for key in (
        "sign_gestures_both",
        "sign_gestures_strong",
        "sign_gestures_weak",
        "sign_gestures_right",
        "sign_gestures_left",
    ):
        events = sign_script.get(key) or []
        if not isinstance(events, list):
            continue
        for item in events:
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "gloss": item.get("gloss_id") or item.get("gloss") or item.get("word"),
                    "start_frame": item.get("start_frame"),
                    "end_frame": item.get("end_frame"),
                    "start_time": item.get("start"),
                    "end_time": item.get("end"),
                    "stream": key,
                    "_raw": item,
                }
            )
    return result


def load_json_safe(path: Path) -> dict[str, Any] | None:
    """JSON 파일을 읽어 반환한다. 실패 시 None을 반환하고 에러를 로그에 기록한다."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {path} — {e}")
    except OSError as e:
        logger.error(f"파일 읽기 실패: {path} — {e}")
    return None


def find_label_video_pairs(
    root: Path,
    label_suffixes: tuple[str, ...] = (".json",),
    video_suffixes: tuple[str, ...] = (".mp4", ".avi", ".mov"),
) -> list[tuple[Path, Path | None]]:
    """라벨링데이터 하위의 JSON과 원천데이터 하위의 동명 영상을 매칭한다.

    AI Hub 표준 폴더 구조:
        <root>/Training/라벨링데이터/**/*.json
        <root>/Training/원천데이터/**/*.mp4
        <root>/Validation/라벨링데이터/**/*.json
        <root>/Validation/원천데이터/**/*.mp4

    구조를 인식하지 못하면 root 전체를 재귀 탐색해 JSON을 수집한다.
    """
    pairs: list[tuple[Path, Path | None]] = []

    # ── AI Hub 표준 구조 탐지 ────────────────────────────────────────────────
    label_roots = [
        p for p in root.rglob("라벨링데이터")
        if p.is_dir()
    ]
    if not label_roots:
        # 표준 구조 없음 → 전체 탐색
        label_roots = [root]

    # 영상 디렉터리를 빠르게 찾기 위한 인덱스.
    # AI Hub 영상 파일은 라벨 stem 뒤에 카메라/각도 suffix가 붙는 경우가 있다.
    # 예: label=NIA_SL_G1_COLDWAVE000010_1_TW07.json
    #     video=NIA_SL_G1_COLDWAVE000010_1_TW07_D.mp4
    video_index: dict[str, Path] = {}
    video_prefix_index: dict[str, Path] = {}
    for vsuffix in video_suffixes:
        for vpath in root.rglob(f"*{vsuffix}"):
            stem = vpath.stem.lower()
            video_index[stem] = vpath
            base_stem = _strip_view_suffix(stem)
            if base_stem != stem:
                current = video_prefix_index.get(base_stem)
                if current is None or _view_priority(vpath.stem) < _view_priority(current.stem):
                    video_prefix_index[base_stem] = vpath

    for lroot in label_roots:
        for lsuffix in label_suffixes:
            for lpath in lroot.rglob(f"*{lsuffix}"):
                label_stem = lpath.stem.lower()
                vpath = video_index.get(label_stem) or video_prefix_index.get(label_stem)
                pairs.append((lpath, vpath))

    return pairs


def _strip_view_suffix(stem: str) -> str:
    """Remove a trailing one-letter view suffix from an AI Hub video stem."""
    return re.sub(r"_[a-z]$", "", stem)


def _view_priority(stem: str) -> int:
    """Prefer a stable representative video when several camera views exist."""
    suffix = stem.rsplit("_", 1)[-1].lower()
    return _VIDEO_SUFFIX_PRIORITY.get(suffix, 99)


def relative_path(path: Path, root: Path) -> str:
    """절대 경로를 root 기준 상대경로 문자열로 변환한다.

    변환 실패 시 파일명만 반환한다 (절대경로 manifest 저장 금지).
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
