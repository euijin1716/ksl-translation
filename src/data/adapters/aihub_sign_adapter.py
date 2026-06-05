"""AI Hub 한국수어 영상 데이터셋 adapter.

출처:
    AI Hub — 한국수어 영상 (데이터셋 번호 103)
    https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=103

데이터 신청:
    https://www.aihub.or.kr 회원가입 후 신청
    승인 후 aihub-downloader 또는 웹 다운로드

== 예상 폴더 구조 (A-012 가정) ==

    <root>/
    ├── Training/
    │   ├── 원천데이터/
    │   │   └── {category}/
    │   │       └── {SignerID}_{word}_{take}.mp4
    │   └── 라벨링데이터/
    │       └── {category}/
    │           └── {SignerID}_{word}_{take}.json
    └── Validation/
        ├── 원천데이터/
        └── 라벨링데이터/

또는 단순 구조:
    <root>/
    ├── videos/
    └── labels/

== 어노테이션 JSON 포맷 ==

    {
      "DataInfo": {
        "VideoName":    "P001_감사합니다_001.mp4",
        "FrameRate":    30.0,
        "TotalFrame":   90,
        "SignerID":     "P001",
        "Gender":       "F",
        "Category":     "인사",
        "KoreanText":   "감사합니다"
      },
      "Annotation": [
        {
          "SignGloss":   "감사하다",
          "StartFrame":  5,
          "EndFrame":    85,
          "BothHands":   true
        }
      ]
    }

필드명은 AI Hub 다운로드 버전에 따라 다를 수 있다.
_aihub_utils.py의 필드 후보 목록에서 자동 탐색한다.

== 카테고리 → 프로젝트 도메인 매핑 ==
도메인 변환은 category_domain_map config로 재정의 가능하다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from ..schema import DOMAINS, KSLSample, NMSLabels
from .base import BaseAdapter
from ._aihub_utils import (
    find_label_video_pairs,
    load_json_safe,
    parse_annotations,
    parse_data_info,
    relative_path,
)

logger = logging.getLogger(__name__)

# ── 카테고리 → 프로젝트 도메인 매핑 (기본값) ─────────────────────────────────
_DEFAULT_CATEGORY_DOMAIN_MAP: dict[str, str] = {
    # 재난/안전 카테고리
    "자연재난": "help", "사회재난": "help", "기타재난": "help",
    "재난": "help", "안전": "help", "날씨": "help",
    "delugeflood": "help", "coldwave": "help", "weather": "help",
    # 병원/의료
    "병원": "hospital", "의료": "hospital", "건강": "hospital",
    "hospital": "hospital", "medical": "hospital",
    # 길안내
    "길안내": "directions", "교통": "directions", "이동": "directions",
    "directions": "directions", "transportation": "directions",
    # 주문/결제
    "주문": "order", "결제": "order", "음식": "order", "식당": "order",
    "order": "order", "payment": "order",
    # 예약
    "예약": "reservation", "확인": "reservation",
    "reservation": "reservation",
    # 공공
    "민원": "public", "공공": "public", "행정": "public",
    "public": "public", "government": "public",
    # 도움
    "도움": "help", "긴급": "help", "안전": "help",
    "help": "help", "emergency": "help",
    # 기본
    "인사": "unknown", "일상": "unknown", "감정": "unknown",
    "단어": "unknown", "문장": "unknown", "지수": "unknown",
}


def _map_category(category: str | None, custom_map: dict[str, str]) -> tuple[str, str]:
    """카테고리 문자열 → (프로젝트 도메인, intent_source)"""
    if not category:
        return "unknown", "auto_estimated"
    key = str(category).strip().lower()
    if key in DOMAINS:
        return key, "gold"
    # 직접 매핑 시도
    mapped = custom_map.get(key)
    if mapped:
        return mapped, "auto_estimated"
    # 부분 매칭
    for cat_key, domain in custom_map.items():
        if cat_key in key or key in cat_key:
            return domain, "auto_estimated"
    return "unknown", "auto_estimated"


def _has_nms_events(nms_script: dict[str, Any], key: str) -> bool:
    """Return True when an AIHub NMS event list has at least one event."""
    events = nms_script.get(key)
    return isinstance(events, list) and len(events) > 0


def _first_nms_descriptor(nms_script: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty descriptor from one or more NMS event lists."""
    for key in keys:
        events = nms_script.get(key)
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            descriptor = str(event.get("descriptor") or "").strip()
            if descriptor:
                return descriptor
    return None


def _build_nms_from_raw(raw: dict[str, Any]) -> NMSLabels | None:
    """Build coarse NMS labels from AIHub ``nms_script``.

    AIHub stores non-manual signals as time spans under compact keys such as
    EBf, Hno, Hs, Mo1, Mmo, Mctr, Ci, and Tbt. The current model supervises
    NMS as video-level multi-label targets, so each key is converted to
    present/absent labels. Empty lists are treated as annotated negative
    examples, while missing keys are left unmasked.
    """
    nms_script = raw.get("nms_script")
    if not isinstance(nms_script, dict) or not nms_script:
        return None

    known_keys = {"Ci", "Hs", "EBf", "Hno", "Mmo", "Mo1", "Tbt", "Mctr"}
    if not any(key in nms_script for key in known_keys):
        return None

    mouth_shape = _first_nms_descriptor(nms_script, "Mmo", "Mctr")
    if mouth_shape is None and _has_nms_events(nms_script, "Mctr"):
        mouth_shape = "mouth_shape"

    return NMSLabels(
        eyebrow_furrow=_has_nms_events(nms_script, "EBf") if "EBf" in nms_script else None,
        mouth_open=_has_nms_events(nms_script, "Mo1") if "Mo1" in nms_script else None,
        mouth_shape=mouth_shape if mouth_shape is not None else ("" if "Mmo" in nms_script or "Mctr" in nms_script else None),
        cheek_puff=_has_nms_events(nms_script, "Ci") if "Ci" in nms_script else None,
        head_nod=_has_nms_events(nms_script, "Hno") if "Hno" in nms_script else None,
        head_shake=_has_nms_events(nms_script, "Hs") if "Hs" in nms_script else None,
        head_tilt=_has_nms_events(nms_script, "Tbt") if "Tbt" in nms_script else None,
    )


def _extract_gloss(annots: list[dict[str, Any]]) -> list[str] | None:
    """어노테이션 배열에서 gloss 시퀀스를 추출한다."""
    glosses = []
    seen = set()
    for a in annots:
        g = a.get("gloss")
        if g and str(g).strip():
            token = str(g).strip()
            # sign_script의 both/strong/weak 이벤트가 중복될 수 있어 순서를 유지하며 중복 제거
            if token not in seen:
                glosses.append(token)
                seen.add(token)
    return glosses if glosses else None


def _extract_annotation_spans(annots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for annot in annots:
        start = annot.get("start_frame")
        end = annot.get("end_frame")
        if start is None or end is None:
            continue
        try:
            start_i = int(float(start))
            end_i = int(float(end))
        except (TypeError, ValueError):
            continue
        spans.append(
            {
                "gloss": str(annot.get("gloss") or "").strip(),
                "start_frame": start_i,
                "end_frame": end_i,
            }
        )
    return spans


def _extract_sign_script_spans(raw: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    """sign_script의 초 단위 gloss 이벤트를 원본 프레임 구간으로 환산한다.

    AI Hub sign_script은 start/end가 '초'다(예: 0.832~1.344). keypoint_dataset가
    annotation_spans를 원본 프레임 기준으로 해석하므로 start*fps(원본 fps)로 환산한다.
    (이전엔 start_frame 키만 찾아 전부 드롭 → boundary 라벨이 100% 비어 있었다.)
    """
    sign_script = raw.get("sign_script")
    if not isinstance(sign_script, dict) or not fps or fps <= 0:
        return []
    spans: list[dict[str, Any]] = []
    for key in (
        "sign_gestures_both", "sign_gestures_strong", "sign_gestures_weak",
        "sign_gestures_right", "sign_gestures_left",
    ):
        events = sign_script.get(key) or []
        if not isinstance(events, list):
            continue
        for item in events:
            if not isinstance(item, dict):
                continue
            start, end = item.get("start"), item.get("end")
            if start is None or end is None:
                continue
            try:
                sf, ef = int(round(float(start) * fps)), int(round(float(end) * fps))
            except (TypeError, ValueError):
                continue
            if ef < sf:
                sf, ef = ef, sf
            spans.append({
                "gloss": str(item.get("gloss_id") or item.get("gloss") or "").strip(),
                "start_frame": sf,
                "end_frame": ef,
            })
    spans.sort(key=lambda s: (s["start_frame"], s["end_frame"]))
    return spans


class AIHubSignAdapter(BaseAdapter):
    """AI Hub 한국수어 영상 데이터셋 adapter.

    Args:
        root: 데이터셋 루트 디렉터리
        config: 선택적 설정 딕셔너리.
            - category_domain_map: {카테고리: 도메인} 재정의
            - splits: ["Training", "Validation"] 또는 ["train", "val"]
            - error_limit: 파싱 실패를 무시할 최대 수 (기본 100)
            - skip_missing_video: 영상 매칭 실패 샘플 건너뛰기 (기본 False)

    데이터가 없을 때:
        iter_samples()가 빈 이터레이터를 반환하고 경고를 로그에 남긴다.
    """

    def __init__(
        self,
        root: str | Path,
        config: dict | None = None,
    ) -> None:
        super().__init__(root, config)
        self._cat_domain_map: dict[str, str] = {
            **_DEFAULT_CATEGORY_DOMAIN_MAP,
            **(self.config.get("category_domain_map", {})),
        }
        self._error_limit: int = self.config.get("error_limit", 100)
        self._skip_missing_video: bool = self.config.get("skip_missing_video", False)

    @property
    def dataset_name(self) -> str:
        return "aihub_sign"

    def iter_samples(self) -> Iterator[KSLSample]:
        if not self.root.exists():
            logger.warning(
                f"[AIHubSign] 루트 디렉터리 없음: {self.root}\n"
                "데이터 신청: https://www.aihub.or.kr (데이터셋 번호 103)\n"
                "예상 구조: docs/data_guide.md 참조"
            )
            return

        pairs = find_label_video_pairs(self.root)
        if not pairs:
            logger.warning(f"[AIHubSign] 라벨-영상 쌍을 찾을 수 없음: {self.root}")
            return

        count = 0
        errors = 0
        skipped_video = 0
        for label_path, video_path in pairs:
            if video_path is None and self._skip_missing_video:
                skipped_video += 1
                logger.debug(f"[AIHubSign] 영상 매칭 실패, 건너뜀: {label_path.name}")
                continue

            raw = load_json_safe(label_path)
            if raw is None:
                errors += 1
                if errors >= self._error_limit:
                    logger.error(f"[AIHubSign] 파싱 에러 {errors}회 초과, 중단")
                    break
                continue

            sample = self._parse_sample(raw, label_path, video_path)
            if sample is not None:
                count += 1
                yield sample

        if count == 0:
            logger.warning("[AIHubSign] 로드된 샘플이 없음. 포맷을 확인하세요: docs/data_guide.md")
        else:
            logger.info(
                f"[AIHubSign] {count}개 샘플 로드 완료 "
                f"(에러: {errors}, 영상 없음 건너뜀: {skipped_video})"
            )

    # ── 내부 메서드 ─────────────────────────────────────────────────────────────

    def _parse_sample(
        self,
        raw: dict[str, Any],
        label_path: Path,
        video_path: Path | None,
    ) -> KSLSample | None:
        info = parse_data_info(raw)
        annots = parse_annotations(raw)

        # ── 필수 필드 ──
        korean_text = info.get("korean_text")
        if not korean_text:
            logger.debug(f"[AIHubSign] korean_text 없음, 건너뜀: {label_path.name}")
            return None
        korean_text = str(korean_text).strip()

        signer_id = str(info.get("signer_id") or "UNKNOWN").strip()

        # ── 비디오 경로 (상대경로 유지) ──
        if video_path is not None:
            vpath_str = relative_path(video_path, self.root)
        else:
            # 비디오 없음: 라벨 경로 기반 추정
            video_name = info.get("video_name") or label_path.stem
            # 원천데이터 경로 추정
            label_rel = relative_path(label_path, self.root)
            vpath_str = label_rel.replace("라벨링데이터", "원천데이터").replace(".json", ".mp4")
            logger.debug(f"[AIHubSign] 영상 없음, 경로 추정: {vpath_str}")

        fps = float(info.get("fps") or 30.0)
        num_frames = int(info.get("total_frame") or 0)

        category = info.get("category")
        if not category:
            category = self._infer_category_from_path(label_path)
        domain, intent_source = _map_category(category, self._cat_domain_map)

        gloss_tokens = _extract_gloss(annots)
        nms_labels = _build_nms_from_raw(raw)

        # sample_id: 비디오명 기반 (또는 label 파일명)
        sample_id = (
            Path(info.get("video_name") or label_path.name)
            .stem
            .replace(" ", "_")
        )

        # 품질 플래그
        flags: list[str] = []
        if num_frames == 0:
            flags.append("missing_num_frames")
        if gloss_tokens is None:
            flags.append("no_gloss")

        return KSLSample(
            sample_id=f"aihub_sign_{sample_id}",
            dataset_name=self.dataset_name,
            domain=domain,
            scenario_id=f"aihub_{str(category or 'general').lower()}",
            turn_id=0,
            utterance_id=sample_id,
            signer_id=signer_id,
            split_group="train",        # split generator가 덮어씀
            video_path=vpath_str,
            fps=fps,
            num_frames=num_frames,
            korean_text=korean_text,
            gloss_tokens=gloss_tokens,
            nms_labels=nms_labels,
            intent=domain,
            intent_source=intent_source,
            quality_flags=flags,
            has_face=True,
            has_hands=True,
            metadata={
                "original_category": category,
                "gender": info.get("gender"),
                "label_file": str(label_path.name),
                "annotation_count": len(annots),
                "annotation_spans": _extract_sign_script_spans(raw, fps),
                "has_nms": nms_labels is not None,
            },
            source_annotation_path=relative_path(label_path, self.root),
        )

    def _infer_category_from_path(self, label_path: Path) -> str | None:
        """폴더명에서 대략적인 카테고리를 추정한다."""
        parts = list(label_path.parts)
        for marker in ("1.자연재난", "2.사회재난", "4.기타재난", "자연재난", "사회재난", "기타재난"):
            if marker in parts:
                return marker
        # 파일 바로 위 어휘 폴더 예: DELUGEFLOOD, COLDWAVE
        if len(parts) >= 3:
            return parts[-3]
        return None
