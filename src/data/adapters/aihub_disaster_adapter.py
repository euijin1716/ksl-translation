"""AI Hub 재난안전 수어 영상 데이터셋 adapter.

출처:
    AI Hub — 재난안전 수어 영상 (또는 "한국수어 영상(재난안전)")
    검색: https://www.aihub.or.kr 에서 "재난 수어" 검색

데이터 신청:
    https://www.aihub.or.kr 회원가입 후 신청
    일반 수어 영상 데이터와 별도 데이터셋으로 관리될 수 있음

== 예상 폴더 구조 (A-014 가정) ==

AI Hub 일반 수어 영상과 동일한 폴더 구조를 따르되,
카테고리가 재난·안전 관련 어휘로 구성된다.

    <root>/
    ├── Training/
    │   ├── 원천데이터/
    │   │   └── {disaster_category}/
    │   │       └── {SignerID}_{word}_{take}.mp4
    │   └── 라벨링데이터/
    │       └── {disaster_category}/
    │           └── {SignerID}_{word}_{take}.json
    └── Validation/
        ├── 원천데이터/
        └── 라벨링데이터/

== 어노테이션 JSON 포맷 ==

AI Hub 일반 수어 영상과 동일한 구조. 재난 관련 차이점:
    - Category: "화재", "지진", "홍수", "태풍", "대피", "구조" 등
    - NMS 어노테이션이 포함될 수 있음 (표정/감정 포함 가능성)

    {
      "DataInfo": {
        "VideoName":    "P001_화재_대피_001.mp4",
        "FrameRate":    30.0,
        "TotalFrame":   120,
        "SignerID":     "P001",
        "Gender":       "M",
        "Category":     "화재",
        "SubCategory":  "대피",
        "KoreanText":   "불이 났습니다. 빨리 대피하세요."
      },
      "Annotation": [
        {
          "SignGloss":   "불",
          "StartFrame":  5,
          "EndFrame":    30
        },
        {
          "SignGloss":   "대피",
          "StartFrame":  35,
          "EndFrame":    80
        }
      ],
      "NMS": {                   # 있는 경우만
        "facial_expression": "urgent",
        "head_movement": "shake"
      }
    }

== 재난 카테고리 → 프로젝트 도메인 ==
재난안전 카테고리는 대부분 "help" 또는 "public" 도메인으로 매핑된다.
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
    _first,
)

logger = logging.getLogger(__name__)

# ── 재난안전 카테고리 → 프로젝트 도메인 매핑 ───────────────────────────────────
_DEFAULT_DISASTER_DOMAIN_MAP: dict[str, str] = {
    # 긴급/도움 요청
    "화재": "help", "지진": "help", "홍수": "help", "태풍": "help",
    "폭설": "help", "폭염": "help", "한파": "help", "대피": "help",
    "구조": "help", "긴급": "help", "재난": "help", "응급": "help",
    "fire": "help", "earthquake": "help", "flood": "help", "typhoon": "help",
    "evacuation": "help", "rescue": "help", "emergency": "help",
    # 공공 안전/민원
    "안전": "public", "예방": "public", "신고": "public", "경보": "public",
    "safety": "public", "prevention": "public", "alert": "public",
    # 병원/의료 관련
    "부상": "hospital", "치료": "hospital", "의료": "hospital",
    "injury": "hospital", "medical": "hospital",
}

# NMS 감정/표현 필드 후보
_NMS_CANDIDATES: list[str] = [
    "NMS", "nms", "NonManualSignal", "FacialExpression",
    "facial_expression", "expression", "Expression",
]

# 감정 표현 → NMSLabels 매핑
_EXPRESSION_NMS_MAP: dict[str, dict[str, Any]] = {
    "urgent": {"eyebrow_furrow": True, "mouth_open": True},
    "긴급": {"eyebrow_furrow": True, "mouth_open": True},
    "fear": {"eyebrow_raise": True, "eye_wide": True},
    "공포": {"eyebrow_raise": True, "eye_wide": True},
    "surprise": {"eyebrow_raise": True, "eye_wide": True, "mouth_open": True},
    "놀람": {"eyebrow_raise": True, "eye_wide": True, "mouth_open": True},
    "worry": {"eyebrow_furrow": True},
    "걱정": {"eyebrow_furrow": True},
}


def _parse_disaster_nms(raw: dict[str, Any]) -> NMSLabels | None:
    """재난안전 데이터의 NMS 딕셔너리 → NMSLabels 변환."""
    nms_raw = _first(raw, _NMS_CANDIDATES)
    if not nms_raw:
        return None

    kwargs: dict[str, Any] = {}

    if isinstance(nms_raw, dict):
        # head_movement
        hm = nms_raw.get("head_movement", "")
        if isinstance(hm, str):
            hm_lower = hm.lower()
            if "nod" in hm_lower:
                kwargs["head_nod"] = True
            elif "shake" in hm_lower:
                kwargs["head_shake"] = True
            elif "tilt" in hm_lower:
                kwargs["head_tilt"] = True

        # eyebrow
        eb = nms_raw.get("eyebrow", nms_raw.get("eyebrows", ""))
        if isinstance(eb, str):
            eb_lower = eb.lower()
            if "raise" in eb_lower:
                kwargs["eyebrow_raise"] = True
            elif "furrow" in eb_lower or "frown" in eb_lower:
                kwargs["eyebrow_furrow"] = True

        # facial_expression (감정 기반)
        expr = nms_raw.get("facial_expression", nms_raw.get("expression", ""))
        if isinstance(expr, str):
            expr_lower = expr.lower().strip()
            nms_extra = _EXPRESSION_NMS_MAP.get(expr_lower, {})
            kwargs.update(nms_extra)

        # mouth
        mouth = nms_raw.get("mouth", nms_raw.get("mouth_shape", ""))
        if isinstance(mouth, str) and mouth.strip():
            mouth_lower = mouth.lower()
            if "open" in mouth_lower:
                kwargs["mouth_open"] = True
            else:
                kwargs["mouth_shape"] = mouth.strip()

    valid_fields = NMSLabels.__dataclass_fields__
    try:
        return NMSLabels(**{k: v for k, v in kwargs.items() if k in valid_fields})
    except Exception:
        return None


def _map_disaster_category(
    category: str | None,
    sub_category: str | None,
    custom_map: dict[str, str],
) -> tuple[str, str]:
    """재난 카테고리 → (프로젝트 도메인, intent_source)"""
    for cat in [category, sub_category]:
        if not cat:
            continue
        key = str(cat).strip().lower()
        if key in DOMAINS:
            return key, "gold"
        mapped = custom_map.get(key)
        if mapped:
            return mapped, "auto_estimated"
        for cat_key, domain in custom_map.items():
            if cat_key in key or key in cat_key:
                return domain, "auto_estimated"
    return "help", "auto_estimated"   # 재난 데이터 기본값 = help


def _extract_gloss(annots: list[dict[str, Any]]) -> list[str] | None:
    glosses = [str(a["gloss"]).strip() for a in annots if a.get("gloss")]
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


class AIHubDisasterAdapter(BaseAdapter):
    """AI Hub 재난안전 수어 영상 데이터셋 adapter.

    Args:
        root: 데이터셋 루트 디렉터리
        config: 선택적 설정 딕셔너리.
            - disaster_domain_map: {카테고리: 도메인} 재정의
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
        self._domain_map: dict[str, str] = {
            **_DEFAULT_DISASTER_DOMAIN_MAP,
            **(self.config.get("disaster_domain_map", {})),
        }
        self._error_limit: int = self.config.get("error_limit", 100)
        self._skip_missing_video: bool = self.config.get("skip_missing_video", False)

    @property
    def dataset_name(self) -> str:
        return "aihub_disaster"

    def iter_samples(self) -> Iterator[KSLSample]:
        if not self.root.exists():
            logger.warning(
                f"[AIHubDisaster] 루트 디렉터리 없음: {self.root}\n"
                "데이터 신청: https://www.aihub.or.kr (\"재난 수어\" 검색)\n"
                "예상 구조: docs/data_guide.md 참조"
            )
            return

        pairs = find_label_video_pairs(self.root)
        if not pairs:
            logger.warning(f"[AIHubDisaster] 라벨-영상 쌍을 찾을 수 없음: {self.root}")
            return

        count = 0
        errors = 0
        skipped_video = 0
        for label_path, video_path in pairs:
            if video_path is None and self._skip_missing_video:
                skipped_video += 1
                logger.debug(f"[AIHubDisaster] 영상 매칭 실패, 건너뜀: {label_path.name}")
                continue

            raw = load_json_safe(label_path)
            if raw is None:
                errors += 1
                if errors >= self._error_limit:
                    logger.error(f"[AIHubDisaster] 파싱 에러 {errors}회 초과, 중단")
                    break
                continue

            sample = self._parse_sample(raw, label_path, video_path)
            if sample is not None:
                count += 1
                yield sample

        if count == 0:
            logger.warning("[AIHubDisaster] 로드된 샘플이 없음. docs/data_guide.md 참조")
        else:
            logger.info(
                f"[AIHubDisaster] {count}개 샘플 로드 완료 "
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
            logger.debug(f"[AIHubDisaster] korean_text 없음, 건너뜀: {label_path.name}")
            return None
        korean_text = str(korean_text).strip()

        signer_id = str(info.get("signer_id") or "UNKNOWN").strip()

        # ── 비디오 경로 ──
        if video_path is not None:
            vpath_str = relative_path(video_path, self.root)
        else:
            label_rel = relative_path(label_path, self.root)
            vpath_str = label_rel.replace("라벨링데이터", "원천데이터").replace(".json", ".mp4")
            logger.debug(f"[AIHubDisaster] 영상 없음, 경로 추정: {vpath_str}")

        fps = float(info.get("fps") or 30.0)
        num_frames = int(info.get("total_frame") or 0)

        # ── 재난 카테고리 → 도메인 ──
        category = info.get("category")
        # SubCategory 지원 (재난안전 데이터는 세부 분류가 있을 수 있음)
        data_info_raw = raw.get("DataInfo") or raw.get("data_info") or raw
        sub_category = (
            data_info_raw.get("SubCategory")
            or data_info_raw.get("sub_category")
            or data_info_raw.get("SubClass")
        )
        domain, intent_source = _map_disaster_category(
            category, sub_category, self._domain_map
        )

        gloss_tokens = _extract_gloss(annots)
        nms_labels = _parse_disaster_nms(raw)   # NMS 있는 경우 추출

        sample_id = (
            Path(info.get("video_name") or label_path.name)
            .stem
            .replace(" ", "_")
        )

        flags: list[str] = []
        if num_frames == 0:
            flags.append("missing_num_frames")
        if gloss_tokens is None:
            flags.append("no_gloss")

        scenario = "_".join(filter(None, [
            "disaster",
            str(category or "").lower().replace(" ", "_") or None,
            str(sub_category or "").lower().replace(" ", "_") or None,
        ]))

        return KSLSample(
            sample_id=f"aihub_disaster_{sample_id}",
            dataset_name=self.dataset_name,
            domain=domain,
            scenario_id=scenario,
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
                "sub_category": sub_category,
                "gender": info.get("gender"),
                "label_file": str(label_path.name),
                "annotation_count": len(annots),
                "annotation_spans": _extract_annotation_spans(annots),
                "has_nms": nms_labels is not None,
            },
            source_annotation_path=relative_path(label_path, self.root),
        )
