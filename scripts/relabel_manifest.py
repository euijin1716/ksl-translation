#!/usr/bin/env python3
"""기존 매니페스트의 boundary 라벨(metadata.annotation_spans)을 라벨 JSON에서 재계산해 제자리 갱신한다.

배경:
    boundary spans는 build_manifest(어댑터) 시점에만 계산돼 매니페스트에 구워진다. merge는
    chunk를 union만 하므로, 어댑터를 고쳐도 기존 chunk의 옛 라벨(빈 spans)은 그대로다.
    이 스크립트는 보존된 라벨 JSON을 source_annotation_path로 다시 읽어 spans만 재계산해
    제자리 갱신한다. 영상 불필요, keypoint_path/split_group/gloss/nms 등은 건드리지 않는다.
    어댑터의 _extract_sign_script_spans를 그대로 재사용(로직 단일 소스).

사용:
    # 1) 먼저 dry-run으로 확인 (쓰지 않음)
    python scripts/relabel_manifest.py --manifest "data/manifests/chunk_*.jsonl" --dry-run --limit 20
    # 2) 실제 적용 (chunk 제자리 갱신)
    python scripts/relabel_manifest.py --manifest "data/manifests/chunk_*.jsonl"
    # 3) 재머지로 all/train/valid/test 재생성
    python scripts/merge_processed_manifests.py --inputs "data/manifests/chunk_*.jsonl"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="매니페스트 boundary spans 재라벨 (라벨 JSON에서 재계산)")
    p.add_argument("--manifest", nargs="+", required=True, help="대상 매니페스트 경로 또는 glob")
    p.add_argument("--raw-root", default="data/raw", help="라벨 JSON 루트 (기본 data/raw)")
    p.add_argument("--dry-run", action="store_true", help="쓰지 않고 통계만 출력")
    p.add_argument("--limit", type=int, default=0, help="처음 N개만 처리 (검증용; --dry-run 전용, 0=전체)")
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out += sorted(Path().glob(pat)) if any(c in pat for c in "*?[]") else [Path(pat)]
    # 중복 제거(순서 유지)
    seen, uniq = set(), []
    for p in out:
        if p.exists() and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_label(sap: str | None, raw_root: Path) -> Path | None:
    """source_annotation_path(백슬래시 가능)를 실제 라벨 JSON 경로로 해석한다."""
    if not sap:
        return None
    rel = str(sap).replace("\\", "/")
    for c in (
        Path(rel),
        raw_root / rel,
        Path(rel.replace("data/raw/", "")),
        raw_root / Path(rel).name,
    ):
        if c.exists():
            return c
    hits = list(raw_root.rglob(Path(rel).name))   # 최후: 파일명 재탐색
    return hits[0] if hits else None


def _meta_fps(raw: dict) -> float | None:
    md = raw.get("metadata") or raw.get("DataInfo") or {}
    v = md.get("video_fps") or md.get("FrameRate") or md.get("fps") if isinstance(md, dict) else None
    try:
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()

    if args.limit and not args.dry_run:
        print("❌ --limit는 매니페스트를 잘라 쓰게 되므로 --dry-run과 함께만 사용하세요.")
        return 1

    from src.data.adapters._aihub_utils import load_json_safe
    from src.data.adapters.aihub_sign_adapter import _extract_sign_script_spans
    from src.data.manifest import read_manifest, write_manifest

    raw_root = Path(args.raw_root)
    manifests = expand(args.manifest)
    if not manifests:
        print(f"❌ 대상 매니페스트 없음: {args.manifest}")
        return 1

    rc = 0
    for mpath in manifests:
        samples = list(read_manifest(mpath))
        if args.limit:
            samples = samples[: args.limit]
        total = len(samples)
        not_found = was_empty = now_filled = unchanged = 0

        for i, s in enumerate(samples, 1):
            if not (s.metadata or {}).get("annotation_spans"):
                was_empty += 1
            jp = resolve_label(s.source_annotation_path, raw_root)
            if jp is None:
                not_found += 1
                continue
            raw = load_json_safe(jp)
            if raw is None:
                not_found += 1
                continue
            fps = s.fps or _meta_fps(raw) or 30.0
            spans = _extract_sign_script_spans(raw, float(fps))
            if s.metadata is None:
                s.metadata = {}
            s.metadata["annotation_spans"] = spans
            if spans:
                now_filled += 1
            else:
                unchanged += 1
            if args.progress_every and i % args.progress_every == 0:
                print(f"  {mpath.name}: {i}/{total} (채움 {now_filled}, 못찾음 {not_found})")

        if not args.dry_run:
            tmp = mpath.with_suffix(mpath.suffix + ".tmp")
            write_manifest(samples, tmp)
            os.replace(tmp, mpath)   # 원자적 교체

        tag = "  (dry-run, 미적용)" if args.dry_run else "  ✅ 적용됨"
        print(
            f"[{mpath.name}] 총 {total} | 기존 빈 spans {was_empty} | "
            f"재계산 후 채워짐 {now_filled} | spans 여전히 빔 {unchanged} | JSON 못찾음 {not_found}{tag}"
        )
        if not_found:
            rc = 2   # 일부 라벨 JSON 못 찾음 → 비정상 종료코드(보존 점검 필요)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
