"""Manifest 작성기.

샘플 목록을 JSONL 또는 JSON 형식으로 저장/로드한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import KSLSample


def write_manifest(samples: list[KSLSample], path: str | Path) -> None:
    """샘플 목록을 JSONL 파일로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> Iterator[KSLSample]:
    """JSONL manifest를 읽어 KSLSample 이터레이터를 반환한다."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield KSLSample.from_dict(json.loads(line))


def filter_by_split(path: str | Path, split_group: str) -> list[KSLSample]:
    """manifest에서 특정 split_group만 필터링해 반환한다."""
    return [s for s in read_manifest(path) if s.split_group == split_group]
