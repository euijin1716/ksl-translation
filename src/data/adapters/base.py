"""데이터셋 adapter 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ..schema import KSLSample


class BaseAdapter(ABC):
    """모든 데이터셋 adapter가 구현해야 하는 인터페이스.

    각 adapter는:
    - 원본 데이터셋 필드를 최대한 보존한다.
    - 필터링은 여기서 하지 않고 DatasetView/sampler에서 수행한다.
    - 절대 경로를 manifest에 직접 저장하지 않는다.
    """

    def __init__(self, root: str | Path, config: dict | None = None) -> None:
        self.root = Path(root)
        self.config = config or {}

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """데이터셋 고유 이름."""
        ...

    @abstractmethod
    def iter_samples(self) -> Iterator[KSLSample]:
        """원본 데이터셋을 KSLSample 스트림으로 반환한다."""
        ...

    def get_signer_ids(self) -> list[str]:
        """데이터셋 내 모든 signer_id 목록을 반환한다."""
        seen: set[str] = set()
        result: list[str] = []
        for sample in self.iter_samples():
            if sample.signer_id not in seen:
                seen.add(sample.signer_id)
                result.append(sample.signer_id)
        return result
