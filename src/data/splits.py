"""Signer-independent split 생성기.

같은 signer의 데이터가 train/valid/test에 동시에 포함되지 않도록 보장한다.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import KSLSample, SplitManifest


def make_signer_independent_split(
    samples: Iterable[KSLSample],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    dataset_name: str = "dataset",
) -> tuple[list[KSLSample], SplitManifest]:
    """Signer-independent split을 생성하고 샘플에 split_group을 할당한다.

    Args:
        samples: KSLSample 이터러블
        train_ratio: 학습 비율 (남은 비율은 test)
        valid_ratio: 검증 비율
        seed: 재현성을 위한 난수 시드
        dataset_name: manifest에 기록할 데이터셋 이름

    Returns:
        (split이 할당된 샘플 리스트, SplitManifest)

    Raises:
        ValueError: train_ratio + valid_ratio >= 1.0
    """
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.0")

    sample_list = list(samples)

    # signer → samples 매핑
    signer_to_samples: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sample_list):
        signer_to_samples[s.signer_id].append(i)

    signers = sorted(signer_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(signers)

    n = len(signers)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_train = max(1, n_train)
    n_valid = max(1, n_valid)
    # test는 나머지 (최소 1명)
    n_test = max(1, n - n_train - n_valid)
    # 합이 n을 초과하지 않도록 조정
    if n_train + n_valid + n_test > n:
        n_train = n - n_valid - n_test

    train_signers = set(signers[:n_train])
    valid_signers = set(signers[n_train : n_train + n_valid])
    test_signers = set(signers[n_train + n_valid : n_train + n_valid + n_test])

    split_map: dict[str, str] = {}
    for s in train_signers:
        split_map[s] = "train"
    for s in valid_signers:
        split_map[s] = "valid"
    for s in test_signers:
        split_map[s] = "test"

    # 남은 signer (signers가 더 많을 경우) → train에 포함
    for s in signers[n_train + n_valid + n_test :]:
        split_map[s] = "train"

    # 샘플에 split_group 할당
    split_counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    for sample in sample_list:
        group = split_map.get(sample.signer_id, "train")
        object.__setattr__(sample, "split_group", group) if hasattr(sample, "__dataclass_fields__") else None
        # dataclass는 mutable이므로 직접 대입
        sample.split_group = group
        split_counts[group] += 1

    manifest = SplitManifest(
        version="1.0",
        dataset_name=dataset_name,
        split_seed=seed,
        signer_split={
            "train": sorted(train_signers),
            "valid": sorted(valid_signers),
            "test": sorted(test_signers),
        },
        sample_counts=split_counts,
    )
    manifest.validate()

    return sample_list, manifest


def save_manifest(manifest: SplitManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.__dict__, f, ensure_ascii=False, indent=2)


def load_manifest(path: str | Path) -> SplitManifest:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return SplitManifest(**d)


def check_signer_leakage(samples: list[KSLSample]) -> list[str]:
    """train/valid/test 간 signer 누수를 검사하고 문제 signer 목록을 반환한다."""
    split_signers: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        split_signers[s.split_group].add(s.signer_id)

    problems: list[str] = []
    train = split_signers.get("train", set())
    valid = split_signers.get("valid", set())
    test = split_signers.get("test", set())
    for signer in train & valid:
        problems.append(f"[train∩valid] {signer}")
    for signer in train & test:
        problems.append(f"[train∩test] {signer}")
    for signer in valid & test:
        problems.append(f"[valid∩test] {signer}")
    return problems
