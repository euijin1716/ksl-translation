#!/usr/bin/env python3
"""Merge preprocessed chunk manifests and rebuild a global signer split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Chunk manifest JSONL paths or glob patterns.")
    parser.add_argument("--output_dir", default="data/manifests")
    parser.add_argument("--dataset_name", default="processed_chunks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        paths.extend(path for path in matched if path.exists())
    deduped = sorted({path.resolve(): path for path in paths}.values())
    if not deduped:
        raise FileNotFoundError(f"No input manifests matched: {patterns}")
    return deduped


def main() -> None:
    args = parse_args()

    from src.data.manifest import read_manifest, write_manifest
    from src.data.splits import check_signer_leakage, make_signer_independent_split

    merged = {}
    for path in expand_inputs(args.inputs):
        for sample in read_manifest(path):
            old = merged.get(sample.sample_id)
            if old is None:
                merged[sample.sample_id] = sample
                continue
            old_ready = bool(old.keypoint_path)
            new_ready = bool(sample.keypoint_path)
            if new_ready and not old_ready:
                merged[sample.sample_id] = sample
            elif new_ready == old_ready:
                merged[sample.sample_id] = sample

    samples = list(merged.values())
    split_samples, split_manifest = make_signer_independent_split(
        samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )

    problems = check_signer_leakage(split_samples)
    if problems:
        raise RuntimeError("Signer leakage detected:\n" + "\n".join(problems))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(split_samples, output_dir / "all.jsonl")
    for split_group in ("train", "valid", "test"):
        subset = [s for s in split_samples if s.split_group == split_group]
        write_manifest(subset, output_dir / f"{split_group}.jsonl")

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest.__dict__, f, ensure_ascii=False, indent=2)

    counts = {g: sum(1 for s in split_samples if s.split_group == g) for g in ("train", "valid", "test")}
    processed = sum(1 for s in split_samples if s.keypoint_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_samples": len(split_samples),
                "num_with_keypoints": processed,
                "split_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
