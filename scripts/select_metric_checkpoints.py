#!/usr/bin/env python3
"""Evaluate a checkpoint directory and copy metric-best checkpoints.

This is intentionally separate from the training loop because translation
metrics require a tokenizer, vocabulary, manifest, and autoregressive decoding.
Run it after or during training to promote checkpoints by task metrics instead
of relying only on validation loss.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


METRICS = {
    "bleu": "max",
    "chrf": "max",
    "nms_f1": "max",
    "gloss_wer": "min",
    "boundary_f1": "max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints/C")
    parser.add_argument("--config", default="configs/stage_c.yaml")
    parser.add_argument("--manifest", default="data/manifests/valid.jsonl")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gloss_vocab", default="data/manifests/gloss_vocab.json")
    parser.add_argument("--output_dir", default="eval_results/checkpoint_selection")
    parser.add_argument("--draft_mode", default="greedy", choices=["teacher", "greedy"])
    parser.add_argument("--pattern", default="*.pt")
    return parser.parse_args()


def metric_is_better(metric: str, candidate: float, current: float | None) -> bool:
    if current is None:
        return True
    mode = METRICS[metric]
    return candidate > current if mode == "max" else candidate < current


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, output_path: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        args.config,
        "--manifest",
        args.manifest,
        "--split",
        args.split,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--gloss_vocab",
        args.gloss_vocab,
        "--output",
        str(output_path),
        "--draft_mode",
        args.draft_mode,
    ]
    subprocess.run(cmd, check=True)
    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(checkpoint_dir.glob(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir / args.pattern}")

    leaderboard: list[dict] = []
    best: dict[str, dict] = {metric: {"score": None, "checkpoint": None} for metric in METRICS}

    for checkpoint in checkpoints:
        result_path = output_dir / f"{checkpoint.stem}_{args.split}_{args.draft_mode}.json"
        result = evaluate_checkpoint(args, checkpoint, result_path)
        row = {"checkpoint": str(checkpoint), "result_path": str(result_path)}
        row.update({metric: float(result.get(metric, 0.0)) for metric in METRICS})
        leaderboard.append(row)

        for metric in METRICS:
            score = row[metric]
            if metric_is_better(metric, score, best[metric]["score"]):
                best[metric] = {"score": score, "checkpoint": str(checkpoint)}

    promoted: dict[str, str] = {}
    for metric, info in best.items():
        checkpoint = info["checkpoint"]
        if checkpoint is None:
            continue
        dst = checkpoint_dir / f"best_by_{metric}.pt"
        shutil.copy2(checkpoint, dst)
        promoted[metric] = str(dst)

    summary = {
        "split": args.split,
        "draft_mode": args.draft_mode,
        "metrics": METRICS,
        "best": best,
        "promoted": promoted,
        "leaderboard": leaderboard,
    }
    summary_path = output_dir / f"checkpoint_selection_{args.split}_{args.draft_mode}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
