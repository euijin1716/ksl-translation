#!/usr/bin/env python3
"""Orchestrate Stage C draft generation and LLM-corrected evaluation.

Default order:
1. LLM smoke test on eval_results/pred_new.jsonl
2. Claude/provider health check
3. create full test draft_greedy predictions
4. run LLM correction for the full test predictions
5. score final_text with BLEU/chrF/ROUGE-L
6. compare against stage_c_test_new.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/C/best.pt")
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--manifest", default="data/manifests/test.jsonl")
    p.add_argument("--gloss_vocab", default="data/manifests/gloss_vocab.json")
    p.add_argument("--baseline", default="eval_results/stage_c_test_new.json")
    p.add_argument("--smoke_input", default="eval_results/pred_new.jsonl")
    p.add_argument("--smoke_corrected", default="eval_results/pred_new_llm_corrected_smoke.jsonl")
    p.add_argument("--smoke_metrics", default="eval_results/stage_c_test_new_llm_smoke.json")
    p.add_argument("--full_predictions", default="eval_results/pred_test_full.jsonl")
    p.add_argument("--full_corrected", default="eval_results/pred_test_full_llm_corrected.jsonl")
    p.add_argument("--full_metrics", default="eval_results/stage_c_test_new_llm_corrected.json")
    p.add_argument("--provider", default="claude", choices=["dummy", "claude", "openai"])
    p.add_argument("--llm_config", default="configs/base.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--sleep_seconds", type=float, default=0.0)
    p.add_argument("--max_full_samples", type=int, default=None)
    p.add_argument("--force_predictions", action="store_true")
    p.add_argument("--force_smoke", action="store_true")
    p.add_argument("--resume_llm", action="store_true")
    p.add_argument("--skip_health_check", action="store_true")
    p.add_argument("--skip_smoke", action="store_true")
    p.add_argument("--skip_full_predictions", action="store_true")
    p.add_argument("--skip_full_llm", action="store_true")
    p.add_argument("--allow_bad_smoke", action="store_true", help="Continue even if smoke LLM metrics fail guardrails.")
    p.add_argument("--max_smoke_parse_failures", type=int, default=0)
    p.add_argument("--min_smoke_delta_bleu", type=float, default=0.0)
    p.add_argument("--min_smoke_delta_chrf", type=float, default=0.0)
    p.add_argument("--min_smoke_delta_rouge_l", type=float, default=0.0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run(cmd: list[str], dry_run: bool) -> None:
    display = " ".join(cmd)
    logger.info("$ %s", display)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def py() -> str:
    return sys.executable


def maybe_remove(path: Path, enabled: bool, dry_run: bool) -> None:
    if not enabled or not path.exists():
        return
    logger.info("Removing existing file: %s", path)
    if not dry_run:
        path.unlink()


def llm_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--config",
        args.llm_config,
        "--provider",
        args.provider,
    ]
    if args.model:
        values.extend(["--model", args.model])
    return values


def run_health_check(args: argparse.Namespace) -> None:
    if args.skip_health_check:
        logger.info("Skipping provider health check.")
        return
    cmd = [
        py(),
        "scripts/run_llm_corrected_eval.py",
        "--health_check_only",
        *llm_args(args),
    ]
    run(cmd, args.dry_run)


def run_smoke(args: argparse.Namespace) -> None:
    if args.skip_smoke:
        logger.info("Skipping 20-sample LLM smoke test.")
        return
    smoke_input = ROOT / args.smoke_input
    if not smoke_input.exists():
        raise FileNotFoundError(f"Smoke input not found: {smoke_input}")

    maybe_remove(ROOT / args.smoke_corrected, args.force_smoke, args.dry_run)
    cmd = [
        py(),
        "scripts/run_llm_corrected_eval.py",
        "--input",
        args.smoke_input,
        "--output",
        args.smoke_corrected,
        "--metrics_output",
        args.smoke_metrics,
        "--baseline",
        args.baseline,
        "--max_samples",
        "20",
        *llm_args(args),
    ]
    if args.resume_llm:
        cmd.append("--resume")
    if args.sleep_seconds:
        cmd.extend(["--sleep_seconds", str(args.sleep_seconds)])
    run(cmd, args.dry_run)


def enforce_smoke_gate(args: argparse.Namespace) -> None:
    if args.skip_smoke or args.allow_bad_smoke or args.dry_run:
        return
    metrics_path = ROOT / args.smoke_metrics
    if not metrics_path.exists():
        raise FileNotFoundError(f"Smoke metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    parse_failed = int(metrics.get("parse_failed_count") or 0)
    if parse_failed > args.max_smoke_parse_failures:
        failures.append(
            f"parse_failed_count={parse_failed} > {args.max_smoke_parse_failures}"
        )
    checks = [
        ("delta_bleu", args.min_smoke_delta_bleu),
        ("delta_chrf", args.min_smoke_delta_chrf),
        ("delta_rouge_l", args.min_smoke_delta_rouge_l),
    ]
    for key, minimum in checks:
        value = float(metrics.get(key) or 0.0)
        if value < minimum:
            failures.append(f"{key}={value:.4f} < {minimum:.4f}")
    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(
            "LLM smoke guard failed; stopping before full test LLM calls. "
            f"{detail}. Inspect smoke outputs or rerun with --allow_bad_smoke."
        )
    logger.info(
        "LLM smoke guard passed: delta_bleu=%+.4f delta_chrf=%+.4f "
        "delta_rouge_l=%+.4f parse_failed=%s",
        float(metrics.get("delta_bleu") or 0.0),
        float(metrics.get("delta_chrf") or 0.0),
        float(metrics.get("delta_rouge_l") or 0.0),
        parse_failed,
    )


def ensure_full_predictions(args: argparse.Namespace, device: str) -> None:
    if args.skip_full_predictions:
        logger.info("Skipping full draft prediction generation.")
        return
    manifest = ROOT / args.manifest
    out = ROOT / args.full_predictions
    expected = count_jsonl(manifest)
    current = count_jsonl(out)
    target = args.max_full_samples or expected

    if out.exists() and not args.force_predictions and current >= target:
        logger.info("Full predictions already exist: %s lines=%s target=%s", out, current, target)
        return

    maybe_remove(out, args.force_predictions, args.dry_run)
    cmd = [
        py(),
        "scripts/run_predict_samples.py",
        "--checkpoint",
        args.checkpoint,
        "--config",
        args.config,
        "--manifest",
        args.manifest,
        "--gloss_vocab",
        args.gloss_vocab,
        "--split",
        "test",
        "--device",
        device,
        "--start",
        "0",
        "--num_samples",
        str(target),
        "--draft_mode",
        "greedy",
        "--output",
        args.full_predictions,
    ]
    run(cmd, args.dry_run)


def run_full_llm(args: argparse.Namespace) -> None:
    if args.skip_full_llm:
        logger.info("Skipping full LLM correction and metrics.")
        return
    full_input = ROOT / args.full_predictions
    if not full_input.exists() and not args.dry_run:
        raise FileNotFoundError(f"Full predictions not found: {full_input}")

    cmd = [
        py(),
        "scripts/run_llm_corrected_eval.py",
        "--input",
        args.full_predictions,
        "--output",
        args.full_corrected,
        "--metrics_output",
        args.full_metrics,
        "--baseline",
        args.baseline,
        *llm_args(args),
    ]
    if args.max_full_samples is not None:
        cmd.extend(["--max_samples", str(args.max_full_samples)])
    if args.resume_llm:
        cmd.append("--resume")
    if args.sleep_seconds:
        cmd.extend(["--sleep_seconds", str(args.sleep_seconds)])
    run(cmd, args.dry_run)


def print_comparison(metrics_path: Path, baseline_path: Path) -> None:
    if not metrics_path.exists() or not baseline_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    logger.info("Comparison summary")
    logger.info("  samples: baseline=%s llm=%s", baseline.get("num_samples"), metrics.get("num_samples"))
    logger.info("  BLEU:    %.4f -> %.4f (%+.4f)", baseline.get("bleu", 0.0), metrics.get("final_bleu", 0.0), metrics.get("delta_final_vs_baseline_bleu", 0.0))
    logger.info("  chrF:    %.4f -> %.4f (%+.4f)", baseline.get("chrf", 0.0), metrics.get("final_chrf", 0.0), metrics.get("delta_final_vs_baseline_chrf", 0.0))
    logger.info("  ROUGE-L: %.4f -> %.4f (%+.4f)", baseline.get("rouge_l", 0.0), metrics.get("final_rouge_l", 0.0), metrics.get("delta_final_vs_baseline_rouge_l", 0.0))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("Resolved device: %s", device)
    if not shutil.which(py()):
        raise RuntimeError(f"Python executable not found: {py()}")

    run_smoke(args)
    enforce_smoke_gate(args)
    run_health_check(args)
    ensure_full_predictions(args, device)
    run_full_llm(args)

    if not args.dry_run:
        print_comparison(ROOT / args.full_metrics, ROOT / args.baseline)


if __name__ == "__main__":
    main()
