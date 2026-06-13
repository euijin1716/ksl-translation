#!/usr/bin/env python3
"""Run LLM correction for prediction JSONL and score final_text.

This script is intentionally prediction-file based:

1. read records that already contain draft_greedy/reference_text
2. call ContextCorrector to produce final_text
3. save corrected JSONL incrementally
4. compute BLEU/chrF/ROUGE-L for draft and final_text

It can be used for both a 20-sample smoke test and the full test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import compute_bleu, compute_chrf, compute_rouge_l
from src.llm.factory import build_corrector
from src.llm.provider import LLMOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="eval_results/pred_new.jsonl")
    p.add_argument("--output", default="eval_results/pred_new_llm_corrected.jsonl")
    p.add_argument("--metrics_output", default="eval_results/stage_c_test_new_llm_corrected.json")
    p.add_argument("--baseline", default="eval_results/stage_c_test_new.json")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--provider", default=None, choices=["dummy", "claude", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--max_retries", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--sleep_seconds", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail_fast", action="store_true")
    p.add_argument("--save_raw_response", action="store_true")
    p.add_argument("--keep_history", action="store_true", help="Do not reset previous-turn context per sample.")
    p.add_argument("--metrics_only", action="store_true", help="Only score an existing corrected JSONL.")
    p.add_argument("--health_check_only", action="store_true", help="Only instantiate provider and call health_check().")
    return p.parse_args()


def load_llm_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    path = Path(args.config)
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm_cfg = dict(cfg.get("llm", {}))
    if args.provider:
        llm_cfg["provider"] = args.provider
        if args.model is None:
            if args.provider == "dummy":
                llm_cfg["model"] = "dummy"
            elif args.provider == "openai":
                llm_cfg["model"] = "gpt-4o"
    if args.model:
        llm_cfg["model"] = args.model
    if args.max_tokens is not None:
        llm_cfg["max_tokens"] = args.max_tokens
    if args.max_retries is not None:
        llm_cfg["max_retries"] = args.max_retries
    return llm_cfg


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("index"))


def confidence_from_record(record: dict[str, Any]) -> float:
    conf = record.get("draft_greedy_conf") or {}
    for key in ("mean_p", "min_p"):
        value = conf.get(key)
        if value is not None:
            return float(value)
    return 0.0


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for record in iter_jsonl(path):
        keys.add(record_key(record))
    return keys


def correct_predictions(args: argparse.Namespace, llm_cfg: dict[str, Any]) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corrector = build_corrector(llm_cfg)
    existing = load_existing_keys(output_path) if args.resume else set()
    mode = "a" if args.resume and output_path.exists() else "w"
    provider = llm_cfg.get("provider", "dummy")
    model = llm_cfg.get("model", "")

    records = list(iter_jsonl(input_path))
    records = records[args.start :]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    logger.info(
        "LLM correction input=%s output=%s records=%s provider=%s model=%s resume=%s",
        input_path,
        output_path,
        len(records),
        provider,
        model,
        args.resume,
    )

    written = 0
    skipped = 0
    with output_path.open(mode, encoding="utf-8") as out:
        for i, record in enumerate(records, start=1):
            key = record_key(record)
            if key in existing:
                skipped += 1
                continue

            draft = str(record.get("draft_greedy") or record.get("draft_text") or "")
            gloss = [str(x) for x in record.get("pred_gloss") or []]
            nms = record.get("pred_nms") or {}
            domain = str(record.get("domain") or "unknown")
            confidence = confidence_from_record(record)

            if not args.keep_history:
                corrector.reset_history()

            t0 = time.perf_counter()
            try:
                llm_out = corrector.correct(
                    korean_draft=draft,
                    gloss_hypotheses=gloss,
                    gloss_confidences=[],
                    nms_summary=nms,
                    confidence=confidence,
                    domain=domain,
                )
                error = ""
            except Exception as exc:
                if args.fail_fast:
                    raise
                logger.exception("LLM correction failed for %s; falling back to draft.", key)
                llm_out = LLMOutput(
                    final_text=draft,
                    retry_or_clarify=True,
                    uncertain_spans=[{"text": draft, "reason": "llm_exception"}],
                    normalization_notes="llm_exception",
                )
                error = repr(exc)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            corrected = dict(record)
            corrected.update(
                {
                    "final_text": llm_out.final_text,
                    "uncertain_spans": llm_out.uncertain_spans,
                    "retry_or_clarify": llm_out.retry_or_clarify,
                    "normalization_notes": llm_out.normalization_notes,
                    "llm_provider": provider,
                    "llm_model": model,
                    "llm_elapsed_ms": round(elapsed_ms, 3),
                    "llm_confidence_used": confidence,
                    "llm_error": error,
                }
            )
            if args.save_raw_response:
                corrected["llm_raw_response"] = llm_out.raw_response

            out.write(json.dumps(corrected, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            if i == 1 or i % 25 == 0:
                logger.info("corrected progress %s/%s written=%s skipped=%s", i, len(records), written, skipped)

    logger.info("LLM correction done: written=%s skipped=%s output=%s", written, skipped, output_path)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * p))
    return values[idx]


def score_file(corrected_path: Path, metrics_path: Path, baseline_path: Path | None, llm_cfg: dict[str, Any]) -> dict[str, Any]:
    records = list(iter_jsonl(corrected_path))
    scored = [r for r in records if r.get("reference_text") is not None]
    refs = [str(r.get("reference_text") or "") for r in scored]
    drafts = [str(r.get("draft_greedy") or r.get("draft_text") or "") for r in scored]
    finals = [str(r.get("final_text") or "") for r in scored]
    domains = [str(r.get("domain") or "unknown") for r in scored]

    draft_bleu = compute_bleu(drafts, refs)["bleu"] if scored else 0.0
    final_bleu = compute_bleu(finals, refs)["bleu"] if scored else 0.0
    draft_chrf = compute_chrf(drafts, refs) if scored else 0.0
    final_chrf = compute_chrf(finals, refs) if scored else 0.0
    draft_rouge = compute_rouge_l(drafts, refs) if scored else 0.0
    final_rouge = compute_rouge_l(finals, refs) if scored else 0.0

    domain_pairs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for domain, draft, final, ref in zip(domains, drafts, finals, refs):
        domain_pairs[domain].append((draft, final, ref))

    domain_scores = {}
    for domain, pairs in domain_pairs.items():
        d = [p[0] for p in pairs]
        f = [p[1] for p in pairs]
        r = [p[2] for p in pairs]
        domain_scores[domain] = {
            "num_samples": len(pairs),
            "draft_bleu": compute_bleu(d, r)["bleu"],
            "final_bleu": compute_bleu(f, r)["bleu"],
            "draft_chrf": compute_chrf(d, r),
            "final_chrf": compute_chrf(f, r),
            "draft_rouge_l": compute_rouge_l(d, r),
            "final_rouge_l": compute_rouge_l(f, r),
        }

    elapsed = [float(r.get("llm_elapsed_ms") or 0.0) for r in records]
    changed = sum(1 for r in records if str(r.get("final_text") or "") != str(r.get("draft_greedy") or ""))
    retry = sum(1 for r in records if bool(r.get("retry_or_clarify")))
    uncertain = sum(1 for r in records if r.get("uncertain_spans"))
    errors = sum(1 for r in records if r.get("llm_error"))
    notes = Counter(str(r.get("normalization_notes") or "") for r in records)
    parse_failed = notes.get("parse_failed", 0)

    baseline = None
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "split": "test",
        "num_samples": len(scored),
        "input_file": str(corrected_path),
        "llm_provider": llm_cfg.get("provider", "dummy"),
        "llm_model": llm_cfg.get("model", ""),
        "draft_bleu": draft_bleu,
        "draft_chrf": draft_chrf,
        "draft_rouge_l": draft_rouge,
        "final_bleu": final_bleu,
        "final_chrf": final_chrf,
        "final_rouge_l": final_rouge,
        "delta_bleu": final_bleu - draft_bleu,
        "delta_chrf": final_chrf - draft_chrf,
        "delta_rouge_l": final_rouge - draft_rouge,
        "domain_scores": domain_scores,
        "domain_distribution": dict(Counter(domains)),
        "changed_count": changed,
        "changed_rate": changed / max(len(records), 1),
        "retry_or_clarify_count": retry,
        "retry_or_clarify_rate": retry / max(len(records), 1),
        "uncertain_count": uncertain,
        "uncertain_rate": uncertain / max(len(records), 1),
        "llm_error_count": errors,
        "parse_failed_count": parse_failed,
        "parse_failed_rate": parse_failed / max(len(records), 1),
        "normalization_note_distribution": dict(notes),
        "llm_latency_ms_mean": statistics.mean(elapsed) if elapsed else 0.0,
        "llm_latency_ms_p95": percentile(elapsed, 0.95),
        "metric_warnings": [],
    }

    if baseline:
        result["baseline_file"] = str(baseline_path)
        result["baseline_num_samples"] = baseline.get("num_samples")
        result["baseline_bleu"] = baseline.get("bleu")
        result["baseline_chrf"] = baseline.get("chrf")
        result["baseline_rouge_l"] = baseline.get("rouge_l")
        result["delta_final_vs_baseline_bleu"] = final_bleu - float(baseline.get("bleu", 0.0))
        result["delta_final_vs_baseline_chrf"] = final_chrf - float(baseline.get("chrf", 0.0))
        result["delta_final_vs_baseline_rouge_l"] = final_rouge - float(baseline.get("rouge_l", 0.0))
        if baseline.get("num_samples") != len(scored):
            result["metric_warnings"].append(
                "baseline sample count differs; compare deltas only when using the same sample set."
            )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Saved metrics: %s | BLEU %.2f -> %.2f | chrF %.2f -> %.2f | ROUGE-L %.2f -> %.2f",
        metrics_path,
        draft_bleu,
        final_bleu,
        draft_chrf,
        final_chrf,
        draft_rouge,
        final_rouge,
    )
    return result


def main() -> None:
    args = parse_args()
    llm_cfg = load_llm_config(args)

    if args.health_check_only:
        corrector = build_corrector(llm_cfg)
        ok = corrector.provider.health_check()
        print(json.dumps({"provider": llm_cfg.get("provider"), "model": llm_cfg.get("model"), "ok": ok}, ensure_ascii=False))
        raise SystemExit(0 if ok else 1)

    if not args.metrics_only:
        correct_predictions(args, llm_cfg)

    score_file(
        corrected_path=Path(args.output),
        metrics_path=Path(args.metrics_output),
        baseline_path=Path(args.baseline) if args.baseline else None,
        llm_cfg=llm_cfg,
    )


if __name__ == "__main__":
    main()
