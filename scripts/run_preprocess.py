#!/usr/bin/env python3
"""전처리 실행 스크립트.

사용법:
    python scripts/run_preprocess.py --manifest data/manifests/train.jsonl \\
        --config configs/base.yaml --output_root data/keypoints
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--output_root", default="data/keypoints")
    p.add_argument("--num_workers", type=int, default=1,
                   help="병렬 전처리 worker 수. Windows/GPU 메모리를 고려해 2~4 권장")
    p.add_argument("--progress_every", type=int, default=25,
                   help="worker별 진행 로그 출력 간격")
    p.add_argument("--no_skip_existing", action="store_true",
                   help="이미 pose.npy가 있는 샘플도 다시 처리")
    p.add_argument("--target_fps", type=float, default=None,
                   help="Override MediaPipe sampling FPS. Lower values are faster.")
    p.add_argument("--skip_crops", action="store_true",
                   help="Extract keypoints only and skip ROI crop image generation.")
    p.add_argument("--crop_only_missing", action="store_true",
                   help="Do not rerun MediaPipe for existing keypoints; only create missing crops when saved bbox arrays exist.")
    return p.parse_args()


def _process_worker(
    worker_id: int,
    samples: list,
    preprocess_cfg: dict[str, Any],
) -> tuple[list, list[str]]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker_logger = logging.getLogger(__name__)
    worker_logger.info(f"Worker {worker_id}: processing {len(samples)} samples")

    from src.preprocess.pipelines.extraction_pipeline import ExtractionPipeline

    with ExtractionPipeline(preprocess_cfg) as pipeline:
        processed, failed = pipeline.process_batch(samples)
    worker_logger.info(f"Worker {worker_id}: done ok={len(processed)} failed={len(failed)}")
    return processed, failed


def _split_round_robin(items: list, num_workers: int) -> list[list]:
    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        chunks[i % num_workers].append(item)
    return [chunk for chunk in chunks if chunk]


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocess_cfg = cfg.get("preprocess", {})
    preprocess_cfg["keypoint_root"] = args.output_root
    preprocess_cfg["skip_existing"] = True if args.crop_only_missing else not args.no_skip_existing
    preprocess_cfg["progress_every"] = args.progress_every
    preprocess_cfg["enable_crops"] = not args.skip_crops
    preprocess_cfg["crop_missing_from_saved_bbox"] = True
    if args.target_fps is not None:
        extractor_cfg = dict(preprocess_cfg.get("extractor_config", {}))
        extractor_cfg["target_fps"] = args.target_fps
        preprocess_cfg["target_fps"] = args.target_fps
        preprocess_cfg["extractor_config"] = extractor_cfg

    from src.data.manifest import read_manifest, write_manifest

    samples = list(read_manifest(args.manifest))
    logger.info(
        f"Processing {len(samples)} samples "
        f"(workers={args.num_workers}, skip_existing={preprocess_cfg['skip_existing']})"
    )

    if args.num_workers <= 1:
        from src.preprocess.pipelines.extraction_pipeline import ExtractionPipeline

        with ExtractionPipeline(preprocess_cfg) as pipeline:
            processed, failed = pipeline.process_batch(samples)
    else:
        processed, failed = [], []
        chunks = _split_round_robin(samples, args.num_workers)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(_process_worker, i, chunk, preprocess_cfg)
                for i, chunk in enumerate(chunks, start=1)
            ]
            for future in as_completed(futures):
                worker_processed, worker_failed = future.result()
                processed.extend(worker_processed)
                failed.extend(worker_failed)
                logger.info(
                    f"Collected worker result: total_ok={len(processed)} "
                    f"total_failed={len(failed)}"
                )

    logger.info(f"Done: {len(processed)} ok, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed samples: {failed}")

    # keypoint_path가 채워진 샘플로 manifest를 덮어쓴다.
    failed_ids = set(failed)
    failed_samples = [s for s in samples if s.sample_id in failed_ids]
    all_samples = sorted(processed + failed_samples, key=lambda s: s.sample_id)
    write_manifest(all_samples, args.manifest)
    logger.info(f"Manifest updated: {args.manifest}")

    # split별 manifest(train/valid/test)도 함께 갱신한다.
    manifest_dir = Path(args.manifest).parent
    for split_group in ("train", "valid", "test"):
        split_path = manifest_dir / f"{split_group}.jsonl"
        if split_path.exists():
            subset = [s for s in all_samples if s.split_group == split_group]
            write_manifest(subset, split_path)
            logger.info(f"  {split_path.name} updated ({len(subset)} samples)")


if __name__ == "__main__":
    main()
