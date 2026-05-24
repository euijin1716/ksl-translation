#!/usr/bin/env python3
"""데이터셋 adapter → signer-independent split → manifest 생성 스크립트.

AI Hub / NIASL2021 데이터를 받은 뒤 이 스크립트 한 번으로
data/manifests/all.jsonl, train.jsonl, valid.jsonl, test.jsonl 을 만든다.

그 다음 순서:
    python scripts/build_manifest.py --datasets aihub_sign --root_aihub_sign data/aihub_sign
    python scripts/run_preprocess.py --manifest data/manifests/all.jsonl
    python scripts/run_train.py --manifest data/manifests/train.jsonl

사용 예시:
    # AI Hub 수어 영상 한 개만
    python scripts/build_manifest.py \\
        --datasets aihub_sign \\
        --root_aihub_sign data/aihub_sign

    # AI Hub 두 개 합산
    python scripts/build_manifest.py \\
        --datasets aihub_sign aihub_disaster \\
        --root_aihub_sign data/aihub_sign \\
        --root_aihub_disaster data/aihub_disaster

    # 세 개 모두 합산
    python scripts/build_manifest.py \\
        --datasets niasl2021 aihub_sign aihub_disaster \\
        --root_niasl2021 data/niasl2021 \\
        --root_aihub_sign data/aihub_sign \\
        --root_aihub_disaster data/aihub_disaster
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="adapter → manifest 생성")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["niasl2021", "aihub_sign", "aihub_disaster"],
        required=True,
        help="사용할 데이터셋 목록",
    )
    p.add_argument("--root_niasl2021",    default="data/niasl2021",      help="NIASL2021 루트")
    p.add_argument("--root_aihub_sign",   default="data/aihub_sign",     help="AI Hub 수어 루트")
    p.add_argument("--root_aihub_disaster", default="data/aihub_disaster", help="AI Hub 재난 루트")
    p.add_argument("--manifest_dir",      default="data/manifests",      help="manifest 저장 디렉터리")
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--train_ratio",       type=float, default=0.8)
    p.add_argument("--valid_ratio",       type=float, default=0.1)
    # 포맷 불일치 시 디버그 로그 확인용
    p.add_argument("--debug",             action="store_true", help="DEBUG 레벨 로그 출력")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.data.manifest import write_manifest
    from src.data.splits import check_signer_leakage, make_signer_independent_split

    # ── 1. 각 adapter에서 샘플 수집 ───────────────────────────────────────────
    all_samples = []

    if "niasl2021" in args.datasets:
        from src.data.adapters.niasl2021_adapter import NIASL2021Adapter
        adapter = NIASL2021Adapter(root=args.root_niasl2021)
        samples = list(adapter.iter_samples())
        logger.info(f"NIASL2021: {len(samples)}개 샘플")
        all_samples.extend(samples)

    if "aihub_sign" in args.datasets:
        from src.data.adapters.aihub_sign_adapter import AIHubSignAdapter
        adapter = AIHubSignAdapter(
            root=args.root_aihub_sign,
            config={"skip_missing_video": True},
        )
        samples = list(adapter.iter_samples())
        logger.info(f"AIHub Sign: {len(samples)}개 샘플")
        all_samples.extend(samples)

    if "aihub_disaster" in args.datasets:
        from src.data.adapters.aihub_disaster_adapter import AIHubDisasterAdapter
        adapter = AIHubDisasterAdapter(
            root=args.root_aihub_disaster,
            config={"skip_missing_video": True},
        )
        samples = list(adapter.iter_samples())
        logger.info(f"AIHub Disaster: {len(samples)}개 샘플")
        all_samples.extend(samples)

    if not all_samples:
        logger.error(
            "수집된 샘플이 없습니다.\n"
            "데이터 경로와 폴더 구조를 확인하세요: docs/data_guide.md\n"
            "포맷 불일치 확인 방법: --debug 옵션으로 재실행"
        )
        sys.exit(1)

    logger.info(f"전체 샘플 수: {len(all_samples)}")

    # ── 2. signer-independent split ───────────────────────────────────────────
    dataset_label = "_".join(sorted(args.datasets))
    split_samples, split_manifest = make_signer_independent_split(
        all_samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        dataset_name=dataset_label,
    )

    # signer leakage 검사
    problems = check_signer_leakage(split_samples)
    if problems:
        logger.error(f"Signer leakage 발견!\n" + "\n".join(problems))
        sys.exit(1)

    counts = {g: sum(1 for s in split_samples if s.split_group == g)
              for g in ("train", "valid", "test")}
    signer_counts = {g: len(v) for g, v in split_manifest.signer_split.items()}
    logger.info(f"Split 결과: samples={counts}")
    logger.info(f"           signers={signer_counts}")

    # ── 3. manifest 저장 ───────────────────────────────────────────────────────
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_path = manifest_dir / "all.jsonl"
    write_manifest(split_samples, all_path)
    logger.info(f"전체 manifest: {all_path} ({len(split_samples)}개)")

    for split_group in ("train", "valid", "test"):
        subset = [s for s in split_samples if s.split_group == split_group]
        if subset:
            split_path = manifest_dir / f"{split_group}.jsonl"
            write_manifest(subset, split_path)
            logger.info(f"  {split_path.name}: {len(subset)}개 샘플")

    logger.info("\n다음 단계:")
    logger.info(f"  1. 전처리:  python scripts/run_preprocess.py --manifest {all_path} --config configs/base.yaml")
    logger.info(f"  2. 학습:    python scripts/run_train.py --manifest data/manifests/train.jsonl --config configs/base.yaml")


if __name__ == "__main__":
    main()
