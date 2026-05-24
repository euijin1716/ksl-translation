#!/usr/bin/env python3
"""더미 비디오 파일 + manifest 생성 스크립트.

실제 데이터 없이 전처리 파이프라인 end-to-end 검증용.
더미 비디오를 만들고, signer-independent split manifest를 data/manifests/ 에 저장한다.

사용법:
    python scripts/setup_dummy_data.py
    python scripts/setup_dummy_data.py --num_samples 10 --fps 25
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_dummy_video(path: Path, num_frames: int = 30, fps: float = 25.0,
                     width: int = 640, height: int = 480) -> None:
    """MediaPipe가 얼굴/손을 검출할 수 있는 더미 영상을 생성한다.

    - 밝은 피부색 배경에 타원(얼굴)과 사각형(손) 도형을 그려
      랜드마크 검출 가능성을 높인다.
    - 완전한 검출 보장은 어렵지만 zero 텐서보다 현실적인 입력을 만든다.
    """
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    cx, cy = width // 2, height // 3          # 얼굴 중심
    face_rx, face_ry = width // 8, height // 6

    lhx, lhy = width // 4, height * 2 // 3    # 왼손 중심
    rhx, rhy = width * 3 // 4, height * 2 // 3  # 오른손 중심
    hand_r = width // 12

    skin = (180, 140, 100)       # BGR 피부색

    for i in range(num_frames):
        frame = np.full((height, width, 3), 230, dtype=np.uint8)  # 밝은 회색 배경

        # 얼굴 타원
        offset = int(3 * np.sin(i * 0.2))     # 미세 흔들림
        cv2.ellipse(frame, (cx + offset, cy), (face_rx, face_ry),
                    0, 0, 360, skin, -1)
        # 눈 (흰자 + 동공)
        for ex in [cx - face_rx // 3, cx + face_rx // 3]:
            cv2.circle(frame, (ex + offset, cy - face_ry // 5), face_rx // 6, (255, 255, 255), -1)
            cv2.circle(frame, (ex + offset, cy - face_ry // 5), face_rx // 10, (30, 30, 30), -1)
        # 입
        cv2.ellipse(frame, (cx + offset, cy + face_ry // 3),
                    (face_rx // 3, face_ry // 6), 0, 0, 180, (80, 60, 60), 2)

        # 왼손 (손가락 5개)
        loff = int(4 * np.sin(i * 0.3 + 1))
        cv2.circle(frame, (lhx + loff, lhy), hand_r, skin, -1)
        for fi in range(5):
            angle = -60 + fi * 30
            fx = lhx + loff + int(hand_r * 1.4 * np.sin(np.radians(angle)))
            fy = lhy - int(hand_r * 1.4 * np.cos(np.radians(angle)))
            cv2.line(frame, (lhx + loff, lhy), (fx, fy), skin, hand_r // 3)
            cv2.circle(frame, (fx, fy), hand_r // 4, skin, -1)

        # 오른손
        roff = int(4 * np.sin(i * 0.3 + 2))
        cv2.circle(frame, (rhx + roff, rhy), hand_r, skin, -1)
        for fi in range(5):
            angle = -60 + fi * 30
            fx = rhx + roff + int(hand_r * 1.4 * np.sin(np.radians(angle)))
            fy = rhy - int(hand_r * 1.4 * np.cos(np.radians(angle)))
            cv2.line(frame, (rhx + roff, rhy), (fx, fy), skin, hand_r // 3)
            cv2.circle(frame, (fx, fy), hand_r // 4, skin, -1)

        out.write(frame)
    out.release()


def parse_args():
    p = argparse.ArgumentParser(description="더미 비디오 + manifest 생성")
    p.add_argument("--num_samples", type=int, default=20, help="생성할 샘플 수 (기본 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video_root", default="data/dummy/videos")
    p.add_argument("--manifest_dir", default="data/manifests")
    return p.parse_args()


def main():
    args = parse_args()

    from src.data.adapters.dummy_adapter import DummyAdapter
    from src.data.manifest import write_manifest
    from src.data.splits import make_signer_independent_split

    # 1. 더미 샘플 메타데이터 생성 (DummyAdapter)
    logger.info(f"샘플 {args.num_samples}개 생성 중 (seed={args.seed})...")
    adapter = DummyAdapter(num_samples=args.num_samples, seed=args.seed)
    all_samples = list(adapter.iter_samples())

    # 2. 실제 더미 비디오 파일 생성
    logger.info(f"더미 비디오 파일 생성 중 → {args.video_root}/")
    video_root = Path(args.video_root)
    for sample in all_samples:
        video_path = Path(sample.video_path)
        if not video_path.exists():
            make_dummy_video(
                path=video_path,
                num_frames=sample.num_frames,
                fps=sample.fps,
            )
    logger.info(f"  생성 완료: {len(all_samples)}개 mp4")

    # 3. signer-independent split
    logger.info("Signer-independent split 생성 중...")
    split_samples, split_manifest = make_signer_independent_split(
        all_samples, dataset_name="dummy", seed=args.seed
    )
    split_manifest.validate()   # signer leakage 검사

    counts = {g: sum(1 for s in split_samples if s.split_group == g)
              for g in ("train", "valid", "test")}
    logger.info(f"  split: {counts}")
    logger.info(f"  signer split: { {g: len(v) for g, v in split_manifest.signer_split.items()} }")

    # 4. manifest 저장 (전체 / split별)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_path = manifest_dir / "all.jsonl"
    write_manifest(split_samples, all_path)
    logger.info(f"전체 manifest 저장: {all_path}  ({len(split_samples)}개)")

    for split_group in ("train", "valid", "test"):
        subset = [s for s in split_samples if s.split_group == split_group]
        if subset:
            split_path = manifest_dir / f"{split_group}.jsonl"
            write_manifest(subset, split_path)
            logger.info(f"  {split_group}.jsonl  ({len(subset)}개 샘플)")

    logger.info("setup_dummy_data 완료.")
    logger.info(f"다음 명령으로 전처리를 실행하세요:")
    logger.info(f"  python scripts/run_preprocess.py --manifest {all_path} --config configs/base.yaml")


if __name__ == "__main__":
    main()
