#!/usr/bin/env python3
"""Simulated KSL video demo pipeline.

This script prints a realistic end-to-end demo flow without requiring an
actual video, MediaPipe, or model checkpoint. The classes are intentionally
small and swappable so the fake implementations can later be replaced by real
video cropping, signal recognition, and model inference code.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


DEFAULT_VIDEO = "E:\\sua\\ksl_weather_eval\\data\\raw\\plzzzzz\\ulsan_mbc.mov"
DEFAULT_CHECKPOINT = "checkpoints/C/best.pt"
DEFAULT_RESULT = "강풍 주의보가 내려진 가운데 충전 중인 전기차에서 차에서 불이난 상황을 가정한 훈련."


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float
    exists: bool


@dataclass(frozen=True)
class CropResult:
    source_path: str
    cropped_path: str
    roi_xywh: tuple[int, int, int, int]
    kept_frames: int
    crop_confidence: float


@dataclass(frozen=True)
class SignalWindow:
    start_frame: int
    end_frame: int
    manual_label: str
    manual_confidence: float
    non_manual_label: str
    non_manual_confidence: float
    activity_state: str


@dataclass(frozen=True)
class ModelPrediction:
    final_text: str
    draft_text: str
    gloss_tokens: list[str]
    confidence: float
    simulated_accuracy: float
    latency_ms: int


class VideoProcessor(Protocol):
    def load(self, video_path: str) -> VideoInfo:
        ...

    def crop_signer(self, info: VideoInfo) -> CropResult:
        ...


class SignalRecognizer(Protocol):
    def recognize(self, crop: CropResult) -> list[SignalWindow]:
        ...


class ModelRunner(Protocol):
    def load(self, checkpoint: str) -> None:
        ...

    def predict(self, crop: CropResult, signals: list[SignalWindow]) -> ModelPrediction:
        ...


class DemoLogger:
    def __init__(self, sleep_scale: float = 0.35) -> None:
        self.sleep_scale = max(0.0, sleep_scale)
        self._step = 0

    def log(self, message: str, delay: float = 0.12) -> None:
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{stamp}] {message}", flush=True)
        if self.sleep_scale:
            time.sleep(delay * self.sleep_scale)

    def section(self, title: str) -> None:
        self._step += 1
        self.log(f"\n[{self._step}/5] {title}", delay=0.18)


class FakeVideoProcessor:
    """Demo-only video loader/cropper.

    Replace this class with an OpenCV or MediaPipe based implementation later.
    Keep the method signatures the same and the rest of the pipeline can stay
    unchanged.
    """

    def __init__(self, logger: DemoLogger, frames: int) -> None:
        self.logger = logger
        self.frames = frames

    def load(self, video_path: str) -> VideoInfo:
        exists = Path(video_path).exists()
        info = VideoInfo(
            path=video_path,
            fps=29.97,
            width=1920,
            height=1080,
            total_frames=self.frames,
            duration_sec=round(self.frames / 29.97, 2),
            exists=exists,
        )
        mode = "실제 파일 감지" if exists else "데모 입력으로 가정"
        self.logger.log(f"video.open path={video_path} ({mode})")
        self.logger.log(
            f"metadata fps={info.fps:.2f}, size={info.width}x{info.height}, "
            f"frames={info.total_frames}, duration={info.duration_sec:.2f}s"
        )
        return info

    def crop_signer(self, info: VideoInfo) -> CropResult:
        self.logger.log("signer.roi detect: upper-body + hands search window")
        self.logger.log("signer.track stabilize: 5-frame median smoothing")
        result = CropResult(
            source_path=info.path,
            cropped_path="tmp/demo_simulation/cropped_signer_roi.mp4",
            roi_xywh=(1160, 120, 520, 760),
            kept_frames=info.total_frames,
            crop_confidence=0.94,
        )
        x, y, w, h = result.roi_xywh
        self.logger.log(
            f"crop.write roi=(x={x}, y={y}, w={w}, h={h}), "
            f"kept_frames={result.kept_frames}, confidence={result.crop_confidence:.2f}"
        )
        return result


class FakeSignalRecognizer:
    """Demo-only manual/non-manual signal recognizer."""

    _manual_labels = [
        "양손 상승",
        "오른손 지시",
        "양손 전방 이동",
        "손 모양 전환",
        "마무리 정지",
    ]
    _nms_labels = [
        "눈썹 상승",
        "시선 전방",
        "입 모양 강조",
        "고개 끄덕임",
        "중립 표정",
    ]

    def __init__(self, logger: DemoLogger, rng: random.Random, window: int = 16) -> None:
        self.logger = logger
        self.rng = rng
        self.window = window

    def recognize(self, crop: CropResult) -> list[SignalWindow]:
        windows: list[SignalWindow] = []
        for idx, start in enumerate(range(0, crop.kept_frames, self.window)):
            end = min(start + self.window - 1, crop.kept_frames - 1)
            progress = idx / max(1, crop.kept_frames // self.window)
            manual_conf = _bounded(0.79 + progress * 0.10 + self.rng.uniform(-0.015, 0.02))
            nms_conf = _bounded(0.76 + progress * 0.11 + self.rng.uniform(-0.02, 0.02))
            state = "ongoing" if end < crop.kept_frames - 1 else "ended"
            window = SignalWindow(
                start_frame=start,
                end_frame=end,
                manual_label=self._manual_labels[idx % len(self._manual_labels)],
                manual_confidence=manual_conf,
                non_manual_label=self._nms_labels[idx % len(self._nms_labels)],
                non_manual_confidence=nms_conf,
                activity_state=state,
            )
            windows.append(window)
            self.logger.log(
                "signal.window "
                f"{start:04d}-{end:04d} | "
                f"수지={window.manual_label}({manual_conf:.2f}) | "
                f"비수지={window.non_manual_label}({nms_conf:.2f}) | "
                f"state={state}",
                delay=0.08,
            )
        return windows


class FakeModelRunner:
    """Demo-only model runner with realistic progress logs."""

    def __init__(
        self,
        logger: DemoLogger,
        rng: random.Random,
        target_accuracy: float,
        final_text: str,
    ) -> None:
        self.logger = logger
        self.rng = rng
        self.target_accuracy = _bounded(target_accuracy)
        self.final_text = final_text
        self._loaded_checkpoint: str | None = None

    def load(self, checkpoint: str) -> None:
        self._loaded_checkpoint = checkpoint
        self.logger.log(f"model.load checkpoint={checkpoint}")
        self.logger.log("model.ready streams=landmark+hand_crop+face_expr, decoder=stage_c")

    def predict(self, crop: CropResult, signals: list[SignalWindow]) -> ModelPrediction:
        if self._loaded_checkpoint is None:
            raise RuntimeError("model must be loaded before predict()")

        gloss = ["강풍 주의보", "충전", "전기차", "차", "불", "훈련"]
        partials = [
            "강풍 주의보...",
            "강풍 주의보가 내려진 가운데...",
            "강풍 주의보가 내려진 가운데 충전 중인...",
            "강풍 주의보가 내려진 가운데 충전 중인 전기차에서 ...",
            self.final_text,
        ]
        start = time.perf_counter()
        last_conf = 0.0
        for idx, signal in enumerate(signals):
            ratio = (idx + 1) / len(signals)
            last_conf = _bounded(0.68 + ratio * (self.target_accuracy - 0.68))
            partial = partials[min(int(ratio * len(partials)), len(partials) - 1)]
            self.logger.log(
                "model.infer "
                f"frames={signal.start_frame:04d}-{signal.end_frame:04d} | "
                f"gloss_top1={gloss[min(idx, len(gloss) - 1)]} | "
                f"confidence={last_conf:.2f} | partial=\"{partial}\"",
                delay=0.10,
            )

        elapsed_ms = max(180, int((time.perf_counter() - start) * 1000) + 236)
        draft = "강풍 주의보가 내려진 가운데 충전 중인 전기차에서 차에서 불이난 상황을 가정한 훈련"
        return ModelPrediction(
            final_text=self.final_text,
            draft_text=draft,
            gloss_tokens=gloss,
            confidence=last_conf,
            simulated_accuracy=self.target_accuracy,
            latency_ms=elapsed_ms,
        )


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return round(min(high, max(low, value)), 3)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="최종 발표용 KSL 추론 로그를 시뮬레이션합니다."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="나중에 실제 영상으로 교체할 입력 경로")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="나중에 실제 모델로 교체할 체크포인트 경로")
    parser.add_argument("--frames", type=int, default=96, help="시뮬레이션할 프레임 수")
    parser.add_argument("--target-accuracy", type=float, default=0.88, help="최종 로그에 표시할 목표 정확도")
    parser.add_argument("--result-text", default=DEFAULT_RESULT, help="최종 인식 결과 문장")
    parser.add_argument("--seed", type=int, default=7, help="재현 가능한 로그 생성을 위한 난수 시드")
    parser.add_argument("--sleep-scale", type=float, default=1.0, help="로그 출력 지연 배율")
    parser.add_argument("--no-sleep", action="store_true", help="테스트/녹화 리허설용으로 지연 없이 출력")
    parser.add_argument("--json-out", type=Path, help="최종 결과를 JSON 파일로 저장")
    return parser


def run_demo(args: argparse.Namespace) -> ModelPrediction:
    rng = random.Random(args.seed)
    logger = DemoLogger(sleep_scale=0.0 if args.no_sleep else args.sleep_scale)
    video = FakeVideoProcessor(logger, frames=max(1, args.frames))
    signals = FakeSignalRecognizer(logger, rng)
    model = FakeModelRunner(logger, rng, args.target_accuracy, args.result_text)

    logger.section("영상 불러오기")
    video_info = video.load(args.video)

    logger.section("수어 영역 크롭")
    crop = video.crop_signer(video_info)

    logger.section("수지/비수지 신호 인식")
    signal_windows = signals.recognize(crop)

    logger.section("모델 로드 및 크롭 영상 테스트")
    model.load(args.checkpoint)
    prediction = model.predict(crop, signal_windows)

    logger.section("최종 결과")
    logger.log(f"draft_text=\"{prediction.draft_text}\"")
    logger.log(f"final_text=\"{prediction.final_text}\"")
    logger.log(f"gloss={', '.join(prediction.gloss_tokens)}")
    logger.log(
        f"confidence={prediction.confidence:.2f}, "
        f"simulated_accuracy={prediction.simulated_accuracy:.2f}, "
        f"latency={prediction.latency_ms}ms"
    )
    logger.log("mode=simulation;")

    if args.json_out:
        payload = {
            "video": asdict(video_info),
            "crop": asdict(crop),
            "signals": [asdict(item) for item in signal_windows],
            "prediction": asdict(prediction),
            "simulation": True,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.log(f"json.write path={args.json_out}")

    return prediction


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
