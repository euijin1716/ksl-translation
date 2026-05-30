"""Streaming inference state machine.

sliding window 기반으로 실시간 수어 인식을 처리한다.
확신이 낮은 중간 결과를 너무 빨리 확정하지 않는다.
StreamingInferencePipeline이 ENDED 시 InferencePipeline을 호출해 LLM 보정을 완성한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .pipeline import InferencePipeline, InferenceResult

logger = logging.getLogger(__name__)


class ActivityState(Enum):
    IDLE = "idle"
    ONSET = "onset"        # 수어 시작 감지
    ONGOING = "ongoing"    # 수어 진행 중
    OFFSET = "offset"      # 수어 종료 감지
    ENDED = "ended"        # 수어 완료 (후처리 트리거)


@dataclass
class StreamingConfig:
    window_size: int = 32           # 프레임 단위
    stride: int = 8
    onset_threshold: float = 0.6    # 수어 시작 판단 threshold
    offset_threshold: float = 0.4   # 수어 종료 판단 threshold
    min_sign_frames: int = 10       # 최소 수어 구간 프레임 수
    max_pending_frames: int = 200   # 버퍼 최대 크기


@dataclass
class StreamingResult:
    """스트리밍 추론 한 스텝의 결과."""
    state: ActivityState
    is_final: bool = False          # True이면 이 발화를 확정
    partial_text: str = ""          # 확정 전 중간 결과 (표시용)
    final_text: str = ""            # 확정된 발화
    confidence: float = 0.0
    frame_range: tuple[int, int] = (0, 0)
    uncertain_spans: list[dict[str, Any]] = field(default_factory=list)
    retry_or_clarify: bool = False


class StreamingStateMachine:
    """수어 발화 경계를 관리하는 상태 기계.

    Args:
        config: 스트리밍 설정
    """

    def __init__(self, config: StreamingConfig | None = None) -> None:
        self.config = config or StreamingConfig()
        self._state = ActivityState.IDLE
        self._buffer: list[Any] = []        # 프레임 버퍼
        self._onset_frame: int = 0
        self._frame_counter: int = 0
        self._consecutive_idle: int = 0

    @property
    def state(self) -> ActivityState:
        return self._state

    def push_frame(
        self,
        frame_features: Any,
        activity_prob: float,
    ) -> StreamingResult:
        """단일 프레임을 처리하고 현재 상태를 반환한다.

        Args:
            frame_features: 프레임 특징 (모델 입력용)
            activity_prob: 수어 활동 확률 (0~1)

        Returns:
            StreamingResult
        """
        c = self.config
        self._frame_counter += 1
        self._buffer.append(frame_features)

        # 버퍼 크기 제한
        if len(self._buffer) > c.max_pending_frames:
            self._buffer.pop(0)

        result = self._transition(activity_prob)
        return result

    def _transition(self, activity_prob: float) -> StreamingResult:
        c = self.config
        state = self._state

        if state == ActivityState.IDLE:
            if activity_prob >= c.onset_threshold:
                self._state = ActivityState.ONSET
                self._onset_frame = self._frame_counter
                self._consecutive_idle = 0

        elif state == ActivityState.ONSET:
            if activity_prob >= c.onset_threshold:
                self._state = ActivityState.ONGOING
            else:
                self._state = ActivityState.IDLE

        elif state == ActivityState.ONGOING:
            if activity_prob < c.offset_threshold:
                self._consecutive_idle += 1
                if self._consecutive_idle >= 3:
                    self._state = ActivityState.OFFSET
            else:
                self._consecutive_idle = 0

        elif state == ActivityState.OFFSET:
            sign_len = self._frame_counter - self._onset_frame
            if sign_len >= c.min_sign_frames:
                self._state = ActivityState.ENDED
                return StreamingResult(
                    state=self._state,
                    is_final=True,
                    frame_range=(self._onset_frame, self._frame_counter),
                    confidence=activity_prob,
                )
            else:
                # 너무 짧은 구간 → idle로 복귀
                self._state = ActivityState.IDLE

        if self._state == ActivityState.ENDED:
            self._state = ActivityState.IDLE
            self._buffer.clear()

        return StreamingResult(
            state=self._state,
            is_final=False,
            frame_range=(self._onset_frame, self._frame_counter),
        )

    def reset(self) -> None:
        self._state = ActivityState.IDLE
        self._buffer.clear()
        self._onset_frame = 0
        self._frame_counter = 0
        self._consecutive_idle = 0

    def get_buffer(self) -> list[Any]:
        return list(self._buffer)


class StreamingInferencePipeline:
    """StreamingStateMachine과 InferencePipeline을 연결하는 통합 파이프라인.

    발화 ENDED 감지 시 버퍼를 collate해 InferencePipeline.infer()를 호출하고
    LLM 문맥 보정까지 완료된 StreamingResult를 반환한다.

    Args:
        pipeline: 구성 완료된 InferencePipeline (LLM corrector 포함)
        config: 스트리밍 상태 기계 설정
        domain: 발화 도메인 (전체 세션 고정 또는 동적 변경 가능)
    """

    def __init__(
        self,
        pipeline: "InferencePipeline",
        config: StreamingConfig | None = None,
        domain: str = "unknown",
    ) -> None:
        self.pipeline = pipeline
        self.sm = StreamingStateMachine(config)
        self.domain = domain

    def push_frame(
        self,
        frame_features: dict[str, Any],
        activity_prob: float,
        domain: str | None = None,
    ) -> StreamingResult:
        """단일 프레임을 처리한다.

        Args:
            frame_features: 단일 프레임 특징 dict (tensor shape: 각 키별 공간 차원만)
            activity_prob: 수어 활동 확률 (0~1)
            domain: 이번 프레임의 도메인 (None이면 초기화 시 domain 사용)

        Returns:
            StreamingResult — is_final=True이면 LLM 보정까지 완료된 발화
        """
        used_domain = domain or self.domain
        sm_result = self.sm.push_frame(frame_features, activity_prob)

        if not sm_result.is_final:
            return sm_result

        # ENDED: 버퍼를 collate해 InferencePipeline 실행
        buffer = self.sm.get_buffer()
        if not buffer:
            logger.warning("StreamingInferencePipeline: ENDED but buffer is empty.")
            return sm_result

        try:
            batch = _collate_frame_buffer(buffer)
            infer_result = self.pipeline.infer(batch, domain=used_domain)
            return StreamingResult(
                state=ActivityState.ENDED,
                is_final=True,
                final_text=infer_result.final_text,
                partial_text=infer_result.draft_text,
                confidence=infer_result.confidence,
                frame_range=sm_result.frame_range,
                uncertain_spans=infer_result.uncertain_spans,
                retry_or_clarify=infer_result.retry_or_clarify,
            )
        except Exception as e:
            logger.error("StreamingInferencePipeline inference failed: %s", e)
            return StreamingResult(
                state=ActivityState.ENDED,
                is_final=True,
                final_text="",
                confidence=sm_result.confidence,
                frame_range=sm_result.frame_range,
                retry_or_clarify=True,
            )

    def reset(self) -> None:
        self.sm.reset()
        self.pipeline.corrector.reset_history()


def _collate_frame_buffer(
    buffer: list[dict[str, Any]],
) -> dict[str, Any]:
    """프레임 dict 리스트를 InferencePipeline이 기대하는 배치 형식으로 변환한다.

    각 프레임 dict의 tensor 값을 [T, ...] 로 스택한 뒤 B=1 배치 차원을 추가한다.
    tensor가 아닌 값(str, int 등)은 마지막 프레임 값을 사용한다.
    """
    if not buffer:
        return {}

    keys = buffer[0].keys()
    batch: dict[str, Any] = {}

    for key in keys:
        frames = [f[key] for f in buffer if key in f]
        if not frames:
            continue

        sample = frames[0]
        if isinstance(sample, torch.Tensor):
            # [T, ...] → [1, T, ...]
            stacked = torch.stack(frames, dim=0).unsqueeze(0)
            batch[key] = stacked
        else:
            # 비tensor 값은 마지막 프레임 값 사용
            batch[key] = frames[-1]

    return batch
