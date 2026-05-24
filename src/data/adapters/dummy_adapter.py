"""더미 데이터셋 adapter.

실제 데이터 없이 end-to-end smoke test를 가능하게 한다.
스키마 구조는 실제 KSLSample과 동일해야 한다.
"""

from __future__ import annotations

import random
from typing import Iterator

from ..schema import KSLSample, NMSLabels
from .base import BaseAdapter

_DUMMY_SIGNERS = [f"S{i:03d}" for i in range(1, 21)]  # S001 ~ S020
_DUMMY_DOMAINS = ["hospital", "directions", "order", "reservation", "public", "help"]
_DUMMY_GLOSSES = [
    ["머리", "아프다"], ["배", "아프다"], ["어디", "병원"],
    ["길", "알려주다"], ["버스", "타다"], ["지하철", "어디"],
    ["이것", "주문하다"], ["계산", "하다"], ["얼마"],
    ["예약", "하다"], ["확인", "하다"], ["언제"],
    ["민원", "신청"], ["서류", "필요"], ["어떻게"],
    ["도움", "요청"], ["긴급", "도움"], ["전화"],
]
_DUMMY_TEXTS = [
    "두통이 있습니다.", "배가 아픕니다.", "병원이 어디 있나요?",
    "길을 알려주세요.", "버스를 타고 싶습니다.", "지하철역이 어디인가요?",
    "이것을 주문하겠습니다.", "계산해 주세요.", "얼마예요?",
    "예약하고 싶습니다.", "예약을 확인해 주세요.", "언제 가능한가요?",
    "민원을 신청하고 싶습니다.", "필요한 서류가 있나요?", "어떻게 하면 되나요?",
    "도움이 필요합니다.", "긴급 상황입니다.", "전화해 주세요.",
]


class DummyAdapter(BaseAdapter):
    """더미 샘플을 생성하는 adapter.

    Args:
        num_samples: 생성할 샘플 수 (기본 20)
        seed: 재현성을 위한 난수 시드
    """

    def __init__(self, num_samples: int = 20, seed: int = 42) -> None:
        super().__init__(root=".", config={})
        self.num_samples = num_samples
        self.seed = seed

    @property
    def dataset_name(self) -> str:
        return "dummy"

    def iter_samples(self) -> Iterator[KSLSample]:
        rng = random.Random(self.seed)
        for i in range(self.num_samples):
            idx = i % len(_DUMMY_TEXTS)
            signer_id = _DUMMY_SIGNERS[i % len(_DUMMY_SIGNERS)]
            domain = _DUMMY_DOMAINS[idx % len(_DUMMY_DOMAINS)]
            yield KSLSample(
                sample_id=f"dummy_{i:05d}",
                dataset_name=self.dataset_name,
                domain=domain,
                scenario_id=f"sc_{domain}_{idx:02d}",
                turn_id=0,
                utterance_id=f"utt_{i:05d}",
                signer_id=signer_id,
                split_group="train",      # split generator가 덮어씀
                video_path=f"data/dummy/videos/{i:05d}.mp4",
                fps=25.0,
                num_frames=rng.randint(30, 90),
                korean_text=_DUMMY_TEXTS[idx],
                gloss_tokens=_DUMMY_GLOSSES[idx],
                nms_labels=NMSLabels(
                    eyebrow_raise=rng.choice([True, False, None]),
                    mouth_open=rng.choice([True, False]),
                    head_nod=rng.choice([True, False, None]),
                ),
                intent=domain,
                intent_source="gold",
                quality_flags=[],
                has_face=True,
                has_hands=True,
                metadata={"dummy": True, "index": i},
            )
