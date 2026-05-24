"""Dummy PyTorch Dataset.

실제 데이터가 없어도 모델 forward pass와 smoke test가 가능하도록 한다.
스키마 구조는 실제 데이터와 동일하다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .adapters.dummy_adapter import DummyAdapter
from .schema import KSLSample
from .signals import encode_nms_detail_labels, encode_nms_labels
from .splits import make_signer_independent_split

# 랜드마크 차원 상수 (assumptions.md 참조)
T_DEFAULT = 64          # 기본 시계열 길이
POSE_JOINTS = 25        # 상반신 (A-002)
HAND_JOINTS = 21        # 손 랜드마크
FACE_BLENDSHAPE = 52    # MediaPipe blendshape (A-001)
FACE_KEY_SUBSET = 68    # 주요 얼굴 랜드마크
HAND_CROP_H = 112       # A-004
HAND_CROP_W = 112


DUMMY_DRAFT_LEN = 8     # Stage C dummy 한국어 토큰 시퀀스 길이


class DummyDataset(Dataset):
    """더미 KSL 데이터셋.

    Args:
        split_group: "train", "valid", "test"
        num_samples: 전체 생성 샘플 수 (split 전)
        seed: 난수 시드
        max_len: 시계열 최대 길이 (패딩 기준)
        decoder_vocab_size: 한국어 decoder vocab 크기 (tgt_tokens 범위에 사용)
        decoder_bos_id: BOS 토큰 ID
        decoder_eos_id: EOS 토큰 ID
        decoder_pad_id: PAD 토큰 ID
    """

    def __init__(
        self,
        split_group: str = "train",
        num_samples: int = 20,
        seed: int = 42,
        max_len: int = T_DEFAULT,
        decoder_vocab_size: int = 32000,
        decoder_bos_id: int = 0,
        decoder_eos_id: int = 2,
        decoder_pad_id: int = 1,
    ) -> None:
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_bos_id = decoder_bos_id
        self.decoder_eos_id = decoder_eos_id
        self.decoder_pad_id = decoder_pad_id
        adapter = DummyAdapter(num_samples=num_samples, seed=seed)
        all_samples = list(adapter.iter_samples())
        split_samples, self.manifest = make_signer_independent_split(
            all_samples, dataset_name="dummy", seed=seed
        )
        self.samples: list[KSLSample] = [
            s for s in split_samples if s.split_group == split_group
        ]
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        # 특수토큰 범위(0~4)를 피해 일반 토큰 범위에서 샘플링하기 위한 최솟값
        self._tok_start = max(5, self.decoder_bos_id + 1, self.decoder_eos_id + 1,
                              self.decoder_pad_id + 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        T = min(sample.num_frames, self.max_len)

        # ── 랜드마크 텐서 (정규화된 [0,1] 범위 난수) ────────────────────────
        pose = torch.from_numpy(
            self.rng.random((T, POSE_JOINTS, 3), dtype=np.float32)
        )
        left_hand = torch.from_numpy(
            self.rng.random((T, HAND_JOINTS, 3), dtype=np.float32)
        )
        right_hand = torch.from_numpy(
            self.rng.random((T, HAND_JOINTS, 3), dtype=np.float32)
        )
        face_blendshape = torch.from_numpy(
            self.rng.random((T, FACE_BLENDSHAPE), dtype=np.float32)
        )
        face_key_subset = torch.from_numpy(
            self.rng.random((T, FACE_KEY_SUBSET, 3), dtype=np.float32)
        )
        presence_mask = torch.ones(T, 4, dtype=torch.bool)  # [T, streams]

        # ── hand crop 시퀀스 (dummy RGB 이미지) ──────────────────────────────
        left_hand_crop = torch.from_numpy(
            self.rng.random((T, 3, HAND_CROP_H, HAND_CROP_W), dtype=np.float32)
        )
        right_hand_crop = torch.from_numpy(
            self.rng.random((T, 3, HAND_CROP_H, HAND_CROP_W), dtype=np.float32)
        )

        # ── 라벨 ──────────────────────────────────────────────────────────────
        # gloss: token index 시퀀스 (dummy vocab size=50)
        gloss_len = len(sample.gloss_tokens) if sample.gloss_tokens else 1
        gloss_ids = torch.randint(1, 50, (gloss_len,))

        # intent: 도메인 인덱스 (A-007)
        domain_list = ["hospital", "directions", "order", "reservation", "public", "help", "unknown"]
        intent_label = torch.tensor(domain_list.index(sample.domain) if sample.domain in domain_list else 6)

        # NMS: 얼굴표정/머리움직임/시선 등 비수지 신호 multi-label
        nms_label, nms_mask = encode_nms_labels(sample.nms_labels)
        nms_detail_label, nms_detail_mask = encode_nms_detail_labels(sample.nms_labels)

        # activity: 1 = 수어 진행 중 (dummy에서는 전체 구간이 수어)
        activity = torch.ones(T, dtype=torch.long)
        if T > 0:
            activity[0] = 2
            activity[-1] = 2

        # ── Stage C: 한국어 draft 토큰 (teacher-forcing용) ────────────────
        # tgt_tokens  = [BOS, tok1, ..., tokN]          (decoder 입력)
        # draft_labels = [tok1, ..., tokN, EOS]          (decoder 정답, CE loss)
        body = torch.randint(self._tok_start, self.decoder_vocab_size, (DUMMY_DRAFT_LEN - 1,))
        tgt_tokens = torch.cat([torch.tensor([self.decoder_bos_id]), body])   # [DUMMY_DRAFT_LEN]
        draft_labels = torch.cat([body, torch.tensor([self.decoder_eos_id])]) # [DUMMY_DRAFT_LEN]

        return {
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "split_group": sample.split_group,
            # 랜드마크
            "pose": pose,                          # [T, 25, 3]
            "left_hand": left_hand,                # [T, 21, 3]
            "right_hand": right_hand,              # [T, 21, 3]
            "face_blendshape": face_blendshape,    # [T, 52]
            "face_key_subset": face_key_subset,    # [T, 68, 3]
            "presence_mask": presence_mask,        # [T, 4]
            "source_frame_idx": torch.arange(T, dtype=torch.long),
            # 이미지 crop
            "left_hand_crop": left_hand_crop,      # [T, 3, 112, 112]
            "right_hand_crop": right_hand_crop,    # [T, 3, 112, 112]
            # 라벨
            "gloss_ids": gloss_ids,                # [G]
            "nms_label": nms_label,                # [12]
            "nms_mask": nms_mask,                  # [12]
            "nms_detail_label": nms_detail_label,  # [5]
            "nms_detail_mask": nms_detail_mask,    # [5]
            "intent_label": intent_label,          # scalar
            "activity": activity,                  # [T]
            "seq_len": torch.tensor(T),
            # Stage C: 한국어 draft 토큰
            "tgt_tokens": tgt_tokens,              # [L]  decoder 입력 (BOS + body)
            "draft_labels": draft_labels,          # [L]  CE loss 정답 (body + EOS)
        }


def make_dummy_batch(batch_size: int = 2, max_len: int = T_DEFAULT) -> dict[str, Any]:
    """Collator 없이 단일 배치를 바로 만드는 유틸리티 (smoke test용)."""
    dataset = DummyDataset(split_group="train", num_samples=20, max_len=max_len)
    items = [dataset[i % len(dataset)] for i in range(batch_size)]
    return collate_fn(items)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """가변 길이 시퀀스를 최대 길이로 패딩해 배치를 만든다."""
    keys_tensor_seq = [
        "pose", "left_hand", "right_hand", "face_blendshape",
        "face_key_subset", "presence_mask",
        "left_hand_crop", "right_hand_crop", "activity", "source_frame_idx",
    ]
    keys_tensor_fixed = [
        "intent_label",
        "seq_len",
        "nms_label",
        "nms_mask",
        "nms_detail_label",
        "nms_detail_mask",
    ]
    keys_str = [
        "sample_id",
        "domain",
        "split_group",
        "korean_text",
        "gloss_tokens_raw",
        "nms_labels_raw",
    ]

    max_t = max(item["seq_len"].item() for item in batch)

    collated: dict[str, Any] = {}

    for key in keys_str:
        if any(key in item for item in batch):
            collated[key] = [item.get(key) for item in batch]

    for key in keys_tensor_fixed:
        collated[key] = torch.stack([item[key] for item in batch])

    for key in keys_tensor_seq:
        if not any(key in item for item in batch):
            continue
        template = next(item[key] for item in batch if key in item)
        tensors = []
        for item in batch:
            if key in item:
                tensors.append(item[key])
            else:
                seq_len = int(item["seq_len"].item())
                tensors.append(torch.zeros((seq_len,) + template.shape[1:], dtype=template.dtype))
        padded = []
        for t in tensors:
            pad_len = max_t - t.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + t.shape[1:]
                t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
            padded.append(t)
        collated[key] = torch.stack(padded)

    # gloss: ragged → list of tensors (CTC는 배치별 다른 길이 허용)
    collated["gloss_ids"] = [item["gloss_ids"] for item in batch]

    # ── Stage C: tgt_tokens / draft_labels 패딩 ───────────────────────────
    # tgt_tokens: PAD=0으로 패딩, tgt_padding mask(True=pad) 생성
    # draft_labels: CE ignore_index=-100으로 패딩
    if "tgt_tokens" in batch[0]:
        tgt_list = [item["tgt_tokens"] for item in batch]
        lbl_list = [item["draft_labels"] for item in batch]
        max_l = max(t.shape[0] for t in tgt_list)

        tgt_padded, lbl_padded, tgt_padding = [], [], []
        for tgt, lbl in zip(tgt_list, lbl_list):
            pad_len = max_l - tgt.shape[0]
            if pad_len > 0:
                tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=tgt.dtype)])
                lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
            tgt_padded.append(tgt)
            lbl_padded.append(lbl)
            tgt_padding.append(
                torch.cat([torch.zeros(tgt.shape[0] - pad_len, dtype=torch.bool),
                           torch.ones(pad_len, dtype=torch.bool)])
            )
        collated["tgt_tokens"] = torch.stack(tgt_padded)    # [B, L] int
        collated["draft_labels"] = torch.stack(lbl_padded)  # [B, L] int (-100=ignore)
        collated["tgt_padding"] = torch.stack(tgt_padding)  # [B, L] bool

    return collated
