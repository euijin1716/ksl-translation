"""KSL Stage C model.

Stage C is the only supported model path. It always uses landmark, hand visual,
face expression, fusion, recognition heads, and the Korean draft decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .decoder import DecoderConfig, KoreanDraftDecoder
from .fusion import FusionConfig, FusionModule
from .heads import AllHeads, HeadsConfig
from .streams.face_expr_encoder import FaceExprEncoder, FaceExprEncoderConfig
from .streams.hand_visual_encoder import HandVisualEncoder, HandVisualEncoderConfig
from .streams.landmark_encoder import LandmarkEncoder, LandmarkEncoderConfig


@dataclass
class ModelConfig:
    stage: str = "C"
    landmark: LandmarkEncoderConfig = field(default_factory=LandmarkEncoderConfig)
    hand_visual: HandVisualEncoderConfig = field(default_factory=HandVisualEncoderConfig)
    face_expr: FaceExprEncoderConfig = field(default_factory=FaceExprEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


class KSLModel(nn.Module):
    """한국수어 인식/번역 통합 모델.

    forward() 반환값:
    {
        "gloss_logits":    [B, T, vocab],     # 항상
        "nms_logits":      [B, T, nms_cls],  # 항상
        "intent_logits":   [B, intent_cls],  # 항상
        "boundary_logits": [B, T, bnd_cls],  # 항상
        "draft_logits":    [B, L, vocab],    # teacher-forcing
        "draft_tokens":    [B, L],           # greedy inference
        "fused":           [B, T, d_model],  # downstream 사용 가능
    }
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.stage != "C":
            raise ValueError("Only Stage C is supported.")

        # ── 스트림 인코더 ────────────────────────────────────────────────────
        self.landmark_encoder = LandmarkEncoder(c.landmark)

        self.hand_encoder: HandVisualEncoder = HandVisualEncoder(c.hand_visual)

        self.face_encoder: FaceExprEncoder = FaceExprEncoder(c.face_expr)

        # ── Fusion ──────────────────────────────────────────────────────────
        # stream_dims은 실제로 사용하는 스트림에 맞게 설정
        stream_dims = [c.landmark.d_model]
        stream_dims.append(c.hand_visual.d_model)
        stream_dims.append(c.face_expr.d_model)

        fusion_cfg = FusionConfig(
            method=c.fusion.method,
            stream_dims=stream_dims,
            d_model=c.heads.d_model,
            nhead=c.fusion.nhead,
            dropout=c.fusion.dropout,
        )
        self.fusion = FusionModule(fusion_cfg)

        # ── Heads ────────────────────────────────────────────────────────────
        self.heads = AllHeads(c.heads)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder: KoreanDraftDecoder = KoreanDraftDecoder(c.decoder)

    def forward(
        self,
        pose: torch.Tensor,
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
        face_blendshape: torch.Tensor,
        face_key_subset: torch.Tensor | None = None,
        presence_mask: torch.Tensor | None = None,
        left_hand_crop: torch.Tensor | None = None,
        right_hand_crop: torch.Tensor | None = None,
        tgt_tokens: torch.Tensor | None = None,
        tgt_padding: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pose:              [B, T, pose_joints, 3]
            left_hand:         [B, T, 21, 3]
            right_hand:        [B, T, 21, 3]
            face_blendshape:   [B, T, 52]
            face_key_subset:   [B, T, 68, 3] or None
            presence_mask:     [B, T, 4] or None
            left_hand_crop:    [B, T, 3, H, W] or None
            right_hand_crop:   [B, T, 3, H, W] or None
            tgt_tokens:        [B, L] int  (teacher-forcing)
            tgt_padding:       [B, L] bool
            src_key_padding_mask: [B, T] bool

        Returns:
            dict (see class docstring)
        """
        B = pose.shape[0]
        T = pose.shape[1]

        # ── Landmark stream ──────────────────────────────────────────────────
        lm_feat = self.landmark_encoder(
            pose, left_hand, right_hand, face_blendshape,
            face_key_subset, presence_mask, src_key_padding_mask,
        )   # [B, T, d_model]

        streams = [lm_feat]

        # ── Hand visual stream ─────────────────────────────────────────────
        if left_hand_crop is not None and right_hand_crop is not None:
            hand_feat = self.hand_encoder(
                left_hand_crop, right_hand_crop, src_key_padding_mask
            )
        else:
            # Keep fusion shape stable when crop tensors are unavailable.
            hand_feat = torch.zeros(
                B, T, self.config.hand_visual.d_model, device=pose.device
            )
        streams.append(hand_feat)

        # ── Face expression stream ─────────────────────────────────────────
        face_feat = self.face_encoder(face_blendshape, src_key_padding_mask)
        streams.append(face_feat)

        # ── Fusion ──────────────────────────────────────────────────────────
        fused = self.fusion(streams, src_key_padding_mask)   # [B, T, d_model]

        # ── Heads ────────────────────────────────────────────────────────────
        head_outputs = self.heads(fused, src_key_padding_mask)

        out: dict[str, Any] = {**head_outputs, "fused": fused}

        # ── Decoder ───────────────────────────────────────────────────────
        if tgt_tokens is not None:
            out["draft_logits"] = self.decoder(
                fused, tgt_tokens, tgt_padding, src_key_padding_mask
            )
        else:
            out["draft_tokens"] = self.decoder.greedy_decode(
                fused, src_key_padding_mask
            )

        return out
