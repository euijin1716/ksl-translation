"""Landmark stream encoder.

pose + left_hand + right_hand + face_key_subset를
단일 시계열로 flatten하고 Transformer Encoder로 인코딩한다.
표정 강도(face_blendshape)는 E3(FaceExprEncoder)가 전담하므로 여기서는 쓰지 않는다.

Stage C의 landmark stream으로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .manual_features import MANUAL_FEATURE_DIM, extract_manual_signal_features


@dataclass
class LandmarkEncoderConfig:
    pose_joints: int = 25
    hand_joints: int = 21
    face_blendshape_dim: int = 52
    face_key_joints: int = 68
    coord_dim: int = 3           # xyz
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    manual_feature_dim: int = MANUAL_FEATURE_DIM


class LandmarkEncoder(nn.Module):
    """랜드마크 스트림 인코더.

    입력:
        pose:             [B, T, pose_joints, 3]
        left_hand:        [B, T, hand_joints, 3]
        right_hand:       [B, T, hand_joints, 3]
        face_blendshape:  [B, T, blendshape_dim]
        face_key_subset:  [B, T, face_key_joints, 3]  (선택)
        presence_mask:    [B, T, 4]                   (선택)

    출력:
        features: [B, T, d_model]
    """

    def __init__(self, config: LandmarkEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LandmarkEncoderConfig()
        c = self.config

        # 입력 차원 계산
        pose_dim = c.pose_joints * c.coord_dim
        hand_dim = c.hand_joints * c.coord_dim
        face_key_dim = c.face_key_joints * c.coord_dim
        in_dim = pose_dim + hand_dim * 2 + face_key_dim + 4  # +4 = presence_mask

        self.input_proj = nn.Linear(in_dim, c.d_model)
        self.manual_proj = nn.Sequential(
            nn.Linear(c.manual_feature_dim, c.d_model),
            nn.LayerNorm(c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.pos_enc = nn.Embedding(c.max_seq_len, c.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=c.num_layers)

    def forward(
        self,
        pose: torch.Tensor,
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
        face_blendshape: torch.Tensor | None = None,  # E3 전담, E1 입력에는 사용 안 함
        face_key_subset: torch.Tensor | None = None,
        presence_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pose:            [B, T, pose_joints, 3]
            left_hand:       [B, T, hand_joints, 3]
            right_hand:      [B, T, hand_joints, 3]
            face_blendshape: 사용 안 함 (E3 전담). 호출부 호환을 위해 시그니처만 유지.
            face_key_subset: [B, T, face_key_joints, 3] or None
            presence_mask:   [B, T, 4] bool or float, or None
            src_key_padding_mask: [B, T] bool True=pad

        Returns:
            [B, T, d_model]
        """
        B, T = pose.shape[:2]
        c = self.config

        parts = [
            pose.reshape(B, T, -1),
            left_hand.reshape(B, T, -1),
            right_hand.reshape(B, T, -1),
        ]

        if face_key_subset is not None:
            parts.append(face_key_subset.reshape(B, T, -1))
        else:
            parts.append(torch.zeros(B, T, c.face_key_joints * c.coord_dim, device=pose.device))

        if presence_mask is not None:
            parts.append(presence_mask.float())
        else:
            parts.append(torch.ones(B, T, 4, device=pose.device))

        x = torch.cat(parts, dim=-1)           # [B, T, in_dim]
        x = self.input_proj(x)                 # [B, T, d_model]
        manual_features = extract_manual_signal_features(left_hand, right_hand)
        x = x + self.manual_proj(manual_features)

        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        x = x + self.pos_enc(pos)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x
