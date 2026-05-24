"""Hand visual stream encoder.

손 crop 이미지 시퀀스를 CNN + Temporal Transformer로 인코딩한다.
왼손/오른손을 각각 독립적으로 처리한 뒤 concat한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HandVisualEncoderConfig:
    cnn_out_dim: int = 128          # CNN backbone 출력 채널 수
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512


class SimpleCNNBackbone(nn.Module):
    """경량 CNN backbone (smoke test 및 Stage C baseline용)."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 56x56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                         # 28x28
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HandVisualEncoder(nn.Module):
    """Hand crop 시퀀스 인코더.

    입력:
        left_hand_crop:  [B, T, 3, H, W]
        right_hand_crop: [B, T, 3, H, W]

    출력:
        features: [B, T, d_model]
    """

    def __init__(self, config: HandVisualEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or HandVisualEncoderConfig()
        c = self.config

        self.left_cnn = SimpleCNNBackbone(c.cnn_out_dim)
        self.right_cnn = SimpleCNNBackbone(c.cnn_out_dim)

        self.proj = nn.Linear(c.cnn_out_dim * 2, c.d_model)
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
        left_hand_crop: torch.Tensor,
        right_hand_crop: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            left_hand_crop:  [B, T, 3, H, W]
            right_hand_crop: [B, T, 3, H, W]

        Returns:
            [B, T, d_model]
        """
        B, T, C, H, W = left_hand_crop.shape

        # 각 프레임을 CNN으로 처리
        left_feat = self.left_cnn(left_hand_crop.view(B * T, C, H, W)).view(B, T, -1)
        right_feat = self.right_cnn(right_hand_crop.view(B * T, C, H, W)).view(B, T, -1)

        x = torch.cat([left_feat, right_feat], dim=-1)  # [B, T, cnn_out*2]
        x = self.proj(x)                                 # [B, T, d_model]

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_enc(pos)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x
