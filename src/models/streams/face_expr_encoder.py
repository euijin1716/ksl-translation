"""Face expression / NMS stream encoder.

face blendshape 또는 face crop을 입력으로 받아
표정/비수지신호 특징을 인코딩한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FaceExprEncoderConfig:
    blendshape_dim: int = 52
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 512


class FaceExprEncoder(nn.Module):
    """Face expression 스트림 인코더.

    입력:
        face_blendshape: [B, T, blendshape_dim]

    출력:
        features: [B, T, d_model]
    """

    def __init__(self, config: FaceExprEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or FaceExprEncoderConfig()
        c = self.config

        self.input_proj = nn.Linear(c.blendshape_dim, c.d_model)
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
        face_blendshape: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            face_blendshape: [B, T, blendshape_dim]

        Returns:
            [B, T, d_model]
        """
        B, T, _ = face_blendshape.shape
        x = self.input_proj(face_blendshape)   # [B, T, d_model]

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_enc(pos)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x
