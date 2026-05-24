"""Fusion module.

여러 스트림 특징을 합치는 모듈.
Late fusion (concat + projection) 또는 Cross-attention 방식을 지원한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class FusionConfig:
    method: str = "late"              # "late" | "cross_attention"
    stream_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    d_model: int = 256
    nhead: int = 4
    dropout: float = 0.1


class LateFusion(nn.Module):
    """단순 concat + projection."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        in_dim = sum(config.stream_dims)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            streams: list of [B, T, d_i]

        Returns:
            [B, T, d_model]
        """
        x = torch.cat(streams, dim=-1)
        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """Landmark stream을 query로, 다른 stream들을 key/value로 cross-attention."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        # 각 스트림을 d_model로 맞춤
        self.projs = nn.ModuleList([
            nn.Linear(d, config.d_model) for d in config.stream_dims
        ])
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            streams: list of [B, T, d_i], 첫 번째가 query (landmark)

        Returns:
            [B, T, d_model]
        """
        projected = [proj(s) for proj, s in zip(self.projs, streams)]
        query = projected[0]
        # key/value: 나머지 스트림을 concat해 시퀀스 길이 방향으로 쌓음
        if len(projected) > 1:
            kv = torch.cat(projected[1:], dim=1)   # [B, T*(N-1), d]
        else:
            kv = query
        attn_out, _ = self.attn(query, kv, kv)
        return self.norm(query + attn_out)


class FusionModule(nn.Module):
    """다중 스트림 융합 모듈."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionConfig()
        if self.config.method == "cross_attention":
            self.impl = CrossAttentionFusion(self.config)
        else:
            self.impl = LateFusion(self.config)

    def forward(self, streams: list[torch.Tensor]) -> torch.Tensor:
        return self.impl(streams)
