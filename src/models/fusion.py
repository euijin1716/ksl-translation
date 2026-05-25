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

    def forward(
        self,
        streams: list[torch.Tensor],
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            streams: list of [B, T, d_i]
            src_key_padding_mask: 사용 안 함 (per-frame concat이라 프레임 간 혼합 없음). 인터페이스 통일용.

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

    def forward(
        self,
        streams: list[torch.Tensor],
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            streams: list of [B, T, d_i], 첫 번째가 query (landmark)
            src_key_padding_mask: [B, T] bool, True=pad. KV에 쌓이는 패딩 프레임이
                attention되지 않도록 KV 스트림 수만큼 time축으로 복제해 전달한다.

        Returns:
            [B, T, d_model]
        """
        projected = [proj(s) for proj, s in zip(self.projs, streams)]
        query = projected[0]
        # key/value: 나머지 스트림을 concat해 시퀀스 길이 방향으로 쌓음
        if len(projected) > 1:
            kv = torch.cat(projected[1:], dim=1)   # [B, T*(N-1), d]
            if src_key_padding_mask is not None:
                kv_mask = src_key_padding_mask.repeat(1, len(projected) - 1)  # [B, T*(N-1)]
            else:
                kv_mask = None
        else:
            kv = query
            kv_mask = src_key_padding_mask
        attn_out, _ = self.attn(query, kv, kv, key_padding_mask=kv_mask)
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

    def forward(
        self,
        streams: list[torch.Tensor],
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.impl(streams, src_key_padding_mask)
