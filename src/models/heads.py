"""Recognition heads for the Stage C KSL model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadsConfig:
    d_model: int = 256
    gloss_vocab_size: int = 1001
    nms_num_classes: int = 12
    intent_num_classes: int = 7
    boundary_num_classes: int = 3
    nms_eyebrow_classes: int = 4
    nms_eye_classes: int = 4
    nms_mouth_shape_classes: int = 11
    nms_head_movement_classes: int = 5
    nms_gaze_direction_classes: int = 7
    dropout: float = 0.1


class GlossHead(nn.Module):
    """CTC gloss recognition head."""

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.gloss_vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NMSHead(nn.Module):
    """Coarse multi-label non-manual signal head."""

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.nms_num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class NMSDetailHeads(nn.Module):
    """Fine categorical non-manual signal heads."""

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                "eyebrow": _frame_head(config, config.nms_eyebrow_classes),
                "eye": _frame_head(config, config.nms_eye_classes),
                "mouth_shape": _frame_head(config, config.nms_mouth_shape_classes),
                "head_movement": _frame_head(config, config.nms_head_movement_classes),
                "gaze_direction": _frame_head(config, config.nms_gaze_direction_classes),
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {f"nms_{name}_logits": head(x) for name, head in self.heads.items()}


class IntentHead(nn.Module):
    """Utterance-level domain/intent head."""

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.intent_num_classes),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)
        return self.proj(pooled)


class BoundaryHead(nn.Module):
    """Frame-level signing activity/boundary head."""

    def __init__(self, config: HeadsConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.boundary_num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AllHeads(nn.Module):
    """Container for all recognition heads."""

    def __init__(self, config: HeadsConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadsConfig()
        self.gloss = GlossHead(self.config)
        self.nms = NMSHead(self.config)
        self.nms_detail = NMSDetailHeads(self.config)
        self.intent = IntentHead(self.config)
        self.boundary = BoundaryHead(self.config)

    def forward(
        self,
        fused: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "gloss_logits": self.gloss(fused),
            "nms_logits": self.nms(fused),
            **self.nms_detail(fused),
            "intent_logits": self.intent(fused, padding_mask),
            "boundary_logits": self.boundary(fused),
        }


def _frame_head(config: HeadsConfig, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(config.d_model, num_classes),
    )
