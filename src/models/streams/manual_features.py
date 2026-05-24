"""Manual signal feature extraction from hand landmarks.

The raw hand landmarks are still passed to the Transformer. These engineered
features make handshape, movement, and trajectory cues explicit so the Stage C
model does not need to infer all of them from flattened coordinates alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MANUAL_FEATURE_DIM = 76

_TIP = [4, 8, 12, 16, 20]
_MCP = [1, 5, 9, 13, 17]
_ANGLE_TRIPLES = [(1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]


def extract_manual_signal_features(
    left_hand: torch.Tensor,
    right_hand: torch.Tensor,
) -> torch.Tensor:
    """Return explicit handshape, movement, and trajectory features.

    Args:
        left_hand:  [B, T, 21, 3]
        right_hand: [B, T, 21, 3]

    Returns:
        [B, T, 76] feature tensor.
    """
    left = _per_hand_features(left_hand)
    right = _per_hand_features(right_hand)

    left_center = left_hand.mean(dim=2)
    right_center = right_hand.mean(dim=2)
    rel_center = right_center - left_center
    rel_dist = rel_center.norm(dim=-1, keepdim=True)
    rel_velocity = _velocity(rel_center)
    rel_speed = _velocity(rel_dist)

    return torch.cat([left, right, rel_center, rel_dist, rel_velocity, rel_speed], dim=-1)


def _per_hand_features(hand: torch.Tensor) -> torch.Tensor:
    wrist = hand[:, :, 0, :]
    center = hand.mean(dim=2)
    scale = (hand[:, :, 9, :] - wrist).norm(dim=-1, keepdim=True).clamp(min=1e-4)

    tip = hand[:, :, _TIP, :]
    mcp = hand[:, :, _MCP, :]

    tip_dist = (tip - wrist.unsqueeze(2)).norm(dim=-1) / scale
    mcp_dist = (mcp - wrist.unsqueeze(2)).norm(dim=-1).clamp(min=1e-4)
    extension = (tip - wrist.unsqueeze(2)).norm(dim=-1) / mcp_dist
    curl = torch.stack([_joint_angle(hand, *triple) for triple in _ANGLE_TRIPLES], dim=-1)
    spread = (tip[:, :, 1:, :] - tip[:, :, :-1, :]).norm(dim=-1) / scale

    palm_a = hand[:, :, 5, :] - wrist
    palm_b = hand[:, :, 17, :] - wrist
    palm_normal = F.normalize(torch.cross(palm_a, palm_b, dim=-1), dim=-1, eps=1e-6)

    wrist_vel = _velocity(wrist)
    wrist_speed = wrist_vel.norm(dim=-1, keepdim=True)
    center_vel = _velocity(center)
    center_speed = center_vel.norm(dim=-1, keepdim=True)
    trajectory = wrist - wrist[:, :1, :]
    trajectory_dist = trajectory.norm(dim=-1, keepdim=True)

    return torch.cat(
        [
            tip_dist,
            extension,
            curl,
            spread,
            palm_normal,
            wrist_vel,
            wrist_speed,
            center_vel,
            center_speed,
            trajectory,
            trajectory_dist,
        ],
        dim=-1,
    )


def _joint_angle(hand: torch.Tensor, a_idx: int, b_idx: int, c_idx: int) -> torch.Tensor:
    a = hand[:, :, a_idx, :] - hand[:, :, b_idx, :]
    c = hand[:, :, c_idx, :] - hand[:, :, b_idx, :]
    a = F.normalize(a, dim=-1, eps=1e-6)
    c = F.normalize(c, dim=-1, eps=1e-6)
    return (a * c).sum(dim=-1)


def _velocity(x: torch.Tensor) -> torch.Tensor:
    first = torch.zeros_like(x[:, :1])
    return torch.cat([first, x[:, 1:] - x[:, :-1]], dim=1)
