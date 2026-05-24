"""Signal label helpers.

Non-manual signals are supervised at two levels:
- coarse multi-label targets for robust detection
- fine categorical targets for expression, mouth shape, gaze, and head motion
"""

from __future__ import annotations

from typing import Any

import torch

from .schema import NMSLabels


NMS_KEYS: list[str] = [
    "eyebrow_raise",
    "eyebrow_furrow",
    "eye_wide",
    "eye_squint",
    "nose_wrinkle",
    "mouth_open",
    "mouth_shape",
    "cheek_puff",
    "head_nod",
    "head_shake",
    "head_tilt",
    "gaze_direction",
]

NMS_DETAIL_GROUPS: list[str] = [
    "eyebrow",
    "eye",
    "mouth_shape",
    "head_movement",
    "gaze_direction",
]

NMS_DETAIL_CLASSES: dict[str, list[str]] = {
    "eyebrow": ["neutral", "raise", "furrow", "both"],
    "eye": ["neutral", "wide", "squint", "both"],
    "mouth_shape": [
        "neutral",
        "open",
        "a",
        "i",
        "u",
        "e",
        "o",
        "round",
        "spread",
        "closed",
        "other",
    ],
    "head_movement": ["neutral", "nod", "shake", "tilt", "complex"],
    "gaze_direction": ["forward", "left", "right", "up", "down", "away", "other"],
}

_NEUTRAL_VALUES = {"", "none", "neutral", "forward", "false", "0", "no"}


def encode_nms_labels(
    nms_labels: NMSLabels | dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode coarse NMS labels into target and mask tensors.

    Returns:
        values: [len(NMS_KEYS)] float tensor with 0/1 targets.
        mask:   [len(NMS_KEYS)] float tensor. 1 means the field is annotated.
    """
    values = torch.zeros(len(NMS_KEYS), dtype=torch.float32)
    mask = torch.zeros(len(NMS_KEYS), dtype=torch.float32)
    if nms_labels is None:
        return values, mask

    data = _as_dict(nms_labels)

    for i, key in enumerate(NMS_KEYS):
        if key not in data or data[key] is None:
            continue
        raw = data[key]
        mask[i] = 1.0
        if isinstance(raw, bool):
            values[i] = 1.0 if raw else 0.0
        elif isinstance(raw, (int, float)):
            values[i] = 1.0 if float(raw) > 0 else 0.0
        else:
            values[i] = 0.0 if _norm(raw) in _NEUTRAL_VALUES else 1.0

    return values, mask


def encode_nms_detail_labels(
    nms_labels: NMSLabels | dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode fine-grained NMS categorical labels.

    Returns:
        labels: [len(NMS_DETAIL_GROUPS)] long tensor.
        mask:   [len(NMS_DETAIL_GROUPS)] float tensor. 1 means supervised.
    """
    labels = torch.zeros(len(NMS_DETAIL_GROUPS), dtype=torch.long)
    mask = torch.zeros(len(NMS_DETAIL_GROUPS), dtype=torch.float32)
    if nms_labels is None:
        return labels, mask

    data = _as_dict(nms_labels)
    detail = {
        "eyebrow": _eyebrow_class(data),
        "eye": _eye_class(data),
        "mouth_shape": _mouth_shape_class(data),
        "head_movement": _head_movement_class(data),
        "gaze_direction": _gaze_class(data),
    }
    annotated = {
        "eyebrow": _has_any(data, "eyebrow_raise", "eyebrow_furrow"),
        "eye": _has_any(data, "eye_wide", "eye_squint"),
        "mouth_shape": _has_any(data, "mouth_shape", "mouth_open"),
        "head_movement": _has_any(data, "head_nod", "head_shake", "head_tilt"),
        "gaze_direction": _has_any(data, "gaze_direction"),
    }

    for i, group in enumerate(NMS_DETAIL_GROUPS):
        classes = NMS_DETAIL_CLASSES[group]
        labels[i] = classes.index(detail[group])
        mask[i] = 1.0 if annotated[group] else 0.0

    return labels, mask


def _as_dict(nms_labels: NMSLabels | dict[str, Any]) -> dict[str, Any]:
    return nms_labels.to_dict() if isinstance(nms_labels, NMSLabels) else dict(nms_labels)


def _has_any(data: dict[str, Any], *keys: str) -> bool:
    return any(k in data and data[k] is not None for k in keys)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0
    if value is None:
        return False
    return _norm(value) not in _NEUTRAL_VALUES


def _eyebrow_class(data: dict[str, Any]) -> str:
    raise_ = _truthy(data.get("eyebrow_raise"))
    furrow = _truthy(data.get("eyebrow_furrow"))
    if raise_ and furrow:
        return "both"
    if raise_:
        return "raise"
    if furrow:
        return "furrow"
    return "neutral"


def _eye_class(data: dict[str, Any]) -> str:
    wide = _truthy(data.get("eye_wide"))
    squint = _truthy(data.get("eye_squint"))
    if wide and squint:
        return "both"
    if wide:
        return "wide"
    if squint:
        return "squint"
    return "neutral"


def _mouth_shape_class(data: dict[str, Any]) -> str:
    raw = data.get("mouth_shape")
    if raw is None and _truthy(data.get("mouth_open")):
        return "open"

    value = _norm(raw)
    if value in {"", "none", "neutral"}:
        return "neutral"
    if value in {"closed", "close", "shut", "\ub2e4\ubb38", "\ub2e4\ubb3c\uae30"}:
        return "closed"
    if value in {"open", "mouth_open", "mo1", "\uc785\ubc8c\ub9bc", "\ubc8c\ub9bc"}:
        return "open"
    if value in {"a", "ah", "\uc544"}:
        return "a"
    if value in {"i", "ee", "smile", "\uc774"}:
        return "i"
    if value in {"u", "oo", "\uc6b0"}:
        return "u"
    if value in {"e", "eh", "\uc5d0", "\uc560"}:
        return "e"
    if value in {"o", "oh", "\uc624"}:
        return "o"
    if value in {"round", "rounded", "pucker", "lip_round", "\uc624\ubbc0\ub9bc", "\ub465\uae00\uac8c"}:
        return "round"
    if value in {"spread", "wide", "stretch", "lip_spread", "\uc606\uc73c\ub85c"}:
        return "spread"
    return "other"


def _head_movement_class(data: dict[str, Any]) -> str:
    active = [
        name
        for name, key in (("nod", "head_nod"), ("shake", "head_shake"), ("tilt", "head_tilt"))
        if _truthy(data.get(key))
    ]
    if len(active) == 0:
        return "neutral"
    if len(active) == 1:
        return active[0]
    return "complex"


def _gaze_class(data: dict[str, Any]) -> str:
    value = _norm(data.get("gaze_direction"))
    if value in {"", "none", "neutral", "center", "front", "forward", "\uc815\uba74"}:
        return "forward"
    if value in {"left", "l", "\uc67c\ucabd"}:
        return "left"
    if value in {"right", "r", "\uc624\ub978\ucabd"}:
        return "right"
    if value in {"up", "u", "\uc704"}:
        return "up"
    if value in {"down", "d", "\uc544\ub798"}:
        return "down"
    if value in {"away", "side", "averted"}:
        return "away"
    return "other"


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()
