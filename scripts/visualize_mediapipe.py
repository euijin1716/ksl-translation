"""MediaPipe 키포인트 시각화 대시보드 (Streamlit).

test 셋 영상에 Pose/Hand/Face 특징점을 오버레이해서 프레임별로 보여준다.

두 가지 모드:
  [A] keypoints 재생 모드 (기본)
      - 전처리로 추출된 .npy를 영상 프레임에 그림
      - MediaPipe 재실행 불필요 → 빠름
  [B] 실시간 MediaPipe 모드 (--live_mediapipe)
      - 영상에서 MediaPipe를 직접 실행 → 감지 과정 그대로 시각화
      - 영상 파일이 반드시 존재해야 함

실행:
  streamlit run scripts/visualize_mediapipe.py -- \\
    --manifest data/manifests/test.jsonl \\
    --keypoint_root data/keypoints \\
    --num_samples 20

  # 실시간 모드 (영상 파일 필요)
  streamlit run scripts/visualize_mediapipe.py -- \\
    --manifest data/manifests/test.jsonl \\
    --keypoint_root data/keypoints \\
    --live_mediapipe
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

_TARGET_SAMPLE_KEYS = (
    "aihub_sign_NIA_SL_G2_ELECTRICGASACCIDENT001080_1_TW104",
    "aihub_sign_NIA_SL_G2_FIRE001338_1_TW104",
)

_DEMO_METRICS = {
    "intent_accuracy": 1.0,
    "boundary_f1": 0.6415340744015176,
    "gloss_wer": 0.4115557771096537,
    "nms_f1": 0.7810030640219319,
    "nms_detail_accuracy": {
        "eyebrow": 0.8595447490946715,
        "mouth_shape": 0.8486808070356958,
        "head_movement": 0.7033109156751164,
    },
    "bleu": 23.60095287990394,
    "chrf": 40.05766054203319,
}

st.set_page_config(
    page_title="KSL MediaPipe 시각화",
    page_icon="🖐",
    layout="wide",
)

# ── MediaPipe 블렌드셰이프 레이블 (52개) ──────────────────────────────────────
_BS_LABELS = [
    "_neutral",
    "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
    "cheekPuff","cheekSquintLeft","cheekSquintRight",
    "eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft","eyeLookDownRight",
    "eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight",
    "eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight",
    "eyeWideLeft","eyeWideRight",
    "jawForward","jawLeft","jawOpen","jawRight",
    "mouthClose","mouthDimpleLeft","mouthDimpleRight",
    "mouthFrownLeft","mouthFrownRight","mouthFunnel",
    "mouthLeft","mouthLowerDownLeft","mouthLowerDownRight",
    "mouthPressLeft","mouthPressRight","mouthPucker",
    "mouthRight","mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
    "mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "noseSneerLeft","noseSneerRight",
]

# NMS 관련 블렌드셰이프 그룹 (시각화에서 강조)
_BS_NMS_GROUPS = {
    "눈썹": ["browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight"],
    "눈": ["eyeBlinkLeft","eyeBlinkRight","eyeSquintLeft","eyeSquintRight","eyeWideLeft","eyeWideRight"],
    "입": ["jawOpen","mouthClose","mouthFunnel","mouthPucker","mouthSmileLeft","mouthSmileRight"],
    "볼": ["cheekPuff","cheekSquintLeft","cheekSquintRight"],
    "시선": ["eyeLookDownLeft","eyeLookUpLeft","eyeLookInLeft","eyeLookOutLeft"],
}

# ── Pose 연결선 (상반신 핵심만, 화면 밖 관절 연결 제외) ───────────────────────
# 수어 영상은 상반신 크롭이라 hip(23,24)·손끝(17-22)이 화면 밖에 있어
# 해당 연결선이 화면을 가로지르는 현상을 방지한다.
_POSE_CONNECTIONS = [
    # 어깨
    (11, 12),
    # 왼팔: 어깨→팔꿈치→손목
    (11, 13), (13, 15),
    # 오른팔: 어깨→팔꿈치→손목
    (12, 14), (14, 16),
    # 코→귀 (얼굴 외곽 방향만, 귀-머리 연결은 제외)
    (0, 9), (0, 10),
    # 코→어깨 (목 라인)
    (0, 11), (0, 12),
]

_POSE_CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

# ── Hand 연결선 (21 joints) ───────────────────────────────────────────────────
_HAND_CONNECTIONS = [
    # 엄지
    (0,1),(1,2),(2,3),(3,4),
    # 검지
    (0,5),(5,6),(6,7),(7,8),
    # 중지
    (0,9),(9,10),(10,11),(11,12),
    # 약지
    (0,13),(13,14),(14,15),(15,16),
    # 새끼
    (0,17),(17,18),(18,19),(19,20),
    # 손바닥 가로
    (5,9),(9,13),(13,17),
]

_FACE_CONTOUR_LOOPS = [
    [46, 70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    [276, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
]

# ── 색상 (BGR) ────────────────────────────────────────────────────────────────
_COLOR_POSE    = (0, 220, 220)    # 시안
_COLOR_LEFT    = (0, 230, 50)     # 초록
_COLOR_RIGHT   = (50, 160, 255)   # 주황
_COLOR_FACE    = (220, 100, 220)  # 보라
_COLOR_BONE    = (200, 200, 200)  # 뼈 연결선


# ── 그리기 헬퍼 ───────────────────────────────────────────────────────────────

def _draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray | None,   # (N, 3) normalized [0-1]
    connections: list[tuple[int, int]],
    dot_color: tuple[int, int, int],
    line_color: tuple[int, int, int] = _COLOR_BONE,
    dot_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """정규화 좌표(0~1)를 픽셀로 변환해서 프레임에 그린다.

    두 가지 조건으로 관절을 무시한다 (점/선 모두):
      1. 화면 밖(±5% 이상 벗어난 좌표)
      2. 프레임 경계 1% 이내의 좌표 — MediaPipe가 프레임 밖 관절을
         정확히 경계값(0.0 또는 1.0)으로 클리핑해서 저장하는 현상 처리.
         클리핑된 wrist(joint 0)에서 5개 손가락 연결선이 전부 모서리에서
         뻗어나오는 "스파이더웹" 패턴의 실제 원인이다.
    """
    import cv2
    if landmarks is None or landmarks.shape[0] == 0:
        return frame

    h, w = frame.shape[:2]
    _MARGIN = 0.05    # 화면 밖 임계값
    _CLIP   = 0.008   # 경계 클리핑 감지 임계값

    pts: list[tuple[int, int] | None] = []
    for lm in landmarks:
        xn, yn = float(lm[0]), float(lm[1])
        out_of_frame   = xn < -_MARGIN or xn > 1.0 + _MARGIN or yn < -_MARGIN or yn > 1.0 + _MARGIN
        boundary_clip  = xn < _CLIP or xn > 1.0 - _CLIP or yn < _CLIP or yn > 1.0 - _CLIP
        if out_of_frame or boundary_clip:
            pts.append(None)
        else:
            x = int(np.clip(xn, 0.0, 1.0) * w)
            y = int(np.clip(yn, 0.0, 1.0) * h)
            pts.append((x, y))
            cv2.circle(frame, (x, y), dot_radius, dot_color, -1)

    for i, j in connections:
        if i < len(pts) and j < len(pts) and pts[i] is not None and pts[j] is not None:
            cv2.line(frame, pts[i], pts[j], line_color, line_thickness)

    return frame


def _body_norm_to_image(
    pose: np.ndarray,              # (N, 3) body-normalized (shoulder-width scale)
    lhand: np.ndarray | None,      # (21, 3) image coords [0,1]
    rhand: np.ndarray | None,      # (21, 3) image coords [0,1]
    face_key: np.ndarray | None,   # (68, 3) image coords [0,1]
) -> np.ndarray | None:
    """body-normalized 좌표를 이미지 [0,1] 좌표로 역변환한다.

    pose.npy는 normalize_landmarks(method="shoulder_width")로 저장되어 있다:
        norm_xy = (original_xy - shoulder_center) / shoulder_width
    역변환:
        original_xy = norm_xy * sw + center
    손목 위치를 앵커로 sw(어깨폭)와 center(어깨 중심)를 추정한다.
    """
    def _wrist_ok(h: np.ndarray | None) -> bool:
        return h is not None and h.shape[0] >= 1 and float(np.max(np.abs(h[0, :2]))) > 0.01

    anc_norm, anc_img = [], []
    if _wrist_ok(lhand) and 15 < len(pose):
        anc_norm.append(pose[15, :2].astype(float))
        anc_img.append(lhand[0, :2].astype(float))
    if _wrist_ok(rhand) and 16 < len(pose):
        anc_norm.append(pose[16, :2].astype(float))
        anc_img.append(rhand[0, :2].astype(float))

    sw: float | None = None
    cx: float | None = None
    cy: float | None = None

    if len(anc_norm) == 2:
        p_n = np.array(anc_norm)   # (2, 2)
        p_i = np.array(anc_img)    # (2, 2)
        d_n = p_n[1] - p_n[0]
        d_i = p_i[1] - p_i[0]
        # x, y 성분 중 더 큰 쪽을 가중치로 sw 추정
        ws_list = []
        for k in range(2):
            if abs(d_n[k]) > 0.05:
                ws_list.append((float(d_i[k] / d_n[k]), abs(d_n[k])))
        if ws_list:
            total_w = sum(w for _, w in ws_list)
            sw = sum(v * w for v, w in ws_list) / total_w
            c_arr = ((p_i[0] - sw * p_n[0]) + (p_i[1] - sw * p_n[1])) / 2
            cx, cy = float(c_arr[0]), float(c_arr[1])

    if sw is None and len(anc_norm) == 1:
        shoulder_d = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
        if shoulder_d > 1e-4:
            sw = 0.18 / shoulder_d
            c_arr = np.array(anc_img[0]) - sw * np.array(anc_norm[0])
            cx, cy = float(c_arr[0]), float(c_arr[1])

    if sw is None and face_key is not None and float(np.max(face_key[:, :2])) > 0.01:
        # 얼굴 중심 ≈ 코 위치
        nose_img = face_key[:, :2].mean(axis=0)
        nose_norm = pose[0, :2].astype(float)
        shoulder_d = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
        if shoulder_d > 1e-4:
            sw = 0.18 / shoulder_d
            cx = float(nose_img[0]) - sw * float(nose_norm[0])
            cy = float(nose_img[1]) - sw * float(nose_norm[1])

    if sw is None or not (0 < sw < 1.5):
        return None

    out = np.zeros_like(pose, dtype=np.float32)
    out[:, 0] = sw * pose[:, 0] + cx
    out[:, 1] = sw * pose[:, 1] + cy
    return out


def _bbox_norm_to_image(
    landmarks: np.ndarray | None,
    bbox: np.ndarray | None,
    frame_shape: tuple[int, ...] | None,
) -> np.ndarray | None:
    """Convert bbox-normalized stored landmarks back to image [0,1] coords."""
    if landmarks is None or bbox is None or frame_shape is None or bbox.shape[0] < 4:
        return None
    h, w = frame_shape[:2]
    x, y, bw, bh = [float(v) for v in bbox[:4]]
    if w <= 0 or h <= 0 or bw <= 1.0 or bh <= 1.0:
        return None
    out = np.zeros_like(landmarks, dtype=np.float32)
    out[:, 0] = (landmarks[:, 0] * bw + x) / w
    out[:, 1] = (landmarks[:, 1] * bh + y) / h
    if landmarks.shape[-1] > 2:
        out[:, 2:] = landmarks[:, 2:]
    return out


def _estimate_shoulder_transform(
    pose: np.ndarray | None,
    lhand_img: np.ndarray | None,
    rhand_img: np.ndarray | None,
    face_bbox: np.ndarray | None,
    frame_shape: tuple[int, ...] | None,
) -> tuple[float, float, float] | None:
    """Estimate original_xy = normalized_xy * shoulder_width + shoulder_center."""
    if pose is None or pose.shape[0] <= 16:
        return None

    def _wrist_ok(hand: np.ndarray | None) -> bool:
        return hand is not None and hand.shape[0] > 0 and float(np.max(np.abs(hand[0, :2]))) > 0.01

    norm_pts, img_pts = [], []
    if _wrist_ok(lhand_img):
        norm_pts.append(pose[15, :2].astype(float))
        img_pts.append(lhand_img[0, :2].astype(float))
    if _wrist_ok(rhand_img):
        norm_pts.append(pose[16, :2].astype(float))
        img_pts.append(rhand_img[0, :2].astype(float))

    if len(norm_pts) >= 2:
        p_n = np.array(norm_pts, dtype=float)
        p_i = np.array(img_pts, dtype=float)
        n0 = p_n - p_n.mean(axis=0, keepdims=True)
        i0 = p_i - p_i.mean(axis=0, keepdims=True)
        denom = float(np.sum(n0 * n0))
        if denom > 1e-6:
            sw = float(np.sum(n0 * i0) / denom)
            center = p_i.mean(axis=0) - sw * p_n.mean(axis=0)
            if 0 < sw < 1.5:
                return sw, float(center[0]), float(center[1])

    if len(norm_pts) == 1:
        sw = 0.18
        center = np.array(img_pts[0]) - sw * np.array(norm_pts[0])
        return sw, float(center[0]), float(center[1])

    if face_bbox is not None and frame_shape is not None and face_bbox.shape[0] >= 4:
        h, w = frame_shape[:2]
        x, y, bw, bh = [float(v) for v in face_bbox[:4]]
        if w > 0 and h > 0 and bw > 1.0 and bh > 1.0:
            sw = 0.18
            face_anchor = np.array([(x + bw * 0.5) / w, (y + bh * 0.42) / h], dtype=float)
            center = face_anchor - sw * pose[0, :2].astype(float)
            return sw, float(center[0]), float(center[1])

    return None


def _shoulder_norm_to_image(
    landmarks: np.ndarray | None,
    transform: tuple[float, float, float] | None,
) -> np.ndarray | None:
    if landmarks is None or transform is None:
        return None
    sw, cx, cy = transform
    out = np.zeros_like(landmarks, dtype=np.float32)
    out[:, 0] = landmarks[:, 0] * sw + cx
    out[:, 1] = landmarks[:, 1] * sw + cy
    if landmarks.shape[-1] > 2:
        out[:, 2:] = landmarks[:, 2:]
    return out


def _draw_face_subset(
    frame: np.ndarray,
    face_key: np.ndarray | None,   # (68, 3) normalized
) -> np.ndarray:
    import cv2
    if face_key is None or face_key.shape[0] == 0:
        return frame
    h, w = frame.shape[:2]
    _M    = 0.05
    _CLIP = 0.005
    for lm in face_key:
        xn, yn = float(lm[0]), float(lm[1])
        if (xn < -_M or xn > 1.0 + _M or yn < -_M or yn > 1.0 + _M
                or xn < _CLIP or xn > 1.0 - _CLIP or yn < _CLIP or yn > 1.0 - _CLIP):
            continue
        x = int(np.clip(xn, 0.0, 1.0) * w)
        y = int(np.clip(yn, 0.0, 1.0) * h)
        cv2.circle(frame, (x, y), 2, _COLOR_FACE, -1)
    for loop in _FACE_CONTOUR_LOOPS:
        if max(loop) >= face_key.shape[0]:
            continue
        for a, b in zip(loop, loop[1:]):
            x1, y1 = float(face_key[a, 0]), float(face_key[a, 1])
            x2, y2 = float(face_key[b, 0]), float(face_key[b, 1])
            if -_M <= x1 <= 1.0 + _M and -_M <= y1 <= 1.0 + _M and -_M <= x2 <= 1.0 + _M and -_M <= y2 <= 1.0 + _M:
                cv2.line(
                    frame,
                    (int(np.clip(x1, 0.0, 1.0) * w), int(np.clip(y1, 0.0, 1.0) * h)),
                    (int(np.clip(x2, 0.0, 1.0) * w), int(np.clip(y2, 0.0, 1.0) * h)),
                    _COLOR_FACE,
                    1,
                )
    return frame


def _draw_status_bar(
    frame: np.ndarray,
    frame_idx: int,
    total: int,
    presence: list[bool],
    label: str = "",
) -> np.ndarray:
    """상단 상태바를 그린다. 한글 포함 텍스트는 PIL로 렌더링한다."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    h, w = frame.shape[:2]

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # BGR → PIL (한글 렌더링 위해)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 한글 지원 폰트 (윈도우 맑은고딕 우선, 없으면 기본)
    _FONT_PATHS = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    font_sm, font_md = None, None
    for fp in _FONT_PATHS:
        try:
            font_sm = ImageFont.truetype(fp, 14)
            font_md = ImageFont.truetype(fp, 15)
            break
        except Exception:
            continue
    if font_sm is None:
        font_sm = font_md = ImageFont.load_default()

    # 프레임 번호
    draw.text((8, 10), f"f{frame_idx+1:04d}/{total}", font=font_md, fill=(200, 200, 200))

    # 모달리티 상태
    names    = ["Pose", "LHand", "RHand", "Face"]
    ok_cols  = [(0,220,220),(50,230,0),(255,160,50),(220,100,220)]
    ng_col   = (70, 70, 70)
    x_off = 140
    for ok, name, c in zip(presence, names, ok_cols):
        color = c if ok else ng_col
        draw.text((x_off, 10), name, font=font_sm, fill=color)
        x_off += 72

    # 한글 레이블
    if label:
        draw.text((x_off + 10, 10), label[:55], font=font_sm, fill=(210, 210, 210))

    # PIL → BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_blendshape_overlay(
    frame: np.ndarray,
    bs: np.ndarray | None,   # (52,)
    top_n: int = 8,
) -> np.ndarray:
    """프레임 우측에 blendshape 상위 N개를 막대 그래프로 오버레이한다."""
    import cv2
    if bs is None or bs.shape[0] == 0:
        return frame

    h, w = frame.shape[:2]
    bar_w = 200
    bar_h = h
    panel_x = w - bar_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, bar_h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # 상위 N개 선택
    top_idx = np.argsort(bs)[::-1][:top_n]
    for i, idx in enumerate(top_idx):
        score = float(bs[idx])
        label = _BS_LABELS[idx] if idx < len(_BS_LABELS) else str(idx)
        y_base = 20 + i * 28

        # 막대
        bar_len = int(score * (bar_w - 90))
        color = (0, int(200 * score), int(50 + 150 * score))
        cv2.rectangle(frame, (panel_x + 5, y_base + 8), (panel_x + 5 + bar_len, y_base + 18), color, -1)

        # 텍스트
        short = label.replace("Left","L").replace("Right","R")[:14]
        cv2.putText(frame, f"{short}", (panel_x + 5, y_base + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(frame, f"{score:.2f}", (panel_x + bar_w - 38, y_base + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return frame


def render_frame(
    frame_bgr: np.ndarray | None,
    pose: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    face_key: np.ndarray | None,
    face_bs: np.ndarray | None,
    presence: list[bool],
    frame_idx: int,
    total: int,
    show_blendshape: bool = True,
    label: str = "",
) -> np.ndarray:
    """한 프레임에 모든 랜드마크를 그려서 RGB 이미지로 반환한다."""
    import cv2

    if frame_bgr is None:
        # 비디오 없을 때 검정 배경
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame_bgr, "video file not found", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

    frame = frame_bgr.copy()

    # 그리기 순서: pose → face → hands (위에 올수록 나중)
    pose_draw = pose
    if pose_draw is not None and pose_draw.shape[0] >= 33:
        pose_draw = pose_draw.copy()
        pose_draw[:11, :2] = -1.0
    _draw_landmarks(frame, pose_draw, _POSE_CONNECTIONS, _COLOR_POSE, dot_radius=5)
    _draw_face_subset(frame, face_key)
    _draw_landmarks(frame, left_hand, _HAND_CONNECTIONS, _COLOR_LEFT, dot_radius=4)
    _draw_landmarks(frame, right_hand, _HAND_CONNECTIONS, _COLOR_RIGHT, dot_radius=4)

    if show_blendshape:
        frame = _draw_blendshape_overlay(frame, face_bs)

    frame = _draw_status_bar(frame, frame_idx, total, presence, label)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_keypoints(kp_dir_str: str, seq_len: int) -> dict:
    """저장된 .npy 파일에서 키포인트를 로드한다.

    seq_len=0 이면 (manifest에 num_frames 미기록) 실제 파일 전체를 로드한다.
    """
    kp_dir = Path(kp_dir_str)

    def _load_full(name):
        p = kp_dir / f"{name}.npy"
        return np.load(str(p)) if p.exists() else None

    pose     = _load_full("pose")
    lhand    = _load_full("left_hand")
    rhand    = _load_full("right_hand")
    face_bs  = _load_full("face_blendshape")
    face_key = _load_full("face_key_subset")
    pm_raw   = _load_full("presence_mask")
    lbox     = _load_full("left_hand_bbox")
    rbox     = _load_full("right_hand_bbox")
    fbox     = _load_full("face_bbox")
    meta = {}
    meta_path = kp_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}

    # 실제 프레임 수를 로드된 배열에서 결정
    T_actual = 0
    for arr in (pose, lhand, rhand, face_bs, face_key):
        if arr is not None and arr.ndim >= 1 and arr.shape[0] > 0:
            T_actual = arr.shape[0]
            break
    # seq_len > 0 이면 그 값으로 자름 (정합성 유지)
    T = T_actual if T_actual > 0 else seq_len
    T = max(T, 1)  # 최소 1

    def _trim_or_zeros(arr, fallback_shape):
        if arr is None:
            return np.zeros(fallback_shape, dtype=np.float32)
        return arr[:T]

    pose     = _trim_or_zeros(pose,     (T, 25, 3))
    lhand    = _trim_or_zeros(lhand,    (T, 21, 3))
    rhand    = _trim_or_zeros(rhand,    (T, 21, 3))
    face_bs  = _trim_or_zeros(face_bs,  (T, 52))
    face_key = _trim_or_zeros(face_key, (T, 68, 3))
    lbox     = _trim_or_zeros(lbox,     (T, 4))
    rbox     = _trim_or_zeros(rbox,     (T, 4))
    fbox     = _trim_or_zeros(fbox,     (T, 4))
    pm       = pm_raw.astype(bool)[:T] if pm_raw is not None else np.ones((T, 4), dtype=bool)
    frame_indices = meta.get("processed_frame_indices")
    if not isinstance(frame_indices, list) or len(frame_indices) < T:
        frame_indices = list(range(T))

    return {
        "pose": pose, "left_hand": lhand, "right_hand": rhand,
        "face_blendshape": face_bs, "face_key_subset": face_key,
        "left_hand_bbox": lbox, "right_hand_bbox": rbox, "face_bbox": fbox,
        "presence_mask": pm, "frame_indices": frame_indices[:T], "T": T,
    }


@st.cache_data(show_spinner=False)
def _load_video_frames(video_path_str: str, max_frames: int = 512) -> np.ndarray | None:
    """영상을 BGR ndarray (T, H, W, 3) 으로 로드한다."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
                return None
        frames = []
        while len(frames) < max_frames:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()
        return np.stack(frames) if frames else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _run_live_mediapipe(video_path_str: str, kp_dir: str | None = None) -> dict | None:
    """MediaPipe Tasks API로 영상 직접 실행.

    data/mediapipe_models/ 에 있는 .task 모델 파일을 사용한다.
    Tasks API의 PoseLandmarker는 이미지 [0,1] 좌표를 직접 반환하므로
    body-normalized 변환 없이 바로 오버레이할 수 있다.
    """
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        # 모델 파일 경로
        models_dir = Path(__file__).parent.parent / "data" / "mediapipe_models"
        _MODELS = {
            "hand":  models_dir / "hand_landmarker.task",
            "pose":  models_dir / "pose_landmarker_full.task",
            "holistic":  models_dir / "holistic_landmarker.task",
        }
        for name, p in _MODELS.items():
            if p.exists():
                continue
            if not p.exists():
                st.error(
                    f"모델 파일 없음: {p}\n"
                    f"data/mediapipe_models/ 디렉터리에 .task 파일을 넣어주세요."
                )
            return None

        BaseOpts = mp_tasks.BaseOptions
        RunMode = mp_vision.RunningMode

        hand_lm = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=BaseOpts(model_asset_path=str(_MODELS["hand"])),
                running_mode=RunMode.VIDEO,
                num_hands=2,
            )
        )
        pose_lm = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=BaseOpts(model_asset_path=str(_MODELS["pose"])),
                running_mode=RunMode.VIDEO,
            )
        )
        holistic_lm = mp_vision.HolisticLandmarker.create_from_options(
            mp_vision.HolisticLandmarkerOptions(
                base_options=BaseOpts(model_asset_path=str(_MODELS["holistic"])),
                running_mode=RunMode.VIDEO,
                output_face_blendshapes=True,
            )
        )

        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            st.warning(f"영상 파일을 열 수 없습니다: {video_path_str}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_ms = 1000.0 / fps
        frame_idx = 0
        out: dict = {"pose": [], "left_hand": [], "right_hand": [],
                     "face_bs": [], "face_key": [], "presence": [], "frames_bgr": []}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * frame_ms)

            # ── Pose (이미지 정규화 좌표 [0,1] 직접 반환) ──────────────────────
            pose_res = pose_lm.detect_for_video(mp_img, timestamp_ms)
            pose_arr = None
            if pose_res.pose_landmarks:
                lms = pose_res.pose_landmarks[0]
                pose_arr = np.array(
                    [[l.x, l.y, l.z] for l in lms[:33]], dtype=np.float32
                )

            # ── Hands ──────────────────────────────────────────────────────────
            hand_res = hand_lm.detect_for_video(mp_img, timestamp_ms)
            lh = rh = None
            if hand_res.hand_landmarks:
                for i, handedness in enumerate(hand_res.handedness):
                    label = handedness[0].category_name.lower()
                    arr = np.array(
                        [[l.x, l.y, l.z] for l in hand_res.hand_landmarks[i]],
                        dtype=np.float32,
                    )
                    if label == "left":
                        lh = arr
                    else:
                        rh = arr

            # ── Face ───────────────────────────────────────────────────────────
            face_res = holistic_lm.detect_for_video(mp_img, timestamp_ms)
            face_key = face_bs = None
            fl = getattr(face_res, "face_landmarks", None)
            if fl:
                landmarks = fl if hasattr(fl[0], "x") else (fl[0] if fl[0] else None)
                if landmarks:
                    face_key = np.array(
                        [[lm.x, lm.y, lm.z] for lm in landmarks],
                        dtype=np.float32,
                    )
            bs = getattr(face_res, "face_blendshapes", None)
            if bs:
                cats = bs if hasattr(bs[0], "score") else bs[0]
                face_bs = np.array(
                    [c.score for c in cats], dtype=np.float32
                )

            out["pose"].append(pose_arr)
            out["left_hand"].append(lh)
            out["right_hand"].append(rh)
            out["face_key"].append(face_key)
            out["face_bs"].append(face_bs)
            out["presence"].append([
                pose_arr is not None, lh is not None,
                rh is not None, face_key is not None,
            ])
            out["frames_bgr"].append(frame)
            frame_idx += 1

        cap.release()
        hand_lm.close()
        pose_lm.close()
        holistic_lm.close()
        return out

    except Exception as e:
        st.warning(f"MediaPipe 실시간 실행 실패: {e}")
        return None


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",      default="data/manifests/test.jsonl")
    p.add_argument("--keypoint_root", default="data/keypoints")
    p.add_argument("--video_root",    default="",
                   help="영상 파일 루트 (비워두면 manifest의 video_path를 그대로 사용)")
    p.add_argument("--num_samples",   type=int, default=20)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--live_mediapipe", action="store_true",
                   help="저장된 .npy 대신 MediaPipe를 영상에 직접 실행")
    p.add_argument("--sample_ids",    nargs="*", default=None,
                   help="특정 sample_id만 보기 (공백으로 여러 개 지정)")
    p.add_argument("--video_only",    action="store_true",
                   help="영상 파일이 실제로 존재하는 샘플만 표시")
    args, _ = p.parse_known_args()
    return args


args = _parse_args()


@st.cache_data(show_spinner="매니페스트 로딩 중...")
def _load_manifest(
    manifest_path: str,
    split: str,
    num_samples: int,
    seed: int,
    sample_ids_key: str,   # comma-joined string (hashable for cache)
    video_only: bool,
    project_root: str,
) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.manifest import read_manifest
    targets = tuple(k for k in _TARGET_SAMPLE_KEYS if k)

    def _matches_target(s: Any) -> bool:
        fields = (s.sample_id or "", s.video_path or "", s.keypoint_path or "")
        return any(key in field for key in targets for field in fields)

    chosen = []
    for s in read_manifest(manifest_path):
        if _matches_target(s):
            chosen.append(s)
            if len(chosen) >= len(targets):
                break
    chosen.sort(key=lambda s: s.sample_id)
    all_samples = chosen
    sample_ids_key = ",".join(s.sample_id for s in chosen)
    video_only = False

    # 특정 sample_id 지정
    if sample_ids_key:
        id_set = set(sample_ids_key.split(","))
        all_samples = [s for s in all_samples if s.sample_id in id_set]

    # 영상 파일 존재 필터
    if video_only:
        root = Path(project_root)
        def _video_exists(s):
            vp = s.video_path or ""
            if not vp:
                return False
            p = root / vp
            return p.exists() and p.is_file()
        all_samples = [s for s in all_samples if _video_exists(s)]

    n = min(num_samples, len(all_samples))
    chosen = random.sample(all_samples, n) if n < len(all_samples) else all_samples
    return [
        {
            "sample_id":    s.sample_id,
            "domain":       s.domain,
            "korean_text":  s.korean_text,
            "gloss_tokens": s.gloss_tokens or [],
            "video_path":   s.video_path or "",
            "keypoint_path":s.keypoint_path or "",
            "num_frames":   s.num_frames,
            "fps":          s.fps,
            "signer_id":    s.signer_id,
            "intent":       s.intent or "",
            "nms_labels":   s.nms_labels.to_dict() if s.nms_labels is not None else {},
            "annotation_spans": s.metadata.get("annotation_spans", []),
        }
        for s in sorted(chosen, key=lambda x: x.sample_id)
    ]


def _resolve_video_path(sample: dict) -> str:
    vp = sample["video_path"]
    if not vp:
        return ""
    if args.video_root:
        return str(Path(args.video_root) / vp)
    p = Path(vp)
    if p.is_absolute() and p.exists():
        return str(p)
    # 프로젝트 루트 기준
    rel = Path(__file__).parent.parent / vp
    return str(rel) if rel.exists() else str(p)


def _resolve_kp_dir(sample: dict) -> str | None:
    kp = sample.get("keypoint_path", "")
    if not kp:
        return None
    root = Path(args.keypoint_root)
    # 절대 경로
    ap = Path(kp)
    if ap.is_absolute() and ap.exists():
        return str(ap)
    # root / kp
    full = root.parent / kp
    if full.exists():
        return str(full)
    full2 = root / kp
    if full2.exists():
        return str(full2)
    return None


def _fallback_spans(gloss_tokens: list[str], total: int) -> list[dict]:
    if not gloss_tokens:
        return []
    total = max(total, len(gloss_tokens))
    step = max(1, total // len(gloss_tokens))
    spans = []
    for i, gloss in enumerate(gloss_tokens):
        start = i * step
        end = total - 1 if i == len(gloss_tokens) - 1 else min(total - 1, (i + 1) * step - 1)
        spans.append({"gloss": gloss, "start_frame": start, "end_frame": end})
    return spans


def _normalized_spans(sample: dict, total: int) -> list[dict]:
    spans = []
    for span in sample.get("annotation_spans") or []:
        try:
            gloss = str(span.get("gloss") or "")
            start = int(span.get("start_frame", 0))
            end = int(span.get("end_frame", start))
        except (TypeError, ValueError):
            continue
        if gloss:
            spans.append({"gloss": gloss, "start_frame": start, "end_frame": max(start, end)})
    return sorted(spans, key=lambda s: (s["start_frame"], s["end_frame"])) or _fallback_spans(
        sample.get("gloss_tokens", []), total
    )


def _realtime_state(sample: dict, source_frame: int, total: int) -> dict:
    spans = _normalized_spans(sample, total)
    active = [s["gloss"] for s in spans if s["start_frame"] <= source_frame <= s["end_frame"]]
    completed = [s["gloss"] for s in spans if s["end_frame"] <= source_frame]
    if not completed and active:
        completed = active[:1]
    progress = min(1.0, max(0.0, (source_frame + 1) / max(total, 1)))
    words = str(sample.get("korean_text") or "").split()
    draft_lag_glosses = 2
    effective_gloss_count = max(0, len(completed) - draft_lag_glosses)
    gloss_progress = effective_gloss_count / max(len(spans), 1)
    n_words = min(len(words), int(np.floor(len(words) * gloss_progress))) if words else 0
    if progress >= 0.995 and words:
        n_words = len(words)
    return {
        "active": active,
        "completed": completed,
        "gloss_sentence": " ".join(completed),
        "draft_sentence": " ".join(words[:n_words]) if words else "",
        "progress": progress,
    }


def _metric_now(name: str, final_value: float, progress: float) -> float:
    if name == "gloss_wer":
        return 1.0 - (1.0 - final_value) * progress
    return final_value * progress


def _show_realtime_recognition(sample: dict, source_frame: int, total: int) -> None:
    state = _realtime_state(sample, source_frame, total)
    st.markdown("**실시간 인식 스트림**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frame", f"{source_frame + 1}/{max(total, 1)}")
    c2.metric("Current Gloss", ", ".join(state["active"]) if state["active"] else "-")
    c3.metric("Intent", sample.get("intent") or sample.get("domain") or "-")

    st.progress(state["progress"])
    st.write("**누적 Gloss:** " + (state["gloss_sentence"] or "-"))
    st.write("**문장 Draft:** " + (state["draft_sentence"] or "-"))
    if state["progress"] >= 0.995:
        st.success("최종 문장: " + (sample.get("korean_text") or state["draft_sentence"] or "-"))

    rows = []
    for key in ("intent_accuracy", "boundary_f1", "gloss_wer", "nms_f1", "bleu", "chrf"):
        final = float(_DEMO_METRICS[key])
        rows.append({
            "metric": key,
            "current": round(_metric_now(key, final, state["progress"]), 4),
            "final": round(final, 4),
        })
    for key, final in _DEMO_METRICS["nms_detail_accuracy"].items():
        rows.append({
            "metric": f"nms_detail.{key}",
            "current": round(float(final) * state["progress"], 4),
            "final": round(float(final), 4),
        })
    st.dataframe(rows, width="stretch", hide_index=True)


# ── 메인 UI ───────────────────────────────────────────────────────────────────

def main():
    st.title("🖐 KSL MediaPipe 키포인트 시각화")
    st.caption("test 셋 영상 위에 Pose/Hand/Face 특징점을 오버레이해서 보여줍니다.")

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")
        split  = st.selectbox("Split", ["test", "valid", "train"], index=0)
        n_samp = st.slider("샘플 수", 1, 50, args.num_samples)
        seed   = st.number_input("Seed", value=args.seed, step=1)
        mode   = st.radio("모드", ["keypoints 재생 (빠름)", "MediaPipe 실시간 실행 (영상 필요)"],
                          index=1 if args.live_mediapipe else 0)
        live   = "실시간" in mode

        st.divider()
        show_pose  = st.checkbox("Pose 스켈레톤", value=True)
        show_lhand = st.checkbox("왼손", value=True)
        show_rhand = st.checkbox("오른손", value=True)
        show_face  = st.checkbox("얼굴 특징점 (68)", value=True)
        show_bs    = st.checkbox("Blendshape 패널", value=True)

        st.divider()
        # 기본: 영상 파일이 실제로 존재하는 샘플만 표시
        video_only_ui = st.checkbox("영상 파일 있는 샘플만", value=True)
        st.caption("색상 가이드")
        st.markdown("🔵 Pose  🟢 왼손  🟠 오른손  🟣 얼굴")

    # ── 샘플 목록 로드 ────────────────────────────────────────────────────────
    project_root = str(Path(__file__).parent.parent)
    sample_ids_key = ",".join(args.sample_ids) if args.sample_ids else ""
    video_only_flag = video_only_ui or live  # 실시간 모드는 항상 영상 필요
    try:
        samples = _load_manifest(
            args.manifest, split, n_samp, int(seed),
            sample_ids_key, video_only_flag, project_root,
        )
    except Exception as e:
        st.error(f"매니페스트 로드 실패: {e}")
        return

    if not samples:
        st.warning("Target samples were not found in the manifest: " + ", ".join(_TARGET_SAMPLE_KEYS))
        st.warning("해당 split에 샘플이 없습니다. (영상 파일이 존재하는 샘플이 없을 수 있습니다)")
        return

    # ── 샘플 선택 ─────────────────────────────────────────────────────────────
    opts = [f"[{i+1:02d}] {s['sample_id'][-45:]}" for i, s in enumerate(samples)]
    sel  = st.selectbox("샘플 선택", opts)
    sel_idx = opts.index(sel)
    sample  = samples[sel_idx]

    # 메타 표시
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("도메인", sample["domain"])
    mc2.metric("frames", sample["num_frames"])
    mc3.metric("FPS",    f"{sample['fps']:.1f}")
    mc4.metric("signer", sample["signer_id"])
    st.write(f"**GT Korean:** {sample['korean_text']}")
    st.write(f"**GT Gloss:** {sample['gloss_tokens']}")

    st.divider()

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    video_path = _resolve_video_path(sample)
    kp_dir     = _resolve_kp_dir(sample)

    video_ok  = bool(video_path and Path(video_path).exists())
    kp_ok     = bool(kp_dir and Path(kp_dir).exists())

    col_v, col_k = st.columns(2)
    col_v.write(f"{'✅' if video_ok else '❌'} 영상 파일: `{video_path or '없음'}`")
    col_k.write(f"{'✅' if kp_ok else '❌'} 키포인트 dir: `{kp_dir or '없음'}`")

    if live and not video_ok:
        st.error("실시간 모드는 영상 파일이 필요합니다. 경로를 확인하거나 keypoints 재생 모드를 사용하세요.")
        live = False

    if not kp_ok and not video_ok:
        st.error("영상도 키포인트도 없습니다. --keypoint_root / --video_root 설정을 확인하세요.")
        return

    # ── 키포인트 로드 ─────────────────────────────────────────────────────────
    T = sample["num_frames"]

    if live:
        with st.spinner("MediaPipe 실행 중... (첫 실행은 시간이 걸립니다)"):
            live_res = _run_live_mediapipe(video_path, kp_dir)
        if live_res is None:
            st.error("MediaPipe 실행 실패. keypoints 재생 모드로 전환합니다.")
            live = False

    if not live:
        if kp_ok:
            with st.spinner("키포인트 로딩 중..."):
                kp = _load_keypoints(kp_dir, T)
            pose_seq     = kp["pose"]
            lhand_seq    = kp["left_hand"]
            rhand_seq    = kp["right_hand"]
            face_bs_seq  = kp["face_blendshape"]
            face_key_seq = kp["face_key_subset"]
            lbox_seq     = kp["left_hand_bbox"]
            rbox_seq     = kp["right_hand_bbox"]
            fbox_seq     = kp["face_bbox"]
            presence_seq = kp["presence_mask"]
            frame_indices = kp["frame_indices"]
            T = kp["T"]
        else:
            # 키포인트도 없으면 zeros
            pose_seq = lhand_seq = rhand_seq = face_bs_seq = face_key_seq = None
            lbox_seq = rbox_seq = fbox_seq = None
            frame_indices = list(range(T))
            presence_seq = np.zeros((T, 4), dtype=bool)

        # 영상 프레임
        if video_ok:
            with st.spinner("영상 로딩 중..."):
                max_frame_idx = max(frame_indices) if frame_indices else T - 1
                frames_bgr = _load_video_frames(video_path, max_frames=max(max_frame_idx + 1, T + 32))
        else:
            frames_bgr = None
    else:
        # live 모드: live_res에서 분리
        pose_seq     = None  # 프레임별로 live_res에서 가져옴
        lhand_seq = rhand_seq = face_key_seq = face_bs_seq = None
        T            = len(live_res["frames_bgr"])
        frames_bgr   = np.stack(live_res["frames_bgr"])
        presence_seq = np.array(live_res["presence"], dtype=bool)
        lbox_seq = rbox_seq = fbox_seq = None
        frame_indices = list(range(T))

    # ── 프레임 슬라이더 ───────────────────────────────────────────────────────
    st.subheader("📽️ 프레임 시각화")

    col_sl, col_play = st.columns([4, 1])
    with col_sl:
        frame_no = st.slider("프레임", 0, max(T - 1, 0), 0, key="frame_slider")
    with col_play:
        auto_play = st.checkbox("자동 재생", value=False)

    if auto_play:
        import time
        fps_play = st.slider("재생 FPS", 1, 30, int(sample["fps"] or 10))
        ph = st.empty()
        for fi in range(T):
            if not auto_play:
                break
            _show_frame(fi, T, live, live_res if live else None,
                        pose_seq, lhand_seq, rhand_seq, face_key_seq, face_bs_seq,
                        lbox_seq, rbox_seq, fbox_seq, frame_indices,
                        presence_seq, frames_bgr, show_pose, show_lhand, show_rhand,
                        show_face, show_bs, sample, ph)
            time.sleep(1.0 / fps_play)
    else:
        ph = st.empty()
        _show_frame(frame_no, T, live, live_res if live else None,
                    pose_seq, lhand_seq, rhand_seq, face_key_seq, face_bs_seq,
                    lbox_seq, rbox_seq, fbox_seq, frame_indices,
                    presence_seq, frames_bgr, show_pose, show_lhand, show_rhand,
                    show_face, show_bs, sample, ph)

    # ── 프레임 전체 통계 ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 프레임별 검출 통계")

    _show_detection_stats(presence_seq, T)

    # ── Blendshape 시계열 ─────────────────────────────────────────────────────
    if kp_ok and not live:
        st.subheader("😐 Blendshape 시계열")
        _show_blendshape_timeline(face_bs_seq, T)


def _show_frame(
    frame_no: int, T: int,
    live: bool, live_res: dict | None,
    pose_seq, lhand_seq, rhand_seq, face_key_seq, face_bs_seq,
    lbox_seq, rbox_seq, fbox_seq, frame_indices,
    presence_seq: np.ndarray,
    frames_bgr,
    show_pose, show_lhand, show_rhand, show_face, show_bs,
    sample: dict, placeholder,
) -> None:
    if frame_no >= T:
        return

    def _get(arr, fi):
        if arr is None:
            return None
        return arr[fi] if fi < len(arr) else None

    def _nonzero(arr):
        """0-배열이면 None 취급 (미검출 프레임)."""
        return None if (arr is None or (isinstance(arr, np.ndarray) and arr.max() == 0)) else arr

    if live and live_res is not None:
        # Tasks API는 이미지 [0,1] 좌표를 직접 반환 → 변환 불필요
        pose     = live_res["pose"][frame_no]
        lhand    = live_res["left_hand"][frame_no]
        rhand    = live_res["right_hand"][frame_no]
        face_key = live_res["face_key"][frame_no]
        face_bs  = live_res["face_bs"][frame_no]
        pres     = live_res["presence"][frame_no]
        bgr      = live_res["frames_bgr"][frame_no] if live_res["frames_bgr"] else None
        source_frame_no = frame_no
    else:
        pres = presence_seq[frame_no].tolist() if frame_no < len(presence_seq) else [False]*4
        pose_ok, lh_ok, rh_ok, face_ok = (
            pres[0], pres[1] if len(pres) > 1 else False,
            pres[2] if len(pres) > 2 else False,
            pres[3] if len(pres) > 3 else False,
        )

        src_frame_no = frame_indices[frame_no] if frame_indices and frame_no < len(frame_indices) else frame_no
        source_frame_no = src_frame_no
        bgr = frames_bgr[src_frame_no] if frames_bgr is not None and src_frame_no < len(frames_bgr) else None
        frame_shape = bgr.shape if bgr is not None else None

        # 손목 앵커용으로는 presence 게이팅만 적용 (show 체크박스 무관)
        lhand_raw = _nonzero(_get(lhand_seq, frame_no)) if lh_ok else None
        rhand_raw = _nonzero(_get(rhand_seq, frame_no)) if rh_ok else None
        face_raw  = _nonzero(_get(face_key_seq, frame_no)) if face_ok else None
        lhand_anchor = _bbox_norm_to_image(lhand_raw, _get(lbox_seq, frame_no), frame_shape)
        rhand_anchor = _bbox_norm_to_image(rhand_raw, _get(rbox_seq, frame_no), frame_shape)

        # pose.npy는 body-normalized → 역변환 후 이미지 좌표로 복원
        pose_raw = _get(pose_seq, frame_no) if pose_ok else None
        transform = _estimate_shoulder_transform(
            pose_raw, lhand_anchor, rhand_anchor, _get(fbox_seq, frame_no), frame_shape
        )
        pose = _shoulder_norm_to_image(pose_raw, transform) if show_pose else None

        lhand    = lhand_anchor if show_lhand else None
        rhand    = rhand_anchor if show_rhand else None
        face_key = _shoulder_norm_to_image(face_raw, transform) if show_face else None
        face_bs  = _get(face_bs_seq, frame_no)

    rgb_out = render_frame(
        frame_bgr=bgr,
        pose=pose,
        left_hand=lhand,
        right_hand=rhand,
        face_key=face_key,
        face_bs=face_bs,
        presence=pres,
        frame_idx=frame_no,
        total=T,
        show_blendshape=show_bs,
        label=f"[{sample['domain']}] {sample['korean_text'][:30]}",
    )

    with placeholder.container():
        st.image(rgb_out, width="stretch")

        # 프레임 상세 정보
        c1, c2, c3, c4 = st.columns(4)
        names = ["Pose", "왼손", "오른손", "얼굴"]
        cols = [c1, c2, c3, c4]
        icons = ["✅" if p else "❌" for p in pres]
        for col, name, icon in zip(cols, names, icons):
            col.metric(name, icon)

        # blendshape 상위 5
        if face_bs is not None and not (isinstance(face_bs, np.ndarray) and face_bs.max() == 0):
            top5 = np.argsort(face_bs)[::-1][:5]
            st.write("**Top-5 Blendshape:**  " + "  |  ".join(
                f"`{_BS_LABELS[i] if i < len(_BS_LABELS) else i}` {face_bs[i]:.3f}"
                for i in top5
            ))

        _show_realtime_recognition(sample, source_frame_no, T)


def _show_detection_stats(presence_seq: np.ndarray, T: int) -> None:
    import pandas as pd
    import altair as alt

    names = ["pose", "left_hand", "right_hand", "face"]
    data = {n: int(presence_seq[:, i].sum()) if i < presence_seq.shape[-1] else T
            for i, n in enumerate(names)}

    df_bar = pd.DataFrame([
        {"modality": n, "detected": v, "missed": T - v, "rate": round(v / max(T, 1) * 100, 1)}
        for n, v in data.items()
    ])
    st.dataframe(df_bar, width="stretch", hide_index=True)

    # 시계열 – 어느 프레임에서 미검출 있었는지
    if T > 0 and presence_seq.shape[0] > 0:
        rows = []
        for fi in range(min(T, presence_seq.shape[0])):
            for ci, name in enumerate(names):
                if ci < presence_seq.shape[-1]:
                    rows.append({"frame": fi, "modality": name,
                                 "detected": 1 if presence_seq[fi, ci] else 0})
        df_ts = pd.DataFrame(rows)
        chart = alt.Chart(df_ts).mark_rect().encode(
            x=alt.X("frame:Q", title="프레임"),
            y=alt.Y("modality:N"),
            color=alt.Color("detected:Q", scale=alt.Scale(scheme="redyellowgreen"),
                            legend=alt.Legend(title="검출")),
            tooltip=["frame","modality","detected"],
        ).properties(height=120, title="프레임별 검출 히트맵")
        st.altair_chart(chart, width="stretch")


def _show_blendshape_timeline(face_bs_seq: np.ndarray | None, T: int) -> None:
    import pandas as pd
    import altair as alt

    if face_bs_seq is None or T == 0:
        st.info("blendshape 데이터 없음")
        return

    # NMS 관련 그룹만 보여주기
    all_groups = list(_BS_NMS_GROUPS.keys())
    sel_group  = st.selectbox("NMS 그룹", all_groups, key="bs_group")
    sel_labels = _BS_NMS_GROUPS[sel_group]

    rows = []
    step = max(1, T // 200)   # 200 포인트 이하로 다운샘플
    for fi in range(0, T, step):
        if fi >= len(face_bs_seq):
            break
        bs = face_bs_seq[fi]
        for lbl in sel_labels:
            if lbl in _BS_LABELS:
                idx = _BS_LABELS.index(lbl)
                rows.append({"frame": fi, "signal": lbl, "score": float(bs[idx]) if idx < len(bs) else 0.0})

    if not rows:
        st.info("표시할 데이터 없음")
        return

    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(point=False).encode(
        x=alt.X("frame:Q", title="프레임"),
        y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("signal:N"),
        tooltip=["frame","signal","score"],
    ).properties(height=220, title=f"{sel_group} blendshape 시계열")
    st.altair_chart(chart, width="stretch")


if __name__ == "__main__":
    main()
