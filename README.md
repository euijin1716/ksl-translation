# KSL Translation Pipeline

일상생활형 **한국수어(KSL) 영상 → 한국어 텍스트** 통역 시스템.

손동작 랜드마크 + 비수지신호(NMS: 얼굴 표정, 시선, 눈썹, 입 모양, 머리 움직임)를 동시에 처리하며,
인식 중간 출력(gloss / NMS / intent / boundary)을 유지한 채 LLM이 최종 문장을 보정한다.

---

## 프로젝트 구조

```
src/
├── data/          # 데이터 스키마, adapter, split, dataset
├── preprocess/    # MediaPipe 기반 특징 추출 파이프라인
├── models/        # 다중 스트림 인코더 + Decoder (Stage A/C)
├── train/         # 학습 루프 (KSLTrainer)
├── infer/         # 추론 파이프라인 + Streaming state machine
├── llm/           # LLM 문맥 보정기 (Claude / dummy)
└── eval/          # 평가 지표 (BLEU, chrF, NMS F1, WER 등)

scripts/
├── build_manifest.py       # 데이터셋 → signer-independent split → JSONL
├── run_preprocess.py       # MediaPipe 추출 → .npy 저장
├── run_train.py            # 학습 실행
├── run_eval.py             # 체크포인트 평가
├── run_smoke_test.py       # end-to-end 동작 확인 (더미 데이터)
└── setup_dummy_data.py     # smoke test용 더미 영상 생성

configs/
├── base.yaml                    # 공통 기본 설정
├── stage_c.yaml                 # Stage C (기본)
├── stage_c_earlystop.yaml       # early stopping 실험
├── stage_c_full.yaml            # 전체 epoch 실험
└── stage_c_verify.yaml          # 검증용 실험
```

---

## 실행 환경

**학습 / 전처리 / 평가** (Windows, GPU 필수)
- Python 3.13 + `.venv`
- PyTorch 2.11 + CUDA 12.6
- MediaPipe 0.10+, Transformers (klue/roberta-base)

```bash
# CUDA 버전에 맞는 PyTorch 별도 설치 필요
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

**개발 환경** (VS Code Dev Container, CPU only)
- Python 3.11-slim Docker 이미지
- `requirements.txt` 기준 설치 (CPU torch)

---

## 실행 순서

### 1. Smoke test (데이터 없이 파이프라인 동작 확인)

```bash
python scripts/setup_dummy_data.py   # 더미 MP4 생성
python scripts/run_smoke_test.py     # end-to-end 검증
```

### 2. 실제 데이터로 학습

```bash
# manifest 생성 (AI Hub 수어 영상 데이터 필요)
python scripts/build_manifest.py \
    --datasets aihub_sign \
    --root_aihub_sign data/aihub_sign

# MediaPipe 특징 추출 (MP4 → .npy)
python scripts/run_preprocess.py \
    --manifest data/manifests/all.jsonl \
    --config configs/base.yaml \
    --num_workers 2

# 학습
python scripts/run_train.py \
    --config configs/stage_c.yaml \
    --manifest data/manifests/train.jsonl

# 평가
python scripts/run_eval.py \
    --checkpoint checkpoints/C/best.pt \
    --manifest data/manifests/test.jsonl
```

---

## 파이프라인 구조

```
[원본 MP4]
    ↓ MediaPipe (Hand + Face + Pose Landmarker)
[.npy 랜드마크 + presence mask]
    ↓
KSLModel
  ├── E1: LandmarkEncoder    (pose + hand + face_key)
  ├── E2: HandVisualEncoder  (hand crop CNN — 현재 비활성)
  ├── E3: FaceExprEncoder    (face blendshape 52차원)
  ├── FusionModule           (late fusion)
  ├── AllHeads               (gloss CTC / NMS / intent / boundary)
  └── KoreanDraftDecoder     (한국어 초안)
    ↓ ContextCorrector (LLM)
[최종 한국어 문장]
```

**중간 출력 (반드시 유지)**:

| 출력 | 설명 |
|---|---|
| `gloss` | CTC decode 결과 |
| `nms` | 비수지신호 (눈썹/눈/입/머리) multi-label |
| `intent` | 도메인 분류 (병원/길안내/주문/예약/민원/도움요청) |
| `boundary` | 수어 구간 O/B-START/B-END |
| `draft_text` | 모델 1차 한국어 초안 |
| `final_text` | LLM 보정 최종 문장 |

---

## 학습 결과 (Stage C, test n=911)

| 지표 | 값 | 비고 |
|---|---|---|
| intent_accuracy | 1.00 | 단일 도메인(help)으로 구성된 test split — 판별력 없음 |
| nms_f1 | 0.76 | |
| boundary_f1 | 0.57 | |
| BLEU | 12.6 | |
| chrF | 33.4 | |

---

## 데이터

- **AI Hub 한국수어 영상** (데이터셋 103) — 주 학습 데이터
- **AI Hub 재난 안전 수어 데이터** — 추가 도메인
- split 방식: **signer-independent** (동일 화자 train/test 혼재 금지)

저장소에는 `data/manifests/test.jsonl`, `valid.jsonl`, `gloss_vocab.json`만 포함됩니다.
`train.jsonl`은 용량(16MB) 상 제외되어 있으므로, clone 후 아래 명령으로 재생성해야 합니다.

```bash
python scripts/build_manifest.py \
    --datasets aihub_sign \
    --root_aihub_sign data/aihub_sign
```
