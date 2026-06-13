"""KSL 평가 대시보드 (Streamlit).

실행:
  streamlit run scripts/eval_dashboard.py -- \\
    --checkpoint checkpoints/C/best.pt \\
    --config configs/stage_c.yaml \\
    --manifest data/manifests/test.jsonl \\
    --gloss_vocab data/manifests/gloss_vocab.json

또는 먼저 trace JSON을 생성하고 그걸 불러오기:
  streamlit run scripts/eval_dashboard.py -- \\
    --trace_json trace_logs/trace_test_20samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import streamlit as st

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KSL 추론 대시보드",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DOMAIN_KO = {
    "hospital": "병원",
    "directions": "길안내",
    "order": "주문/결제",
    "reservation": "예약/확인",
    "public": "공공민원",
    "help": "도움요청",
    "unknown": "알수없음",
}
_PRESENCE_NAMES = ["pose", "left_hand", "right_hand", "face"]
_NMS_KEYS = [
    "eyebrow_raise", "eyebrow_furrow", "eye_wide", "eye_squint",
    "nose_wrinkle", "mouth_open", "mouth_shape", "cheek_puff",
    "head_nod", "head_shake", "head_tilt", "gaze_direction",
]
_DOMAIN_LIST = ["hospital", "directions", "order", "reservation", "public", "help", "unknown"]


# ── 인자 파싱 (streamlit은 -- 이후 인자를 받음) ────────────────────────────────
@st.cache_resource
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--manifest", default="data/manifests/test.jsonl")
    p.add_argument("--gloss_vocab", default="data/manifests/gloss_vocab.json")
    p.add_argument("--trace_json", default=None, help="사전 생성된 trace JSON 파일")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--draft_mode", default="greedy", choices=["teacher", "greedy"])
    args, _ = p.parse_known_args()
    return args


args = _parse_args()


# ── 모델 + 데이터 로더 ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model_and_data():
    import random
    import torch
    import yaml

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_cfg: dict = {}
    if Path("configs/base.yaml").exists():
        base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8")) or {}
    stage_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    from _model_runtime import merge_cfg
    cfg = merge_cfg(base_cfg, stage_cfg)

    from transformers import AutoTokenizer
    from src.data.gloss_vocab import GlossVocab
    from src.data.keypoint_dataset import KeypointDataset
    from src.models.decoder import DecoderConfig
    from src.models.fusion import FusionConfig
    from src.models.heads import HeadsConfig
    from src.models.ksl_model import KSLModel, ModelConfig
    from src.models.streams.face_expr_encoder import FaceExprEncoderConfig
    from src.models.streams.hand_visual_encoder import HandVisualEncoderConfig
    from src.models.streams.landmark_encoder import LandmarkEncoderConfig

    def _dcls(cls, cfg_dict, **kw):
        d = dict(cfg_dict or {}); d.update(kw)
        ok = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in ok})

    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", {}).get("name", "klue/roberta-base"))
    gv = GlossVocab.load(args.gloss_vocab)

    mc = cfg.get("model", {})
    dec_cfg_d = mc.get("decoder", {})
    dec_cfg = DecoderConfig.from_tokenizer(
        tok, d_model=dec_cfg_d.get("d_model", 256), nhead=dec_cfg_d.get("nhead", 4),
        num_layers=dec_cfg_d.get("num_layers", 4), dim_feedforward=dec_cfg_d.get("dim_feedforward", 512),
        dropout=dec_cfg_d.get("dropout", 0.1), max_len=dec_cfg_d.get("max_len", 128),
    )
    model = KSLModel(ModelConfig(
        stage=mc.get("stage", "C"),
        landmark=_dcls(LandmarkEncoderConfig, mc.get("landmark", {})),
        hand_visual=_dcls(HandVisualEncoderConfig, mc.get("hand_visual", {})),
        face_expr=_dcls(FaceExprEncoderConfig, mc.get("face_expr", {})),
        fusion=_dcls(FusionConfig, mc.get("fusion", {})),
        heads=_dcls(HeadsConfig, mc.get("heads", {}), gloss_vocab_size=len(gv)),
        decoder=dec_cfg,
        enable_hand_visual=mc.get("enable_hand_visual", False),
    ))
    from _model_runtime import load_checkpoint
    load_checkpoint(model, Path(args.checkpoint), args.device)
    model.eval()

    dataset = KeypointDataset(
        manifest_path=args.manifest,
        keypoint_root=cfg.get("data", {}).get("keypoint_root", "data/keypoints"),
        crop_root=cfg.get("data", {}).get("crop_root", "data/crops"),
        split_group="test",
        gloss_vocab=gv,
        tokenizer=tok,
        max_seq_len=mc.get("landmark", {}).get("max_seq_len", 512),
        max_text_len=cfg.get("tokenizer", {}).get("max_length", 64),
        load_hand_crops=cfg.get("data", {}).get("load_hand_crops", True),
        sampling_strategy=cfg.get("data", {}).get("sequence_sampling", "uniform"),
        boundary_mode=cfg.get("data", {}).get("boundary_mode", "annotation_or_motion"),
    )
    return model, tok, gv, dataset, cfg


@st.cache_data(show_spinner="추론 실행 중...")
def run_inference_for_index(_model_key, idx: int):
    """단일 샘플 idx에 대해 forward pass 수행 후 직렬화 가능한 dict 반환."""
    import time
    import torch
    from src.data.dummy import collate_fn
    from torch.utils.data import DataLoader, Subset

    model, tok, gv, dataset, cfg = load_model_and_data()
    device = torch.device(args.device)

    loader = DataLoader(Subset(dataset, [idx]), batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))
    batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    seq_len = int(batch_d["seq_len"][0].item())
    T = batch_d["pose"].shape[1]
    mask = torch.arange(T, device=device).unsqueeze(0) >= batch_d["seq_len"].unsqueeze(1)

    t0 = time.perf_counter()
    with torch.no_grad():
        use_teacher = args.draft_mode == "teacher"
        outputs = model(
            pose=batch_d["pose"],
            left_hand=batch_d["left_hand"],
            right_hand=batch_d["right_hand"],
            face_blendshape=batch_d["face_blendshape"],
            face_key_subset=batch_d.get("face_key_subset"),
            presence_mask=batch_d.get("presence_mask"),
            tgt_tokens=batch_d.get("tgt_tokens") if use_teacher else None,
            tgt_padding=batch_d.get("tgt_padding") if use_teacher else None,
            src_key_padding_mask=mask,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # ── 직렬화 가능한 결과 추출 ────────────────────────────────────────────
    result = {
        "sample_id": (batch.get("sample_id") or ["?"])[0],
        "domain": (batch.get("domain") or ["?"])[0],
        "split_group": (batch.get("split_group") or ["test"])[0],
        "korean_ref": (batch.get("korean_text") or [""])[0],
        "gloss_ref": (batch.get("gloss_tokens_raw") or [[]])[0],
        "seq_len": seq_len,
        "elapsed_ms": round(elapsed_ms, 1),
    }

    # presence
    pm = batch_d.get("presence_mask")
    if pm is not None:
        pm_np = pm[0, :seq_len].bool().cpu().numpy()
        result["presence"] = {
            name: int(pm_np[:, i].sum()) for i, name in enumerate(_PRESENCE_NAMES) if i < pm_np.shape[-1]
        }
    else:
        result["presence"] = {name: seq_len for name in _PRESENCE_NAMES}

    # CTC timeline
    gloss_logits = outputs.get("gloss_logits")
    if gloss_logits is not None:
        probs = gloss_logits[0, :seq_len].softmax(dim=-1).cpu()
        argmax_ids = probs.argmax(dim=-1).tolist()
        argmax_probs = probs.max(dim=-1).values.tolist()

        n_buckets = min(16, seq_len)
        bsize = max(1, seq_len // n_buckets)
        timeline = []
        for b in range(0, seq_len, bsize):
            end = min(b + bsize, seq_len)
            bucket_ids = argmax_ids[b:end]
            bucket_probs = argmax_probs[b:end]
            from collections import Counter as _Counter
            cnt = _Counter(i for i in bucket_ids if i != 0)
            if cnt:
                top_id = cnt.most_common(1)[0][0]
                avg_c = sum(p for i, p in zip(bucket_ids, bucket_probs) if i == top_id) / cnt[top_id]
                words = gv.decode([top_id])
                label = words[0] if words else str(top_id)
            else:
                label = "[blank]"
                avg_c = sum(bucket_probs) / max(len(bucket_probs), 1)
            timeline.append({"frames": f"f{b+1:04d}-{end:04d}", "token": label, "conf": round(avg_c, 3)})
        result["ctc_timeline"] = timeline

        # CTC collapsed
        prev, emitted = -1, []
        for t_i, tok_id in enumerate(argmax_ids):
            if tok_id != prev and tok_id != 0:
                emitted.append((tok_id, float(probs[t_i, tok_id])))
            prev = tok_id
        emitted = emitted[:10]
        words_e = gv.decode([t for t, _ in emitted])
        result["gloss_hyp"] = [(w, round(c, 3)) for w, (_, c) in zip(words_e, emitted)]
    else:
        result["ctc_timeline"] = []
        result["gloss_hyp"] = []

    # NMS
    nms_logits = outputs.get("nms_logits")
    if nms_logits is not None:
        nms_probs = torch.sigmoid(nms_logits[0, :seq_len]).mean(dim=0).cpu()
        result["nms"] = {k: round(float(nms_probs[i]), 3) for i, k in enumerate(_NMS_KEYS) if i < len(nms_probs)}
    else:
        result["nms"] = {}

    # NMS detail
    from src.data.signals import NMS_DETAIL_CLASSES
    result["nms_detail"] = {}
    for group, classes in NMS_DETAIL_CLASSES.items():
        logits_d = outputs.get(f"nms_{group}_logits")
        if logits_d is None:
            continue
        p_d = torch.softmax(logits_d[0, :seq_len], dim=-1).mean(dim=0).cpu()
        cls_idx = int(p_d.argmax().item())
        result["nms_detail"][group] = {
            "label": classes[cls_idx],
            "conf": round(float(p_d[cls_idx]), 3),
            "all": {c: round(float(p_d[j]), 3) for j, c in enumerate(classes)},
        }

    # Intent
    intent_logits = outputs.get("intent_logits")
    if intent_logits is not None:
        intent_probs = torch.softmax(intent_logits[0], dim=-1).cpu()
        result["intent"] = {d: round(float(intent_probs[i]), 3) for i, d in enumerate(_DOMAIN_LIST)}
        result["intent_pred"] = _DOMAIN_LIST[int(intent_probs.argmax().item())]
    else:
        result["intent"] = {}
        result["intent_pred"] = "unknown"
    result["intent_gt"] = (batch.get("domain") or ["unknown"])[0]

    # Boundary
    bnd_logits = outputs.get("boundary_logits")
    if bnd_logits is not None:
        bnd_preds = bnd_logits[0, :seq_len].argmax(dim=-1).cpu().tolist()
        from collections import Counter as _Counter2
        cnt2 = _Counter2(bnd_preds)
        result["boundary"] = {name: cnt2.get(c, 0) for c, name in [(0, "idle"), (1, "signing"), (2, "boundary")]}
        result["boundary_last"] = {0: "idle", 1: "ongoing", 2: "ended"}.get(bnd_preds[-1] if bnd_preds else 0, "ended")
        if "activity" in batch_d:
            gt_bnd = batch_d["activity"][0, :seq_len].cpu().tolist()
            from collections import Counter as _Counter3
            result["boundary_gt"] = dict(_Counter3(gt_bnd))
            try:
                from src.eval.metrics import compute_f1
                f1d = compute_f1(bnd_preds, gt_bnd, num_classes=3)
                result["boundary_f1"] = round(f1d["f1"], 3)
            except Exception:
                result["boundary_f1"] = None
    else:
        result["boundary"] = {}
        result["boundary_last"] = "ended"
        result["boundary_gt"] = {}
        result["boundary_f1"] = None

    # Draft text
    draft = ""
    if "draft_logits" in outputs and tok is not None:
        ids_d = outputs["draft_logits"][0].argmax(dim=-1).tolist()
        draft = tok.decode(ids_d, skip_special_tokens=True).strip()
    elif "draft_tokens" in outputs and tok is not None:
        draft = tok.decode(outputs["draft_tokens"][0].tolist(), skip_special_tokens=True).strip()
    result["draft"] = draft

    # Per-sample BLEU/chrF
    try:
        from sacrebleu.metrics import BLEU, CHRF
        result["bleu"] = round(BLEU(effective_order=True).corpus_score([draft], [[result["korean_ref"]]]).score, 2)
        result["chrf"] = round(CHRF().corpus_score([draft], [[result["korean_ref"]]]).score, 2)
    except Exception:
        result["bleu"] = 0.0
        result["chrf"] = 0.0

    # Gloss WER
    hyp_words = [w for w, _ in result["gloss_hyp"]]
    ref_words = result["gloss_ref"]
    if ref_words:
        from scripts.run_eval_trace import _wer_single
        result["gloss_wer"] = round(_wer_single(hyp_words, ref_words), 3)
    else:
        result["gloss_wer"] = None

    return result


# ── UI ────────────────────────────────────────────────────────────────────────

def _prob_bar_html(prob: float, width: int = 200, color: str = "#4c78a8") -> str:
    w = int(prob * width)
    return (
        f'<div style="display:inline-block;background:#eee;width:{width}px;height:14px;border-radius:3px">'
        f'<div style="background:{color};width:{w}px;height:14px;border-radius:3px"></div>'
        f'</div> <span style="font-size:0.85em">{prob:.3f}</span>'
    )


def _render_overview(results: list[dict]) -> None:
    st.subheader("전체 집계")
    bleus = [r["bleu"] for r in results]
    chrfs = [r["chrf"] for r in results]
    wers  = [r["gloss_wer"] for r in results if r.get("gloss_wer") is not None]
    ms    = [r["elapsed_ms"] for r in results]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("avg BLEU",     f"{sum(bleus)/len(bleus):.2f}")
    col2.metric("avg chrF",     f"{sum(chrfs)/len(chrfs):.2f}")
    col3.metric("avg Gloss WER", f"{sum(wers)/len(wers):.3f}" if wers else "—")
    col4.metric("avg 추론(ms)",  f"{sum(ms)/len(ms):.0f}")
    col5.metric("샘플 수",       len(results))

    # 도메인 분포
    from collections import Counter
    domain_cnt = Counter(r["domain"] for r in results)
    st.write("**도메인 분포**")
    dcols = st.columns(len(domain_cnt))
    for col, (d, n) in zip(dcols, domain_cnt.most_common()):
        col.metric(_DOMAIN_KO.get(d, d), n)

    # BLEU 분포 바 차트
    import pandas as pd
    df = pd.DataFrame([
        {"sample": r["sample_id"][-30:], "BLEU": r["bleu"], "chrF": r["chrf"],
         "domain": _DOMAIN_KO.get(r["domain"], r["domain"])}
        for r in sorted(results, key=lambda x: x["bleu"])
    ])
    st.write("**샘플별 BLEU (오름차순)**")
    st.bar_chart(df.set_index("sample")["BLEU"])


def _render_sample(r: dict) -> None:
    import pandas as pd

    st.markdown(f"### 샘플: `{r['sample_id']}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("도메인", _DOMAIN_KO.get(r["domain"], r["domain"]))
    c2.metric("BLEU", r["bleu"])
    c3.metric("chrF", r["chrf"])
    c4.metric("추론(ms)", r["elapsed_ms"])

    # ── 메타데이터 ─────────────────────────────────────────────────────────
    with st.expander("📋 메타데이터 & 정답", expanded=True):
        st.write(f"**GT Korean:** {r['korean_ref'] or '(없음)'}")
        st.write(f"**GT Gloss:** {r['gloss_ref'] or '(없음)'}")
        st.write(f"**seq_len:** {r['seq_len']} frames")

    # ── 키포인트 (MediaPipe) ───────────────────────────────────────────────
    with st.expander("🖐 MediaPipe 키포인트 검출", expanded=True):
        pres = r.get("presence", {})
        seq_len = r["seq_len"]
        rows = []
        for name in _PRESENCE_NAMES:
            detected = pres.get(name, seq_len)
            pct = detected / max(seq_len, 1) * 100
            rows.append({"모달리티": name, "검출 프레임": detected,
                          "전체": seq_len, "검출률(%)": round(pct, 1)})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.progress(min(1.0, sum(pres.values()) / (len(_PRESENCE_NAMES) * seq_len)),
                    text="전체 키포인트 커버리지")

    # ── CTC 타임라인 ──────────────────────────────────────────────────────
    with st.expander("⏱ CTC Gloss 타임라인", expanded=True):
        timeline = r.get("ctc_timeline", [])
        if timeline:
            df_t = pd.DataFrame(timeline)
            st.dataframe(df_t, use_container_width=True, hide_index=True)
        st.write("**CTC Collapsed (최종 gloss 시퀀스):**")
        gloss_hyp = r.get("gloss_hyp", [])
        if gloss_hyp:
            for i, (w, c) in enumerate(gloss_hyp):
                st.write(f"  `{i+1}.` **{w}** — conf={c:.3f}  "
                         + "█" * int(c * 20) + "░" * (20 - int(c * 20)))
        gloss_ref = r.get("gloss_ref", [])
        hyp_words = [w for w, _ in gloss_hyp]
        if gloss_ref:
            from scripts.run_eval_trace import _wer_single
            wer = _wer_single(hyp_words, gloss_ref)
            st.metric("Gloss WER (this sample)", f"{wer:.3f}")
            st.write(f"**GT Gloss:** {gloss_ref}")

    # ── NMS ───────────────────────────────────────────────────────────────
    with st.expander("😐 NMS (비수지 신호)", expanded=True):
        nms = r.get("nms", {})
        if nms:
            df_nms = pd.DataFrame(
                [{"signal": k, "prob": v, "HIGH": "◀" if v >= 0.5 else ""} for k, v in nms.items()]
            )
            st.dataframe(df_nms, use_container_width=True, hide_index=True)
            import altair as alt
            chart = alt.Chart(df_nms).mark_bar().encode(
                x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("signal:N", sort="-x"),
                color=alt.condition(
                    alt.datum.prob >= 0.5,
                    alt.value("#e45756"),
                    alt.value("#4c78a8"),
                ),
                tooltip=["signal", "prob"],
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

        nms_detail = r.get("nms_detail", {})
        if nms_detail:
            st.write("**세부 카테고리:**")
            rows2 = []
            for group, info in nms_detail.items():
                rows2.append({"group": group, "pred": info["label"],
                              "conf": info["conf"], "all": str(info["all"])})
            st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    # ── Intent ────────────────────────────────────────────────────────────
    with st.expander("🎯 Intent 예측", expanded=True):
        intent = r.get("intent", {})
        if intent:
            pred = r.get("intent_pred", "")
            gt   = r.get("intent_gt", "")
            correct = "✅" if pred == gt else "❌"
            st.write(f"예측: **{_DOMAIN_KO.get(pred, pred)}**  |  GT: **{_DOMAIN_KO.get(gt, gt)}** {correct}")
            import altair as alt
            df_i = pd.DataFrame([
                {"domain": _DOMAIN_KO.get(d, d), "prob": p,
                 "type": "예측" if d == pred else ("GT" if d == gt else "기타")}
                for d, p in intent.items()
            ])
            chart_i = alt.Chart(df_i).mark_bar().encode(
                x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("domain:N", sort="-x"),
                color=alt.Color("type:N", scale=alt.Scale(
                    domain=["예측", "GT", "기타"],
                    range=["#e45756", "#72b7b2", "#4c78a8"],
                )),
                tooltip=["domain", "prob", "type"],
            ).properties(height=220)
            st.altair_chart(chart_i, use_container_width=True)

    # ── Boundary ──────────────────────────────────────────────────────────
    with st.expander("⚡ Boundary / Activity", expanded=False):
        bnd = r.get("boundary", {})
        bnd_gt = r.get("boundary_gt", {})
        seq_len = r["seq_len"]
        if bnd:
            df_b = pd.DataFrame([
                {"class": name, "pred_frames": bnd.get(name, 0),
                 "gt_frames": bnd_gt.get(cls, 0),
                 "pred_pct": round(bnd.get(name, 0) / max(seq_len, 1) * 100, 1)}
                for cls, name in [(0, "idle"), (1, "signing"), (2, "boundary")]
            ])
            st.dataframe(df_b, use_container_width=True, hide_index=True)
        f1 = r.get("boundary_f1")
        if f1 is not None:
            st.metric("Boundary F1 (macro)", f"{f1:.3f}")
        st.write(f"Final state: **{r.get('boundary_last', '?')}**")

    # ── 번역 ──────────────────────────────────────────────────────────────
    with st.expander("🇰🇷 번역 비교", expanded=True):
        draft = r.get("draft", "")
        ref   = r.get("korean_ref", "")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**예측 (draft)**")
            st.info(draft or "(없음)")
        with col_r:
            st.markdown("**정답 (GT)**")
            st.success(ref or "(없음)")

        if draft and ref:
            import difflib
            d = difflib.SequenceMatcher(None, ref, draft)
            ops = d.get_opcodes()
            diff_parts = []
            for op, i1, i2, j1, j2 in ops:
                if op == "equal":
                    diff_parts.append(ref[i1:i2])
                elif op == "replace":
                    diff_parts.append(f"~~{ref[i1:i2]}~~ **{draft[j1:j2]}**")
                elif op == "insert":
                    diff_parts.append(f"**+{draft[j1:j2]}**")
                elif op == "delete":
                    diff_parts.append(f"~~{ref[i1:i2]}~~")
            st.markdown("**Diff** (~~삭제~~ / **추가**): " + "".join(diff_parts))

        st.metric("BLEU", r["bleu"])
        st.metric("chrF", r["chrf"])


# ── 메인 UI 진입점 ─────────────────────────────────────────────────────────────

def main():
    st.title("🤟 KSL 추론 대시보드")
    st.caption("KeypointDataset → 모델 → CTC gloss → 번역 전 과정 시각화")

    # ── 사이드바 ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("설정")

        # trace JSON 우선
        if args.trace_json and Path(args.trace_json).exists():
            st.success(f"trace JSON: {args.trace_json}")
            with open(args.trace_json, encoding="utf-8") as f:
                trace_data = json.load(f)
            samples_data = trace_data.get("samples", [])
            mode = "json"
        elif args.checkpoint:
            st.success(f"모델: {Path(args.checkpoint).name}")
            mode = "live"
            samples_data = None
        else:
            st.error("--checkpoint 또는 --trace_json 을 지정하세요.")
            st.stop()

        num_samples = st.slider("샘플 수", 1, 50, args.num_samples)
        show_overview = st.checkbox("전체 집계 보기", value=True)

    # ── 샘플 추론 ────────────────────────────────────────────────────────────
    if mode == "live":
        import random
        random.seed(args.seed)
        _, _, _, dataset, _ = load_model_and_data()
        n = min(num_samples, len(dataset))
        all_indices = random.sample(range(len(dataset)), n)
        all_indices = sorted(all_indices)

        results = []
        prog = st.progress(0, text="샘플 추론 중...")
        for i, idx in enumerate(all_indices):
            r = run_inference_for_index(id(load_model_and_data), idx)
            results.append(r)
            prog.progress((i + 1) / n, text=f"추론 중... {i+1}/{n}")
        prog.empty()
    else:
        results = samples_data[:num_samples]

    # ── 탭 구성 ─────────────────────────────────────────────────────────────
    tab_overview, tab_samples = st.tabs(["📊 전체 집계", "🔍 샘플별 상세"])

    with tab_overview:
        if show_overview and results:
            _render_overview(results)
        elif not results:
            st.info("결과가 없습니다.")

    with tab_samples:
        if not results:
            st.info("결과가 없습니다.")
        else:
            options = [f"[{i+1}] {r['sample_id'][-40:]}" for i, r in enumerate(results)]
            sel = st.selectbox("샘플 선택", options)
            sel_idx = options.index(sel)
            _render_sample(results[sel_idx])


if __name__ == "__main__":
    main()
