"""LLM 보정 효과 테스트.

ClaudeAdapter → ContextCorrector 전체 흐름을 합성 데이터로 검증한다.
BLEU / ChrF 전후 비교로 보정 효과를 수치화한다.

실행:
    pytest tests/test_llm_correction.py -v -s          # 표 출력 포함
    pytest tests/test_llm_correction.py -v             # 간단 통과 여부만
    python tests/test_llm_correction.py                # 단독 실행 (표 출력)
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# sacrebleu 없으면 전체 모듈 skip
pytest.importorskip("sacrebleu")

from sacrebleu.metrics import BLEU, CHRF

from src.llm.adapters.claude_adapter import ClaudeAdapter
from src.llm.corrector import ContextCorrector

# 수어 인식 모델이 생성할 법한 노이즈 패턴
# (조사·어미 탈락, 어순 어색, 문장 부호 없음, 구어체 글로스 나열)
CASES = [
    dict(
        draft="접수 어디 해요",
        ground_truth="어디서 접수하면 되나요?",
        gloss=["접수", "위치-묻다"],
        nms={"eyebrow": "raised", "head": "tilt"},
        confidence=0.80,
        domain="hospital",
    ),
    dict(
        draft="약 처방전 받고 싶어",
        ground_truth="처방전을 받고 싶어요.",
        gloss=["약", "처방전", "받다-원하다"],
        nms={"mouth": "open", "eyebrow": "neutral"},
        confidence=0.65,
        domain="hospital",
    ),
    dict(
        draft="지하철역 가는 길 알려줘",
        ground_truth="지하철역으로 가는 길을 알려주세요.",
        gloss=["지하철역", "길-알려주다"],
        nms={"eyebrow": "raised", "head": "forward"},
        confidence=0.88,
        domain="directions",
    ),
    dict(
        draft="여기서 왼쪽 가면 편의점 있어",
        ground_truth="여기서 왼쪽으로 가면 편의점이 있어요.",
        gloss=["왼쪽", "가다", "편의점-있다"],
        nms={"head": "nod", "eyebrow": "neutral"},
        confidence=0.92,
        domain="directions",
    ),
    dict(
        draft="아메리카노 두 개 주세요 카드 결제",
        ground_truth="아메리카노 두 잔 주세요. 카드로 결제할게요.",
        gloss=["아메리카노", "두-개", "카드-결제"],
        nms={"mouth": "open", "eyebrow": "raised"},
        confidence=0.78,
        domain="order",
    ),
    dict(
        draft="영수증 필요 없어요",
        ground_truth="영수증은 필요 없어요.",
        gloss=["영수증", "필요-없다"],
        nms={"head": "shake", "eyebrow": "neutral"},
        confidence=0.95,
        domain="order",
    ),
    dict(
        draft="내일 오후 세시 두 명 예약 되나요",
        ground_truth="내일 오후 세 시에 두 명 예약이 가능한가요?",
        gloss=["내일", "오후-세시", "두-명", "예약-가능"],
        nms={"eyebrow": "raised", "head": "forward"},
        confidence=0.83,
        domain="reservation",
    ),
    dict(
        draft="예약 취소하고 싶어요 확인 해줘",
        ground_truth="예약을 취소하고 싶어요. 확인해 주세요.",
        gloss=["예약-취소", "확인-요청"],
        nms={"eyebrow": "raised", "mouth": "open"},
        confidence=0.70,
        domain="reservation",
    ),
    dict(
        draft="주민등록등본 어떻게 발급해요",
        ground_truth="주민등록등본은 어떻게 발급받을 수 있나요?",
        gloss=["주민등록등본", "발급-방법"],
        nms={"eyebrow": "raised", "head": "tilt"},
        confidence=0.75,
        domain="public",
    ),
    dict(
        draft="도움 필요해 빨리",
        ground_truth="도움이 필요해요. 빨리 와 주세요.",
        gloss=["도움-필요", "빨리"],
        nms={"eyebrow": "raised", "mouth": "open", "head": "forward"},
        confidence=0.60,
        domain="help",
    ),
]


def _make_corrector() -> ContextCorrector:
    adapter = ClaudeAdapter(model="claude-sonnet-4-6")
    return ContextCorrector(provider=adapter)


def _score(bleu: BLEU, chrf: CHRF, hypothesis: str, reference: str) -> tuple[float, float]:
    return (
        bleu.sentence_score(hypothesis, [reference]).score,
        chrf.sentence_score(hypothesis, [reference]).score,
    )


def _print_table(results: list[dict]) -> None:
    col = "{:<14} {:>5}  {:<30} {:<30} {:<30}  {:^14} {:^14}"
    print()
    print(col.format("도메인", "신뢰도", "초안", "보정 후", "정답", "BLEU전→후", "ChrF전→후"))
    print("-" * 138)
    for r in results:
        bleu_str = "{:.1f}→{:.1f}{}".format(
            r["draft_bleu"], r["corr_bleu"],
            "↑" if r["corr_bleu"] > r["draft_bleu"] else ("=" if r["corr_bleu"] == r["draft_bleu"] else "↓"),
        )
        chrf_str = "{:.1f}→{:.1f}{}".format(
            r["draft_chrf"], r["corr_chrf"],
            "↑" if r["corr_chrf"] > r["draft_chrf"] else ("=" if r["corr_chrf"] == r["draft_chrf"] else "↓"),
        )
        print(col.format(
            r["domain"], r["confidence"],
            r["draft"][:30], r["corrected"][:30], r["ground_truth"][:30],
            bleu_str, chrf_str,
        ))
    n = len(results)
    avg_db = sum(r["draft_bleu"] for r in results) / n
    avg_cb = sum(r["corr_bleu"] for r in results) / n
    avg_dc = sum(r["draft_chrf"] for r in results) / n
    avg_cc = sum(r["corr_chrf"] for r in results) / n
    print("-" * 138)
    print(
        "평균 (n={})  BLEU: {:.1f} → {:.1f} (+{:.1f})   ChrF: {:.1f} → {:.1f} (+{:.1f})".format(
            n, avg_db, avg_cb, avg_cb - avg_db, avg_dc, avg_cc, avg_cc - avg_dc
        )
    )


# ---------------------------------------------------------------------------
# pytest 테스트
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corrector():
    return _make_corrector()


@pytest.fixture(scope="module")
def correction_results(corrector):
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    results = []
    for c in CASES:
        out = corrector.correct(
            korean_draft=c["draft"],
            gloss_hypotheses=c["gloss"],
            nms_summary=c["nms"],
            confidence=c["confidence"],
            domain=c["domain"],
        )
        db, dc = _score(bleu, chrf, c["draft"], c["ground_truth"])
        cb, cc = _score(bleu, chrf, out.final_text, c["ground_truth"])
        results.append({
            "domain": c["domain"],
            "confidence": c["confidence"],
            "draft": c["draft"],
            "corrected": out.final_text,
            "ground_truth": c["ground_truth"],
            "draft_bleu": db, "corr_bleu": cb,
            "draft_chrf": dc, "corr_chrf": cc,
            "retry_or_clarify": out.retry_or_clarify,
            "uncertain_spans": out.uncertain_spans,
        })
    _print_table(results)
    return results


def test_bleu_improves_on_average(correction_results):
    avg_draft = sum(r["draft_bleu"] for r in correction_results) / len(correction_results)
    avg_corr = sum(r["corr_bleu"] for r in correction_results) / len(correction_results)
    assert avg_corr > avg_draft, (
        f"평균 BLEU가 개선되지 않음: {avg_draft:.1f} → {avg_corr:.1f}"
    )


def test_chrf_improves_on_average(correction_results):
    avg_draft = sum(r["draft_chrf"] for r in correction_results) / len(correction_results)
    avg_corr = sum(r["corr_chrf"] for r in correction_results) / len(correction_results)
    assert avg_corr > avg_draft, (
        f"평균 ChrF가 개선되지 않음: {avg_draft:.1f} → {avg_corr:.1f}"
    )


def test_no_empty_output(correction_results):
    for r in correction_results:
        assert r["corrected"].strip(), f"빈 보정 결과: draft='{r['draft']}'"


def test_low_confidence_has_uncertain_spans(correction_results):
    """신뢰도 0.4 미만 케이스는 uncertain_spans가 있어야 한다."""
    low_conf = [r for r in correction_results if r["confidence"] < 0.4]
    for r in low_conf:
        assert r["uncertain_spans"], (
            f"low confidence({r['confidence']}) 케이스에 uncertain_spans 없음: '{r['draft']}'"
        )


# ---------------------------------------------------------------------------
# 단독 실행
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corrector = _make_corrector()
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    results = []
    for c in CASES:
        out = corrector.correct(
            korean_draft=c["draft"],
            gloss_hypotheses=c["gloss"],
            nms_summary=c["nms"],
            confidence=c["confidence"],
            domain=c["domain"],
        )
        db, dc = _score(bleu, chrf, c["draft"], c["ground_truth"])
        cb, cc = _score(bleu, chrf, out.final_text, c["ground_truth"])
        results.append({
            "domain": c["domain"],
            "confidence": c["confidence"],
            "draft": c["draft"],
            "corrected": out.final_text,
            "ground_truth": c["ground_truth"],
            "draft_bleu": db, "corr_bleu": cb,
            "draft_chrf": dc, "corr_chrf": cc,
            "retry_or_clarify": out.retry_or_clarify,
            "uncertain_spans": out.uncertain_spans,
        })
    _print_table(results)
