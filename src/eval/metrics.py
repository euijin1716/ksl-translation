"""평가 메트릭 모음.

BLEU, chrF, NMS F1, boundary F1, intent accuracy.
실제 라이브러리가 없으면 간이 구현을 사용한다.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 4,
) -> dict[str, float]:
    """간이 corpus BLEU 계산.

    Args:
        hypotheses: 예측 문장 목록
        references: 정답 문장 목록 (1-ref)
        max_n: 최대 n-gram 차수

    Returns:
        {"bleu": ..., "bleu_1": ..., ..., "bleu_4": ...}
    """
    # sacrebleu가 설치되어 있으면 위임
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU
        # effective_order=True: 문장이 짧아 상위 n-gram이 없을 때 해당 차수를 제외하고 계산
        bleu = SacreBLEU(effective_order=True)
        result = bleu.corpus_score(hypotheses, [references])
        return {
            "bleu": result.score,
            "bleu_1": result.precisions[0],
            "bleu_2": result.precisions[1],
            "bleu_3": result.precisions[2],
            "bleu_4": result.precisions[3],
        }
    except ImportError:
        pass

    # 간이 구현
    precisions = []
    for n in range(1, max_n + 1):
        match, total = 0, 0
        for hyp, ref in zip(hypotheses, references):
            h_ngrams = _ngrams(hyp.split(), n)
            r_ngrams = _ngrams(ref.split(), n)
            h_count = Counter(h_ngrams)
            r_count = Counter(r_ngrams)
            clipped = {k: min(v, r_count[k]) for k, v in h_count.items()}
            match += sum(clipped.values())
            total += sum(h_count.values())
        precisions.append(match / max(total, 1))

    # brevity penalty
    hyp_len = sum(len(h.split()) for h in hypotheses)
    ref_len = sum(len(r.split()) for r in references)
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    bleu = bp * math.exp(
        sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    )
    result = {"bleu": bleu * 100}
    for i, p in enumerate(precisions):
        result[f"bleu_{i+1}"] = p * 100
    return result


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ── chrF ──────────────────────────────────────────────────────────────────────

def compute_chrf(
    hypotheses: list[str],
    references: list[str],
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    """간이 chrF 계산."""
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
        return chrf.corpus_score(hypotheses, [references]).score
    except ImportError:
        pass

    scores = []
    for hyp, ref in zip(hypotheses, references):
        h_ngrams = Counter(_char_ngrams(hyp, char_order))
        r_ngrams = Counter(_char_ngrams(ref, char_order))
        match = sum(min(h_ngrams[k], r_ngrams[k]) for k in h_ngrams)
        p = match / max(sum(h_ngrams.values()), 1)
        r = match / max(sum(r_ngrams.values()), 1)
        if p + r > 0:
            f = (1 + beta**2) * p * r / (beta**2 * p + r)
        else:
            f = 0.0
        scores.append(f)
    return sum(scores) / max(len(scores), 1) * 100


def _char_ngrams(text: str, n: int) -> list[str]:
    return [text[i:i+n] for i in range(len(text) - n + 1)]


# ── Classification metrics ─────────────────────────────────────────────────────

def compute_f1(
    preds: list[int],
    labels: list[int],
    num_classes: int,
    average: str = "macro",
) -> dict[str, float]:
    """F1 score 계산 (macro 또는 binary)."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for p, l in zip(preds, labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        pr = tp[c] / max(tp[c] + fp[c], 1)
        rc = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-10)
        precisions.append(pr)
        recalls.append(rc)
        f1s.append(f1)

    if average == "macro":
        return {
            "precision": sum(precisions) / num_classes,
            "recall": sum(recalls) / num_classes,
            "f1": sum(f1s) / num_classes,
        }
    return {"precision": precisions[1], "recall": recalls[1], "f1": f1s[1]}


def compute_accuracy(preds: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


# ── WER (gloss) ───────────────────────────────────────────────────────────────

def compute_wer(hypotheses: list[list[str]], references: list[list[str]]) -> float:
    """Word Error Rate 계산 (Levenshtein)."""
    total_dist, total_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        total_dist += _levenshtein(hyp, ref)
        total_len += len(ref)
    return total_dist / max(total_len, 1)


def _levenshtein(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]
