"""Gloss 어휘 사전.

gloss token(문자열) ↔ 정수 인덱스 변환.
CTC 학습에 맞춰 index 0을 blank로 예약한다.

사용법:
    # 학습 manifest에서 자동 구축
    vocab = GlossVocab.build_from_manifest("data/manifests/train.jsonl")
    vocab.save("data/gloss_vocab.json")

    # 로드 후 사용
    vocab = GlossVocab.load("data/gloss_vocab.json")
    ids = vocab.encode(["배", "아프다"])   # → [3, 7]
    tokens = vocab.decode([3, 7])          # → ["배", "아프다"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class GlossVocab:
    """Gloss 어휘 사전.

    특수 인덱스:
        0 = <blank>  (CTC blank)
        1 = <unk>    (미등록 토큰)
    """

    BLANK_ID = 0
    UNK_ID = 1
    BLANK_TOKEN = "<blank>"
    UNK_TOKEN = "<unk>"

    def __init__(self, token2id: dict[str, int]) -> None:
        self._token2id = token2id
        self._id2token = {v: k for k, v in token2id.items()}

    # ── 구축 ─────────────────────────────────────────────────────────────────

    @classmethod
    def build_from_manifest(cls, manifest_path: str | Path) -> "GlossVocab":
        """manifest JSONL에서 gloss_tokens를 읽어 어휘 사전을 구축한다.

        gloss가 없는 샘플(None)은 건너뛴다.
        """
        import json as _json

        tokens: set[str] = set()
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = _json.loads(line)
                gloss = d.get("gloss_tokens")
                if gloss:
                    tokens.update(gloss)

        return cls._from_token_set(tokens)

    @classmethod
    def build_from_samples(cls, samples: list) -> "GlossVocab":
        """KSLSample 리스트에서 어휘 사전을 구축한다."""
        tokens: set[str] = set()
        for s in samples:
            if s.gloss_tokens:
                tokens.update(s.gloss_tokens)
        return cls._from_token_set(tokens)

    @classmethod
    def _from_token_set(cls, tokens: set[str]) -> "GlossVocab":
        token2id: dict[str, int] = {
            cls.BLANK_TOKEN: cls.BLANK_ID,
            cls.UNK_TOKEN: cls.UNK_ID,
        }
        for tok in sorted(tokens):
            if tok not in token2id:
                token2id[tok] = len(token2id)
        return cls(token2id)

    # ── 직렬화 ────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self._token2id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GlossVocab":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["token2id"])

    # ── 변환 ─────────────────────────────────────────────────────────────────

    def encode(self, tokens: list[str]) -> list[int]:
        """토큰 리스트 → ID 리스트. 미등록 토큰은 UNK_ID."""
        return [self._token2id.get(t, self.UNK_ID) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """ID 리스트 → 토큰 리스트."""
        return [self._id2token.get(i, self.UNK_TOKEN) for i in ids]

    def __len__(self) -> int:
        return len(self._token2id)

    @property
    def vocab_size(self) -> int:
        return len(self._token2id)

    def __contains__(self, token: str) -> bool:
        return token in self._token2id
