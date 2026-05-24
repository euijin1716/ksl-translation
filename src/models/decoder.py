"""Korean draft decoder.

fused feature → Korean token sequence (seq2seq style, teacher-forcing).
Stage C에서 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    vocab_size: int = 32000      # klue/roberta-base 기준
    max_len: int = 128
    # klue/roberta-base 기준 특수토큰 ID:
    #   <s>(BOS/CLS)=0, <pad>=1, </s>(EOS/SEP)=2
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, **kwargs) -> "DecoderConfig":
        """HuggingFace tokenizer에서 vocab_size와 특수토큰 ID를 가져온다.

        bos_token_id가 None이면 cls_token_id로 fallback.
        eos_token_id가 None이면 sep_token_id로 fallback.
        """
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.cls_token_id
        if bos is None:
            bos = 0

        eos = tokenizer.eos_token_id
        if eos is None:
            eos = tokenizer.sep_token_id
        if eos is None:
            eos = 2

        pad = tokenizer.pad_token_id
        if pad is None:
            pad = 1

        return cls(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=pad,
            bos_token_id=bos,
            eos_token_id=eos,
            **kwargs,
        )


class KoreanDraftDecoder(nn.Module):
    """Transformer decoder for Korean draft generation.

    입력:
        encoder_out:    [B, T, d_model]  (fused stream features)
        tgt_tokens:     [B, L]           (teacher-forcing 시 정답 토큰, 없으면 greedy)
        tgt_padding:    [B, L] bool      (True=pad)
        memory_padding: [B, T] bool      (True=pad)

    출력:
        logits: [B, L, vocab_size]
    """

    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        c = self.config

        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=c.pad_token_id)
        self.pos_enc = nn.Embedding(c.max_len, c.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
        )
        self.transformer_dec = nn.TransformerDecoder(decoder_layer, num_layers=c.num_layers)
        self.out_proj = nn.Linear(c.d_model, c.vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_padding: torch.Tensor | None = None,
        memory_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Args:
            encoder_out:    [B, T, d_model]
            tgt_tokens:     [B, L] int (입력 토큰, 보통 BOS + 정답[:-1])
            tgt_padding:    [B, L] bool
            memory_padding: [B, T] bool

        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = tgt_tokens.shape
        tgt_emb = self.tok_emb(tgt_tokens)
        pos = torch.arange(L, device=tgt_tokens.device).unsqueeze(0)
        tgt_emb = tgt_emb + self.pos_enc(pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt_tokens.device)

        out = self.transformer_dec(
            tgt=tgt_emb,
            memory=encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=memory_padding,
        )
        return self.out_proj(out)   # [B, L, vocab_size]

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Greedy decoding (추론 시 사용).

        Returns:
            token_ids: [B, L] (EOS 포함)
        """
        c = self.config
        max_len = max_len or c.max_len
        B = encoder_out.shape[0]
        device = encoder_out.device

        tokens = torch.full((B, 1), c.bos_token_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(encoder_out, tokens, memory_padding=memory_padding)
            next_tok = logits[:, -1, :].argmax(dim=-1)   # [B]
            tokens = torch.cat([tokens, next_tok.unsqueeze(-1)], dim=1)
            done = done | (next_tok == c.eos_token_id)
            if done.all():
                break

        return tokens
