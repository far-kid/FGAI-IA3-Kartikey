"""Small encoder-decoder Transformer trained from scratch."""

import math
from pathlib import Path

import sentencepiece as spm
import torch
from torch import nn


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, heads: int = 8, layers: int = 4, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, heads=heads, layers=layers, ff_dim=ff_dim, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(256, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=heads, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.embedding(ids) * math.sqrt(self.config["d_model"]) + self.position(positions)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        length = target_input.size(1)
        causal_mask = torch.triu(torch.ones(length, length, device=source.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(
            self.embed(source), self.embed(target_input),
            tgt_mask=causal_mask,
            src_key_padding_mask=source.eq(0),
            tgt_key_padding_mask=target_input.eq(0),
            memory_key_padding_mask=source.eq(0),
        )
        return self.output(hidden)

    @torch.no_grad()
    def translate(self, source: torch.Tensor, max_new_tokens: int = 80) -> torch.Tensor:
        self.eval()
        target = torch.full((source.size(0), 1), 2, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_new_tokens):
            next_id = self(source, target)[:, -1].argmax(dim=-1)
            next_id = torch.where(finished, 3, next_id)
            target = torch.cat([target, next_id[:, None]], dim=1)
            finished |= next_id.eq(3)
            if finished.all():
                break
        return target


def train_sentencepiece(rows: list[dict], output: Path, vocab_size: int = 8000) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    corpus = output / "spm_corpus.txt"
    with corpus.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["source"] + "\n" + row["target"] + "\n")
    prefix = output / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), vocab_size=vocab_size,
        model_type="bpe", pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        character_coverage=1.0, hard_vocab_limit=False,
    )
    corpus.unlink()
    return prefix.with_suffix(".model")


def load_sentencepiece(path: Path) -> spm.SentencePieceProcessor:
    processor = spm.SentencePieceProcessor()
    processor.load(str(path))
    return processor
