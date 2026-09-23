"""Generative response model: a small encoder-decoder Transformer.

Architecture follows "Attention Is All You Need" (Vaswani et al., 2017):
sinusoidal positional encoding, multi-head self/cross attention,
encoder-decoder layout, causal (look-ahead) masking on the decoder,
trained with teacher forcing on (user message -> bot response) pairs.

This is intentionally small so it trains on CPU in minutes. It is a
demonstrative generative model, not a large LM.
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings(
    "ignore", message=".*nested tensors is in prototype stage.*"
)

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_MAX_SRC_LEN = 48
_MAX_TGT_LEN = 48


def gen_tokenize(text: str) -> List[str]:
    """Word-level tokenizer keeping punctuation as separate tokens."""
    return _TOKEN_RE.findall((text or "").lower())


class GenVocab:
    """Word-level vocabulary with special tokens."""

    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq
        self.itos: List[str] = list(SPECIAL_TOKENS)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    def build(self, texts: Sequence[str]) -> None:
        from collections import Counter

        counts: Counter = Counter()
        for t in texts:
            counts.update(gen_tokenize(t))
        for tok, c in counts.most_common():
            if c >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [self.stoi.get(t, UNK) for t in gen_tokenize(text)]
        return ids[:max_len]

    def decode(self, ids: Sequence[int]) -> str:
        words = []
        for i in ids:
            if i in (PAD, BOS):
                continue
            if i == EOS:
                break
            words.append(self.itos[i] if i < len(self.itos) else "<unk>")
        text = " ".join(words)
        # tidy spacing around punctuation and contraction endings
        text = re.sub(r"\s+([?!.,;:])", r"\1", text)
        text = re.sub(
            r"(\w) '\s*(m|s|t|re|ve|ll|d|em|cause)\b", r"\1'\2", text
        )
        return text.strip()

    def state_dict(self) -> Dict:
        return {"itos": self.itos, "min_freq": self.min_freq}

    def load_state_dict(self, state: Dict) -> None:
        self.itos = state["itos"]
        self.min_freq = state.get("min_freq", 2)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)


_UNICODE_MAP = {
    "—": "-", "–": "-", "―": "-", "−": "-",
    "'": "'", "'": "'", "‚": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "…": "...", "•": "-", "·": "-",
}


def to_ascii(text: str) -> str:
    """Map common unicode punctuation to ASCII so output renders on
    consoles without UTF-8 (e.g. Windows cp1252)."""
    for k, v in _UNICODE_MAP.items():
        text = text.replace(k, v)
    return unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore"
    ).decode("ascii")


def clean_reply(text: str) -> str:
    """Light post-processing for generated replies."""
    text = to_ascii(re.sub(r"\s+", " ", text).strip())
    if not text:
        return text
    text = re.sub(r"\bi\b", "I", text)          # standalone i -> I
    text = re.sub(r"\bim\b", "I'm", text)        # in case 'im' slipped through
    text = text[0].upper() + text[1:]
    # If generation cut off mid-sentence, keep the last complete sentence.
    if text[-1] not in ".!?":
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > 10:
            text = text[: last + 1]
    return text


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, as in the paper."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer for response generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 96,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)
        # weight tying: share input embedding and output projection
        self.out.weight = self.embed.weight

    def _scale(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pos_enc(self.embed(ids) * math.sqrt(self.d_model))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_pad = src == PAD
        tgt_pad = tgt == PAD
        t = tgt.size(1)
        tgt_mask = torch.triu(
            torch.ones(t, t, device=tgt.device, dtype=torch.bool), diagonal=1
        )
        out = self.transformer(
            self._scale(src),
            self._scale(tgt),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(out)

    @torch.no_grad()
    def generate_stream(
        self,
        src_ids: List[int],
        max_len: int = _MAX_TGT_LEN,
        temperature: float = 0.8,
        top_k: int = 6,
        device: Optional[torch.device] = None,
    ):
        """Autoregressive decoding; yields one token id at a time so callers
        can stream output word-by-word."""
        device = device or next(self.parameters()).device
        self.eval()
        src = torch.tensor([[BOS] + src_ids + [EOS]], device=device)
        src_pad = src == PAD
        memory = self.transformer.encoder(
            self._scale(src), src_key_padding_mask=src_pad
        )
        ys = torch.tensor([[BOS]], device=device)
        for _ in range(max_len):
            t = ys.size(1)
            tgt_mask = torch.triu(
                torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1
            )
            out = self.transformer.decoder(
                self._scale(ys),
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad,
            )
            logits = self.out(out[:, -1]) / max(temperature, 1e-5)
            logits[:, [PAD, BOS, UNK]] = float("-inf")  # never emit specials
            if top_k and top_k > 0:
                topv, topi = logits.topk(top_k)
                probs = torch.softmax(topv, dim=-1)
                next_id = topi.gather(-1, torch.multinomial(probs, 1))
            else:
                next_id = logits.argmax(-1, keepdim=True)
            token = next_id.item()
            ys = torch.cat([ys, next_id], dim=1)
            if token == EOS:
                break
            yield token

    @torch.no_grad()
    def generate_beam(
        self,
        src_ids: List[int],
        max_len: int = _MAX_TGT_LEN,
        beam_size: int = 4,
        length_penalty: float = 0.7,
        repetition_penalty: float = 1.2,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        """Beam search decoding — keeps the most probable sequences instead
        of sampling. More coherent than top-k for small models."""
        device = device or next(self.parameters()).device
        self.eval()
        src = torch.tensor([[BOS] + src_ids + [EOS]], device=device)
        src_pad = src == PAD
        memory = self.transformer.encoder(
            self._scale(src), src_key_padding_mask=src_pad
        )

        # beams: (token_id_list, cumulative_log_prob)
        beams: List[Tuple[List[int], float]] = [([BOS], 0.0)]
        for _ in range(max_len):
            candidates: List[Tuple[List[int], float]] = []
            for seq, score in beams:
                if seq[-1] == EOS:
                    candidates.append((seq, score))
                    continue
                ys = torch.tensor([seq], device=device)
                t = ys.size(1)
                tgt_mask = torch.triu(
                    torch.ones(t, t, device=device, dtype=torch.bool),
                    diagonal=1,
                )
                out = self.transformer.decoder(
                    self._scale(ys),
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_pad,
                )
                logits = self.out(out[:, -1])[0]
                # repetition penalty: divide positive logits, multiply negative
                for prev in set(seq[1:]):
                    if logits[prev] > 0:
                        logits[prev] /= repetition_penalty
                    else:
                        logits[prev] *= repetition_penalty
                logits[[PAD, BOS, UNK]] = float("-inf")
                log_probs = torch.log_softmax(logits, dim=-1)
                topv, topi = log_probs.topk(beam_size)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    candidates.append((seq + [i], score + v))

            # rank by length-normalized score
            beams = sorted(
                candidates,
                key=lambda c: c[1] / (len(c[0]) ** length_penalty),
                reverse=True,
            )[:beam_size]
            if all(seq[-1] == EOS for seq, _ in beams):
                break

        best = max(beams, key=lambda c: c[1] / (len(c[0]) ** length_penalty))
        return best[0][1:]  # drop BOS

    def generate(
        self,
        src_ids: List[int],
        max_len: int = _MAX_TGT_LEN,
        temperature: float = 0.8,
        top_k: int = 6,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        """Non-streaming decode; returns the full token id list."""
        return list(
            self.generate_stream(src_ids, max_len, temperature, top_k, device)
        )


class _PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[int], List[int]]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[List[int], List[int]]:
        return self.pairs[i]


def _collate(batch: List[Tuple[List[int], List[int]]]):
    src_max = max(len(s) for s, _ in batch)
    tgt_max = max(len(t) for _, t in batch)
    srcs, tgts_in, tgts_out = [], [], []
    for s, t in batch:
        src = [BOS] + s + [EOS]
        tin = [BOS] + t
        tout = t + [EOS]
        srcs.append(src + [PAD] * (src_max + 2 - len(src)))
        tgts_in.append(tin + [PAD] * (tgt_max + 1 - len(tin)))
        tgts_out.append(tout + [PAD] * (tgt_max + 1 - len(tout)))
    return (
        torch.tensor(srcs),
        torch.tensor(tgts_in),
        torch.tensor(tgts_out),
    )


@dataclass
class GenTrainResult:
    epochs: int
    final_loss: float
    num_pairs: int
    vocab_size: int


def build_pairs(
    documents: List[Tuple[List[str], List[str], str]],
    raw_texts_by_intent: Dict[str, List[str]],
    responses: Dict[str, List[str]],
    exclude_intents: Optional[set] = None,
    max_pairs: int = 60000,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Build (message, response) training pairs from intent data."""
    exclude = exclude_intents or set()
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for tag, texts in raw_texts_by_intent.items():
        if tag in exclude:
            continue
        # skip template responses — the model would learn to emit %%X%%
        resp = [
            r for r in responses.get(tag, [])
            if r.strip() and "%%" not in r
        ]
        if not resp:
            continue
        for text in texts:
            # prepend the intent tag so generation is conditioned on the
            # classified intent, not just the raw message
            src = f"{tag} {text}"
            for r in rng.sample(resp, min(2, len(resp))):
                pairs.append((src, r))
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def train_generator(
    pairs: List[Tuple[str, str]],
    vocab: GenVocab,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: Optional[str] = None,
    logger=None,
) -> Tuple[Seq2SeqTransformer, GenTrainResult]:
    """Train the transformer with teacher forcing."""
    log = logger.info if logger else print
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(42)

    encoded = [
        (
            vocab.encode(s, _MAX_SRC_LEN),
            vocab.encode(t, _MAX_TGT_LEN - 1),
        )
        for s, t in pairs
    ]
    encoded = [(s, t) for s, t in encoded if s and t]
    loader = DataLoader(
        _PairDataset(encoded),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    model = Seq2SeqTransformer(len(vocab)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for src, tin, tout in loader:
            src, tin, tout = src.to(dev), tin.to(dev), tout.to(dev)
            optimizer.zero_grad()
            logits = model(src, tin)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tout.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        final_loss = total / max(1, len(loader))
        log(f"gen epoch {epoch + 1}/{epochs} loss={final_loss:.4f}")

    model.eval()
    return model, GenTrainResult(
        epochs=epochs,
        final_loss=final_loss,
        num_pairs=len(encoded),
        vocab_size=len(vocab),
    )
