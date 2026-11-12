"""Hand-built transformers whose mechanism is known exactly, so every primitive has a right answer.

copy_model: 1 layer, 2 heads, no MLP, no norm. Head 0.0 attends to position 0 from everywhere and copies
that token to the output. Head 0.1 has zero weights. The logits at every position are 5 * onehot(token at
position 0). Residual layout: dims 0-7 token identity, 8-11 position, 12-19 output.

fourier_model: an embedding made of exactly two frequencies (3 and 5) over 16 tokens.
"""
import math

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from glassbox.backends.local_full import LocalFull

V, OUT0, GAIN = 8, 12, 5.0


def _blank(**kw) -> HookedTransformer:
    cfg = HookedTransformerConfig(normalization_type=None, seed=0, **kw)
    model = HookedTransformer(cfg)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    return model


def copy_model() -> HookedTransformer:
    m = _blank(n_layers=1, d_model=20, d_head=8, n_heads=2, n_ctx=4, d_vocab=V, attn_only=True)
    with torch.no_grad():
        for t in range(V):
            m.embed.W_E[t, t] = 1.0
            m.blocks[0].attn.W_V[0, t, t] = 1.0
            m.blocks[0].attn.W_O[0, t, OUT0 + t] = 1.0
            m.unembed.W_U[OUT0 + t, t] = GAIN
        for i in range(4):
            m.pos_embed.W_pos[i, 8 + i] = 1.0
        m.blocks[0].attn.W_K[0, 8, 0] = 1.0      # key at position 0 lights up head-dim 0
        m.blocks[0].attn.b_Q[0, 0] = 100.0       # every query asks for it
    return m


def fourier_model(p: int = 16) -> HookedTransformer:
    m = _blank(n_layers=1, d_model=4, d_head=2, n_heads=1, n_ctx=2, d_vocab=p + 1, attn_only=True)
    idx = torch.arange(p, dtype=torch.float32)
    with torch.no_grad():
        for j, (f, fn) in enumerate([(3, torch.cos), (3, torch.sin), (5, torch.cos), (5, torch.sin)]):
            m.embed.W_E[:p, j] = fn(2 * math.pi * f * idx / p)
    return m


def backend(model: HookedTransformer) -> LocalFull:
    b = LocalFull(device="cpu")
    b.load(model=model, model_id="synthetic")
    return b


def copy_data(n: int = 12, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    clean, corrupt, good, bad = [], [], [], []
    for _ in range(n):
        c, w = torch.randperm(V, generator=g)[:2].tolist()
        rest = torch.randint(0, V, (3,), generator=g).tolist()
        clean.append([c] + rest)
        corrupt.append([w] + rest)
        good.append(c)
        bad.append(w)
    return {"clean": {"tokens": clean}, "corrupt": {"tokens": corrupt},
            "metric": {"type": "logit_diff", "correct": good, "incorrect": bad}}
