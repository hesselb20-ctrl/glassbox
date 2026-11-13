"""Weight-level diff of two GGUF files, one tensor at a time.

For a pair of matching matrices the difference ``delta = A - B`` is summarised without ever holding more than
that one pair in memory:

* ``relative_norm``   ||delta||_F / ||B||_F: how big the change is;
* ``top1_energy``     share of ||delta||_F^2 carried by the top singular component: 1.0 means exactly rank 1;
* ``effective_rank``  participation ratio of the squared singular values;
* ``direction``       the top singular vector on the residual-stream side of the matrix.

The spectrum is taken from the Gram matrix on the residual side (d_model x d_model), accumulated in chunks, so a
17408 x 5120 matrix and a 248320 x 5120 embedding are handled the same way and exactly, not by a randomised SVD.

Quantization is noise here: two different quantizations of the *same* weights also differ. ``noise floor`` runs
the same statistics on such a pair, and an edit only counts if it stands clear of that floor.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch


class GGUFWeights:
    """Lazy access to the tensors of one GGUF file, dequantized one at a time."""

    def __init__(self, path: str | Path) -> None:
        from gguf import GGUFReader

        self.path = Path(path)
        self.reader = GGUFReader(str(path))
        self.tensors = {t.name: t for t in self.reader.tensors}

    def names(self) -> list[str]:
        return list(self.tensors)

    def info(self, name: str) -> dict:
        t = self.tensors[name]
        return {"shape": [int(x) for x in t.shape][::-1], "quant": t.tensor_type.name}

    def load(self, name: str) -> np.ndarray:
        """Dequantized float32 array, shape [out, in] (GGUF stores the dimensions in the opposite order)."""
        from gguf.quants import dequantize

        t = self.tensors[name]
        shape = [int(x) for x in t.shape][::-1]
        return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape)


def residual_side(shape: list[int], d_model: int) -> int | None:
    """Which axis of a matrix lives in the residual stream (0 = it writes there, 1 = it reads from there)."""
    if len(shape) != 2:
        return None
    if shape[0] == d_model and shape[1] != d_model:
        return 0
    if shape[1] == d_model and shape[0] != d_model:
        return 1
    return None


def classify(name: str) -> str:
    """A coarse role for a tensor name, across the naming schemes llama.cpp uses."""
    n = re.sub(r"^blk\.\d+\.", "", name)
    if n in ("ffn_down.weight",):
        return "mlp_out"
    if n in ("attn_output.weight", "ssm_out.weight"):
        return "attn_out"
    if n in ("ffn_up.weight", "ffn_gate.weight"):
        return "mlp_in"
    if n in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"):
        return "attn_in"
    if n == "token_embd.weight":
        return "embedding"
    if n == "output.weight":
        return "unembedding"
    return "other"


def layer_of(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def delta_stats(a: np.ndarray, b: np.ndarray, d_model: int, device: str = "cpu", chunk: int = 4096) -> dict:
    """Statistics of ``a - b`` and its top direction on the residual side. ``a`` and ``b`` are [out, in]."""
    side = residual_side(list(a.shape), d_model)
    if side is None:
        raise ValueError(f"matrix of shape {a.shape} has no unique residual-stream side for d_model={d_model}")
    # arrange so that columns are the residual dimensions: rows x d_model
    A = a.T if side == 0 else a
    B = b.T if side == 0 else b
    G = torch.zeros(d_model, d_model, dtype=torch.float64, device=device)
    ref = 0.0
    for i in range(0, A.shape[0], chunk):
        d = torch.from_numpy(np.ascontiguousarray(A[i:i + chunk]) - np.ascontiguousarray(B[i:i + chunk])).to(device=device, dtype=torch.float64)
        G += d.T @ d
        ref += float((torch.from_numpy(np.ascontiguousarray(B[i:i + chunk])).to(device=device, dtype=torch.float64) ** 2).sum())
    evals, evecs = torch.linalg.eigh(G)
    evals = evals.clamp(min=0).flip(0)
    total = float(evals.sum())
    top = evecs[:, -1].float().cpu().numpy()
    return {
        "relative_norm": float(np.sqrt(total / ref)) if ref > 0 else None,
        "delta_norm": float(np.sqrt(total)),
        "top1_energy": float(evals[0] / total) if total > 0 else 0.0,
        "top4_energy": float(evals[:4].sum() / total) if total > 0 else 0.0,
        "effective_rank": float(total ** 2 / float((evals ** 2).sum())) if total > 0 else 0.0,
        "sigma1": float(evals[0].sqrt()),
        "sigma2": float(evals[1].sqrt()),
        "side": "writes to the residual stream" if side == 0 else "reads from the residual stream",
        "direction": top,                      # unit vector in the residual stream; sign is arbitrary
    }


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
