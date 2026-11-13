"""Weight-level model diff: a planted rank-1 edit must be found, with its direction, under noise."""
import numpy as np
import pytest

from glassbox.audit.weights import classify, cosine, delta_stats, layer_of, residual_side

D = 48


def _pair(rng, shape, side, strength=1.0, noise=0.02):
    """An original matrix, and a copy with one residual direction projected out plus independent noise on both."""
    w = rng.standard_normal(shape).astype(np.float32)
    r = rng.standard_normal(D).astype(np.float32)
    r /= np.linalg.norm(r)
    edited = w - strength * (np.outer(r, r @ w) if side == 0 else np.outer(w @ r, r))
    a = edited + noise * rng.standard_normal(shape).astype(np.float32)
    b = w + noise * rng.standard_normal(shape).astype(np.float32)
    return a, b, r


def test_residual_side_and_names():
    assert residual_side([D, 4 * D], D) == 0 and residual_side([4 * D, D], D) == 1
    assert residual_side([D, D], D) is None and residual_side([D], D) is None
    assert classify("blk.7.ffn_down.weight") == "mlp_out" and classify("blk.7.attn_output.weight") == "attn_out"
    assert classify("blk.7.ffn_up.weight") == "mlp_in" and classify("token_embd.weight") == "embedding"
    assert layer_of("blk.12.attn_q.weight") == 12 and layer_of("output.weight") is None


@pytest.mark.parametrize("shape,side", [((D, 4 * D), 0), ((4 * D, D), 1)])
def test_planted_rank_one_edit_is_found(shape, side):
    rng = np.random.default_rng(0)
    a, b, r = _pair(rng, shape, side)
    s = delta_stats(a, b, D, chunk=64)
    assert cosine(s["direction"], r) > 0.98
    assert s["sigma1"] / s["sigma2"] > 3 and s["top1_energy"] > 0.5
    assert ("writes" in s["side"]) == (side == 0)


def test_noise_only_has_no_structure():
    rng = np.random.default_rng(1)
    a, b, r = _pair(rng, (D, 4 * D), 0, strength=0.0)
    s = delta_stats(a, b, D)
    assert s["top1_energy"] < 0.1 and s["sigma1"] / s["sigma2"] < 1.5
    assert cosine(s["direction"], r) < 0.6


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(2)
    a, b, _ = _pair(rng, (4 * D, D), 1)
    s1, s2 = delta_stats(a, b, D, chunk=7), delta_stats(a, b, D, chunk=4096)
    assert s1["top1_energy"] == pytest.approx(s2["top1_energy"], rel=1e-6)
    assert s1["relative_norm"] == pytest.approx(s2["relative_norm"], rel=1e-6)
