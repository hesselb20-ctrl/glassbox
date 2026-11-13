"""Generate a glassbox-bench task set from a seed.

Compiled track: an algorithm is written in RASP, compiled to transformer weights with Tracr, converted to
a TransformerLens model, and then made hard to read: distractor heads and MLP neurons are added (they read
the real residual stream and write only into extra dimensions nothing else reads), heads and neurons are
permuted, and the whole residual stream is rotated by a random orthogonal matrix folded into the weights.
None of this changes the model's outputs, which is checked after every step.

Trained track: small transformers trained on invented functions. Their mechanism is not known, so they are
scored on behaviour prediction only.

Everything a task needs to be *investigated* goes to ``<out>/seed-N/tasks/<id>/`` (model.pt, task.json).
Everything needed to *grade* it goes to the sealed folder, outside the repo.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import truth as T

ROOT = Path(__file__).resolve().parent.parent.parent
CONS, VOW = "bdfgklmnprstvz", "aeiou"


def sealed_dir(seed: int | str) -> Path:
    """Where answer keys live: outside the repo and outside any investigation folder."""
    return Path(os.environ.get("GLASSBOX_SEALED", Path.home() / ".glassbox-sealed")) / str(seed)


def tracr_python() -> str:
    exe = os.environ.get("GLASSBOX_TRACR_PYTHON")
    if not exe:
        guess = Path.home() / ".venvs" / "glassbox-tracr" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        exe = str(guess) if guess.exists() else None
    if not exe:
        raise RuntimeError("Tracr runs in its own environment (it needs JAX). Create one, `pip install "
                           "git+https://github.com/google-deepmind/tracr.git`, and set GLASSBOX_TRACR_PYTHON to its python.")
    return exe


# --------------------------------------------------------------------------- #
# drawing a task from the seed
# --------------------------------------------------------------------------- #

def _symbols(rng: random.Random, k: int) -> list[str]:
    pool = [c + v for c in CONS for v in VOW]
    return rng.sample(pool, k)


def draw_compiled(rng: random.Random, family: str) -> dict:
    n = rng.randint(5, 6)
    vocab = _symbols(rng, rng.randint(5, 6))
    if family == "hist":
        k = rng.randint(2, 3)
        classes = {t: i % k for i, t in enumerate(rng.sample(vocab, len(vocab)))}
        p = {"classes": classes}
    elif family == "sort-by-key":
        p = {"key": dict(zip(vocab, rng.sample(range(1, len(vocab) + 1), len(vocab))))}
    elif family == "count-compare":
        a, b = rng.sample(vocab, 2)
        gt, eq, lt = rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 3)
        p = {"a": a, "b": b, "out": {"gt": gt, "eq": eq, "lt": lt}}
    elif family == "shift-copy":
        p = {"k": rng.randint(1, n - 2), "f": dict(zip(vocab, rng.sample(vocab, len(vocab))))}
    elif family == "lookup2":
        p = {"ptr": {t: rng.randrange(0, n - 1) for t in vocab},
             "table": {f"{x}|{y}": rng.choice(vocab) for x in vocab for y in vocab}}
    else:
        raise KeyError(family)
    return {"family": family, "params": p, "vocab": vocab, "seq_len": n}


def draw_trained(rng: random.Random, family: str) -> dict:
    vocab = _symbols(rng, 6)
    if family == "pair-table":
        return {"family": family, "params": {"table": {f"{x}|{y}": rng.choice(vocab) for x in vocab for y in vocab}},
                "vocab": vocab, "seq_len": 4}
    return {"family": family, "params": {"rank": dict(zip(vocab, rng.sample(range(len(vocab)), len(vocab))))},
            "vocab": vocab, "seq_len": 4}


# --------------------------------------------------------------------------- #
# Tracr weights -> TransformerLens model
# --------------------------------------------------------------------------- #

def compile_with_tracr(spec: dict) -> tuple[dict, dict]:
    script = Path(__file__).with_name("tracr_compile.py")
    with tempfile.TemporaryDirectory() as d:
        sp, out = Path(d) / "spec.json", Path(d) / "out.npz"
        sp.write_text(json.dumps(spec), encoding="utf-8")
        r = subprocess.run([tracr_python(), str(script), str(sp), str(out)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Tracr compilation failed: {(r.stderr or r.stdout).strip().splitlines()[-1]}")
        with np.load(out, allow_pickle=False) as z:          # close the file before the temp folder goes away
            meta = json.loads(str(z["__meta__"]))
            arrays = {k: np.array(z[k]) for k in z.files if k != "__meta__"}
    return arrays, meta


def to_state(arrays: dict, meta: dict, vocab: list[str]) -> tuple[dict, dict, list[str]]:
    """Tracr parameter arrays -> (TransformerLens config, state dict, unified vocabulary)."""
    H, L, K, M = meta["num_heads"], meta["num_layers"], meta["key_size"], meta["mlp_hidden_size"]
    d = len(meta["residual_labels"])
    inp = meta["input_map"]                                  # token string -> id, includes BOS and compiler_pad
    unified = [None] * len(inp)
    for tok, i in inp.items():
        unified[i] = "PAD" if tok == "compiler_pad" else tok
    for val, _ in meta["out_pairs"]:
        if val not in unified:
            unified.append(val)
    V = len(unified)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    sd: dict[str, torch.Tensor] = {}
    W_E = torch.zeros(V, d)
    W_E[: len(inp)] = t(arrays["token_embed__embeddings"])
    sd["embed.W_E"] = W_E
    sd["pos_embed.W_pos"] = t(arrays["pos_embed__embeddings"])
    W_U = torch.zeros(d, V)
    for val, res_idx in meta["out_pairs"]:
        W_U[res_idx, unified.index(val)] = 1.0
    sd["unembed.W_U"], sd["unembed.b_U"] = W_U, torch.zeros(V)
    for l in range(L):
        a = f"transformer__layer_{l}__attn__"
        for name, key in (("Q", "query"), ("K", "key"), ("V", "value")):
            sd[f"blocks.{l}.attn.W_{name}"] = t(arrays[a + key + "__w"]).reshape(d, H, K).permute(1, 0, 2).contiguous()
            sd[f"blocks.{l}.attn.b_{name}"] = t(arrays[a + key + "__b"]).reshape(H, K)
        sd[f"blocks.{l}.attn.W_O"] = t(arrays[a + "linear__w"]).reshape(H, K, d)
        sd[f"blocks.{l}.attn.b_O"] = t(arrays[a + "linear__b"])
        m = f"transformer__layer_{l}__mlp__"
        sd[f"blocks.{l}.mlp.W_in"], sd[f"blocks.{l}.mlp.b_in"] = t(arrays[m + "linear_1__w"]), t(arrays[m + "linear_1__b"])
        sd[f"blocks.{l}.mlp.W_out"], sd[f"blocks.{l}.mlp.b_out"] = t(arrays[m + "linear_2__w"]), t(arrays[m + "linear_2__b"])
    cfg = dict(n_layers=L, n_heads=H, d_model=d, d_head=K, d_mlp=M, act_fn="relu", normalization_type=None,
               d_vocab=V, d_vocab_out=V, n_ctx=sd["pos_embed.W_pos"].shape[0], attention_dir="bidirectional",
               positional_embedding_type="standard", seed=0)
    return cfg, sd, unified


def hooked_config(cfg: dict):
    """A checkpoint config dict -> HookedTransformerConfig. ``dtype`` is stored as a string ("float64")."""
    from transformer_lens import HookedTransformerConfig

    cfg = dict(cfg)
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = getattr(torch, cfg["dtype"])
    cfg.setdefault("device", "cpu")                   # bench models are tiny; the backend moves them where it wants
    return HookedTransformerConfig(**cfg)


def build_model(cfg: dict, sd: dict):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    model = HookedTransformer(hooked_config(cfg))
    own = model.state_dict()
    for k, v in sd.items():
        assert own[k].shape == v.shape, (k, tuple(own[k].shape), tuple(v.shape))
    missing = [k for k in own if k not in sd and not k.endswith(("mask", "IGNORE"))]
    assert not missing, missing
    model.load_state_dict({k: v.to(model.cfg.dtype) for k, v in sd.items()}, strict=False)
    return model.eval()


def true_components(sd: dict, L: int) -> list[str]:
    """Heads and MLP layers that carry the compiled program (non-zero output weights)."""
    out = []
    for l in range(L):
        W_O, W_V = sd[f"blocks.{l}.attn.W_O"], sd[f"blocks.{l}.attn.W_V"]
        out += [f"{l}.{h}" for h in range(W_O.shape[0]) if W_O[h].abs().max() > 0 and W_V[h].abs().max() > 0]
        if sd[f"blocks.{l}.mlp.W_out"].abs().max() > 0:
            out.append(f"mlp{l}")
    return out


# --------------------------------------------------------------------------- #
# obfuscation (behaviour-preserving)
# --------------------------------------------------------------------------- #

def obfuscate(cfg: dict, sd: dict, rng: random.Random) -> tuple[dict, dict, dict]:
    g = torch.Generator().manual_seed(rng.randrange(2**31))
    L, H, K, M, d = cfg["n_layers"], cfg["n_heads"], cfg["d_head"], cfg["d_mlp"], cfg["d_model"]
    real = set(true_components(sd, L))
    extra_d, extra_h, extra_m = rng.randint(6, 12), rng.randint(1, 2), rng.randint(4, 10)
    D, Hn, Mn = d + extra_d, H + extra_h, M + extra_m

    def scale(w: torch.Tensor) -> float:
        nz = w[w != 0]
        return float(nz.abs().mean()) if nz.numel() else 1.0

    def rand(*shape, s=1.0):
        return torch.randn(*shape, generator=g) * s

    new: dict[str, torch.Tensor] = {}
    pad = lambda w: torch.cat([w, torch.zeros(*w.shape[:-1], extra_d)], dim=-1)  # noqa: E731  (last dim = residual)
    new["embed.W_E"], new["pos_embed.W_pos"] = pad(sd["embed.W_E"]), pad(sd["pos_embed.W_pos"])
    new["unembed.W_U"] = torch.cat([sd["unembed.W_U"], torch.zeros(extra_d, cfg["d_vocab"])], dim=0)
    new["unembed.b_U"] = sd["unembed.b_U"].clone()
    key_heads, record = [], {"extra_residual_dims": extra_d, "distractor_heads_per_layer": extra_h,
                             "distractor_neurons_per_layer": extra_m, "head_permutation": {}, "layers": L}
    for l in range(L):
        p = f"blocks.{l}.attn."
        ws = {}
        for n in "QKV":
            w = torch.cat([sd[p + f"W_{n}"], torch.zeros(H, extra_d, K)], dim=1)             # real heads ignore the extra dims
            dist = rand(extra_h, D, K, s=0.5 * scale(sd[p + f"W_{n}"]) if n != "V" else 1.0)  # distractors read everything
            ws[n] = torch.cat([w, dist], dim=0)
            new[p + f"b_{n}"] = torch.cat([sd[p + f"b_{n}"], torch.zeros(extra_h, K)], dim=0)
        w_o = torch.cat([sd[p + "W_O"], torch.zeros(H, K, extra_d)], dim=2)
        dist_o = torch.zeros(extra_h, K, D)
        dist_o[:, :, d:] = rand(extra_h, K, extra_d, s=1.0)                                  # ... and write only where nothing reads
        w_o = torch.cat([w_o, dist_o], dim=0)
        perm = list(range(Hn))
        rng.shuffle(perm)                                                                     # new index j holds old head perm[j]
        for n in "QKV":
            new[p + f"W_{n}"] = ws[n][perm]
            new[p + f"b_{n}"] = new[p + f"b_{n}"][perm]
        new[p + "W_O"] = w_o[perm]
        new[p + "b_O"] = pad(sd[p + "b_O"])
        record["head_permutation"][str(l)] = perm
        key_heads += [f"{l}.{j}" for j, old in enumerate(perm) if f"{l}.{old}" in real]

        m = f"blocks.{l}.mlp."
        w_in = torch.cat([torch.cat([sd[m + "W_in"], torch.zeros(extra_d, M)], dim=0), rand(D, extra_m, s=1.0)], dim=1)
        b_in = torch.cat([sd[m + "b_in"], rand(extra_m, s=0.5)])
        dist_out = torch.zeros(extra_m, D)
        dist_out[:, d:] = rand(extra_m, extra_d, s=1.0)
        w_out = torch.cat([pad(sd[m + "W_out"]), dist_out], dim=0)
        nperm = torch.randperm(Mn, generator=g)
        new[m + "W_in"], new[m + "b_in"], new[m + "W_out"] = w_in[:, nperm], b_in[nperm], w_out[nperm]
        new[m + "b_out"] = pad(sd[m + "b_out"])

    # rotate the residual stream: x -> x R, folded into every matrix that reads or writes it
    R = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64)).Q
    f64 = {k: v.double() for k, v in new.items()}
    rot: dict[str, torch.Tensor] = {}
    for k, v in f64.items():
        if k in ("embed.W_E", "pos_embed.W_pos") or k.endswith(("attn.W_O", "attn.b_O", "mlp.W_out", "mlp.b_out")):
            rot[k] = v @ R                                             # writes
        elif k.endswith(("attn.W_Q", "attn.W_K", "attn.W_V")):
            rot[k] = torch.einsum("ij,hjk->hik", R.T, v)               # reads
        elif k.endswith("mlp.W_in") or k == "unembed.W_U":
            rot[k] = R.T @ v
        else:
            rot[k] = v
    # float64: after the rotation the large Tracr weights (attention sharpness, discretising MLPs) are spread over
    # every coordinate, and in float32 the rounding error is enough to change some outputs
    new_cfg = {**cfg, "d_model": D, "n_heads": Hn, "d_mlp": Mn, "dtype": "float64"}
    record["rotation"] = f"random orthogonal {D}x{D}, folded into the weights"
    key = sorted(key_heads, key=lambda s: tuple(map(int, s.split(".")))) + sorted(c for c in real if c.startswith("mlp"))
    return new_cfg, rot, {"transforms": record, "true_components": key}


# --------------------------------------------------------------------------- #
# checking a model against the ground truth
# --------------------------------------------------------------------------- #

def sample_inputs(vocab_ids: list[int], n: int, count: int, seed: int) -> list[list[int]]:
    space = len(vocab_ids) ** n
    if space <= count:
        return [[0, *c] for c in itertools.product(vocab_ids, repeat=n)]
    r = random.Random(seed)
    return [[0] + [r.choice(vocab_ids) for _ in range(n)] for _ in range(count)]


def accuracy(model, family: str, params: dict, unified: list[str], input_ids: list[int], n: int, count: int = 1500, seed: int = 1) -> float:
    xs = sample_inputs(input_ids, n, count, seed)
    with torch.no_grad():
        pred = model(torch.tensor(xs))[:, -1].argmax(-1).tolist()
    want = [unified.index(T.truth(family, params, [unified[i] for i in x[1:]])) for x in xs]
    return float(np.mean([a == b for a, b in zip(pred, want)]))


# --------------------------------------------------------------------------- #
# trained track
# --------------------------------------------------------------------------- #

def train_task(spec: dict, seed: int) -> tuple[dict, dict, list[str], dict]:
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    family, params, vocab, n = spec["family"], spec["params"], spec["vocab"], spec["seq_len"]
    unified = ["BOS", "PAD", *vocab]
    ids = list(range(2, 2 + len(vocab)))
    xs = [[0, *c] for c in itertools.product(ids, repeat=n)]
    ys = [unified.index(T.truth(family, params, [unified[i] for i in x[1:]])) for x in xs]
    order = list(range(len(xs)))
    random.Random(seed).shuffle(order)
    cut = int(0.7 * len(xs))
    X, Y = torch.tensor(xs), torch.tensor(ys)
    tr, te = torch.tensor(order[:cut]), torch.tensor(order[cut:])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cfg = dict(n_layers=2, n_heads=2, d_model=48, d_head=24, d_mlp=96, act_fn="relu", normalization_type=None,
               d_vocab=len(unified), d_vocab_out=len(unified), n_ctx=n + 1, seed=seed)
    model = HookedTransformer(hooked_config(cfg))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.5)
    steps = 0
    for steps in range(1, 6001):
        loss = torch.nn.functional.cross_entropy(model(X[tr])[:, -1], Y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if steps % 250 == 0:
            with torch.no_grad():
                acc = float((model(X[te])[:, -1].argmax(-1) == Y[te]).float().mean())
            if acc >= 0.995:
                break
    torch.use_deterministic_algorithms(False)
    with torch.no_grad():
        acc_te = float((model(X[te])[:, -1].argmax(-1) == Y[te]).float().mean())
        acc_all = float((model(X)[:, -1].argmax(-1) == Y).float().mean())
    info = {"train_steps": steps, "heldout_accuracy_on_target": acc_te, "accuracy_on_target_all_inputs": acc_all,
            "n_inputs": len(xs), "train_fraction": 0.7, "device": "cpu (deterministic)"}
    return cfg, {k: v.detach() for k, v in model.state_dict().items()}, unified, info


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #

def generate(seed: int, out_root: Path | str | None = None) -> dict:
    out_root = Path(out_root) if out_root else ROOT / "benchmarks" / "glassbox-bench"
    rng = random.Random(f"glassbox-bench-{seed}")
    jobs = [("compiled", f) for f in T.COMPILED] + [("trained", f) for f in T.TRAINED]
    rng.shuffle(jobs)                                           # task ids do not reveal the family
    pub_root, seal = out_root / f"seed-{seed}" / "tasks", sealed_dir(seed)
    seal.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (track, family) in enumerate(jobs, 1):
        tid = f"t{i}"
        if track == "compiled":
            spec = draw_compiled(rng, family)
            arrays, meta = compile_with_tracr(spec)
            cfg0, sd0, unified = to_state(arrays, meta, spec["vocab"])
            input_ids = [unified.index(t) for t in spec["vocab"]]
            n = spec["seq_len"]
            acc0 = accuracy(build_model(cfg0, sd0), family, spec["params"], unified, input_ids, n)
            if acc0 < 1.0:
                # Tracr and TransformerLens differ in whether attention scores are divided by sqrt(d_head)
                cfg0["use_attn_scale"] = False
                acc0 = accuracy(build_model(cfg0, sd0), family, spec["params"], unified, input_ids, n)
            assert acc0 == 1.0, f"{tid} ({family}): converted model is {acc0:.3f} accurate, expected 1.0"
            cfg, sd, key = obfuscate(cfg0, sd0, rng)
            model = build_model(cfg, sd)
            acc = accuracy(model, family, spec["params"], unified, input_ids, n, count=3000, seed=2)
            assert acc == 1.0, f"{tid} ({family}): obfuscation changed the behaviour ({acc:.4f})"
            extra_key = {"rasp_source": meta["rasp_source"], "tracr": {k: meta[k] for k in ("num_layers", "num_heads", "checked_inputs", "exhaustive")},
                         "components_before_obfuscation": true_components(sd0, cfg0["n_layers"]), **key,
                         "accuracy_after_conversion": acc0, "accuracy_after_obfuscation": acc}
            label = "compiled: mechanism known exactly"
            public_extra = {}
        else:
            spec = draw_trained(rng, family)
            cfg, sd, unified, info = train_task(spec, seed * 100 + i)
            model = build_model(cfg, sd)
            input_ids = [unified.index(t) for t in spec["vocab"]]
            extra_key = {"training": info, "true_components": None}
            label = "trained: behaviorally specified, mechanism unknown"
            public_extra = {"note": "trained by gradient descent on a hidden target function; it may be imperfect"}
        d = pub_root / tid
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg, "state_dict": sd, "vocab": unified}, d / "model.pt")
        public = {"id": tid, "seed": seed, "track": track, "seq_len": spec["seq_len"], "bos_id": 0, "vocab": unified,
                  "input_token_ids": input_ids, "n_layers": cfg["n_layers"], "n_heads": cfg["n_heads"], "d_model": cfg["d_model"],
                  "layout": "token ids: [BOS] + seq_len input tokens; the answer is read at the last position", **public_extra}
        (d / "task.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
        (seal / f"{tid}.json").write_text(json.dumps({"id": tid, "seed": seed, "track": track, "label": label, "family": family,
                                                      "params": spec["params"], "vocab": spec["vocab"], "seq_len": spec["seq_len"],
                                                      "truth_function": f"glassbox.bench.truth.{T.FAMILIES[family].__name__}",
                                                      **extra_key}, indent=1), encoding="utf-8")
        summary.append({"id": tid, "track": track, "n_layers": cfg["n_layers"], "n_heads": cfg["n_heads"], "d_model": cfg["d_model"]})
    (out_root / f"seed-{seed}" / "tasks.json").write_text(json.dumps({"seed": seed, "tasks": summary}, indent=1), encoding="utf-8")
    return {"seed": seed, "public": str(pub_root), "sealed": str(seal), "tasks": summary,
            "note": "the sealed folder holds the answer keys; do not open it until the runs are done"}
