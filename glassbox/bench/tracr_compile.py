"""Compile one bench task from a RASP program to transformer weights with Tracr.

This file runs in its own Python environment (Tracr needs JAX and a numpy that conflicts with the main
glassbox environment). It imports only tracr, numpy and the dependency-free ``truth.py`` next to it.

    python tracr_compile.py spec.json out.npz

spec.json: {"family", "params", "vocab": [...input token strings...], "seq_len": int}
out.npz:   the compiled weights, the residual-stream labels, the input and output token maps, and the
           result of checking the compiled model against the plain-Python ground truth.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
from tracr.compiler import compiling, lib
from tracr.rasp import rasp

_spec = importlib.util.spec_from_file_location("truth", Path(__file__).with_name("truth.py"))
truth_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(truth_mod)

EQ = rasp.Comparison.EQ


def build(family: str, p: dict, seq_len: int):
    """The RASP program of a task family. Returns (program, source text for the answer key)."""
    if family == "hist":
        cls = rasp.Map(lambda t: p["classes"][t], rasp.tokens).named("class")
        out = rasp.SelectorWidth(rasp.Select(cls, cls, EQ).named("same_class")).named("out")
        src = "cls = Map(classes, tokens); out = SelectorWidth(Select(cls, cls, EQ))"
    elif family == "sort-by-key":
        keys = rasp.Map(lambda t: p["key"][t], rasp.tokens).named("key")
        out = lib.make_sort(rasp.tokens, keys, max_seq_len=seq_len, min_key=1).named("out")
        src = "keys = Map(key, tokens); out = make_sort(tokens, keys)   # read at the last position"
    elif family == "count-compare":
        is_a = rasp.Map(lambda t: 1 if t == p["a"] else 0, rasp.tokens).named("is_a")
        is_b = rasp.Map(lambda t: 1 if t == p["b"] else 0, rasp.tokens).named("is_b")
        one = rasp.Map(lambda t: 1, rasp.tokens).named("one_")
        ca = rasp.SelectorWidth(rasp.Select(is_a, one, EQ).named("sel_a")).named("count_a")
        cb = rasp.SelectorWidth(rasp.Select(is_b, one, EQ).named("sel_b")).named("count_b")
        o = p["out"]
        out = rasp.SequenceMap(lambda x, y: o["gt"] if x > y else o["eq"] if x == y else o["lt"], ca, cb).named("out")
        src = "ca = SelectorWidth(Select(tokens==a, 1, EQ)); cb = likewise for b; out = SequenceMap(compare, ca, cb)"
    elif family == "shift-copy":
        prev = lib.shift_by(p["k"], rasp.tokens).named("shifted")
        out = rasp.Map(lambda t: p["f"].get(t), prev).named("out")
        src = "prev = shift_by(k, tokens); out = Map(f, prev)"
    elif family == "lookup2":
        ptr = rasp.Map(lambda t: p["ptr"][t], rasp.tokens).named("pointer")
        fetched = rasp.Aggregate(rasp.Select(rasp.indices, ptr, EQ).named("deref"), rasp.tokens, default=None).named("fetched")
        out = rasp.SequenceMap(lambda x, y: p["table"].get(f"{x}|{y}"), fetched, rasp.tokens).named("out")
        src = "ptr = Map(ptr, tokens); fetched = Aggregate(Select(indices, ptr, EQ), tokens); out = SequenceMap(table, fetched, tokens)"
    else:
        raise KeyError(family)
    return out, src


def main(spec_path: str, out_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    family, p, vocab, n = spec["family"], spec["params"], spec["vocab"], spec["seq_len"]
    program, src = build(family, p, n)
    model = compiling.compile_rasp_to_model(program, vocab=set(vocab), max_seq_len=n, compiler_bos="BOS")

    # check the compiled model against the plain-Python ground truth at the last position
    rng = random.Random(0)
    space = len(vocab) ** n
    inputs = (list(itertools.product(vocab, repeat=n)) if space <= 3000
              else [tuple(rng.choice(vocab) for _ in range(n)) for _ in range(1500)])
    wrong = 0
    out_values = set()
    for toks in inputs:
        got = model.apply(["BOS", *toks]).decoded[-1]
        want = truth_mod.truth(family, p, list(toks))
        out_values.add(str(got))
        wrong += str(got) != want
    if wrong:
        raise SystemExit(f"compiled model disagrees with the ground truth on {wrong} of {len(inputs)} inputs")

    labels = [str(l) for l in model.residual_labels]
    enc = model.output_encoder.encoding_map if model.output_encoder is not None else None
    if enc is None:
        raise SystemExit("the program has a numerical output; bench tasks must end in a categorical value")
    out_pairs = [[str(v), labels.index(f"{program.label}:{v}")] for v in enc]
    cfg = model.model_config
    arrays = {k.replace("/", "__") + "__" + kk: np.asarray(vv) for k, v in model.params.items() for kk, vv in v.items()}
    meta = {"num_heads": cfg.num_heads, "num_layers": cfg.num_layers, "key_size": cfg.key_size,
            "mlp_hidden_size": cfg.mlp_hidden_size, "residual_labels": labels,
            "input_map": {str(k): int(v) for k, v in model.input_encoder.encoding_map.items()},
            "out_pairs": out_pairs, "rasp_source": src, "checked_inputs": len(inputs), "exhaustive": space <= 3000,
            "output_values_seen": sorted(out_values)}
    np.savez(out_path, __meta__=np.array(json.dumps(meta)), **arrays)
    print(json.dumps({"ok": True, "layers": cfg.num_layers, "heads": cfg.num_heads, "d_model": len(labels),
                      "checked": len(inputs), "outputs": sorted(out_values)}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
