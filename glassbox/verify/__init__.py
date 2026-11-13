"""Verification: does the recovered algorithm actually describe the model?

Two independent checks, both on inputs the investigation never used:

1. Agreement. The hypothesis is written as an executable ``predict`` function. It is run on
   held-out inputs and compared with what the real model outputs.
2. Causal. The claimed circuit is tested both ways: ablate the circuit and the behaviour
   should break; ablate everything else and the behaviour should survive.

Null calibration. Passing a fixed threshold is not enough: every effect is also compared with a
null distribution of the same measurement where no finding exists.

* necessity and sufficiency are compared with random component sets (or random subspaces) of the
  same size in the same model, and, on backends with weight access, with the same circuit in
  null models (fresh random weights, and weights shuffled within each matrix);
* agreement is compared with the agreement obtained when the pairing between inputs and model
  outputs is shuffled.

A finding passes only if its effect exceeds the 95th percentile of its null effects. Each finding
reports the effect, the null mean, the null 95th percentile, the percentile it reaches and the
number of null samples.

A finding is only ``verified`` when it passes on a full-precision backend. On a quantized
backend the best possible status is ``screening-pass``.
"""
from __future__ import annotations

import math
import random
import tempfile
from typing import Any

import numpy as np
import torch

from .. import tasks
from ..backends.base import Edit
from ..lab import plots
from ..lab.core import Lab, component_site, parse_component

DEFAULT_THRESHOLDS = {
    "agreement_min": 0.90,               # predict() matches the model on at least this share of held-out inputs
    "circuit_alone_min": 0.70,           # with everything else ablated, at least this share of the metric survives
    "without_circuit_max": 0.50,         # with the circuit ablated, at most this share of the metric survives
    "null_percentile": 95.0,             # every effect must exceed this percentile of its null distribution
}


def compile_predict(code: str, backend) -> Any:
    """Turn hypothesis source code into a callable. The code must define ``predict(tokens, prompt)``.

    This executes code written by your own agent on your own machine; it is not a sandbox.
    """
    ns: dict[str, Any] = {"math": math, "np": np, "torch": torch,
                          "tok": backend.token_id, "detok": lambda i: backend.decode([int(i)])[0]}
    exec(compile(code, "<hypothesis>", "exec"), ns)
    if "predict" not in ns:
        raise ValueError("hypothesis code must define predict(tokens, prompt)")
    return ns["predict"]


def metric_floor(metric_type: str, d_vocab: int) -> float:
    """The value a metric takes when the model has no information about the answer.

    Ratios are measured from this floor, so that "fraction of clean" means the same thing for a logit
    difference (floor 0) and for a log-probability (floor log(1/vocab), where a plain ratio of two
    negative numbers would be misleading).
    """
    return {"logit_diff": 0.0, "logit": 0.0, "accuracy": 0.0, "prob": 1.0 / d_vocab,
            "logprob": -math.log(d_vocab)}[metric_type]


def _frac(a: float, b: float, floor: float = 0.0) -> float:
    return (a - floor) / (b - floor) if abs(b - floor) > 1e-9 else float("nan")


def null_test(name: str, effect: float, null: list[float], kinds: dict[str, int], percentile: float, what: str) -> dict:
    """Compare one effect with its null samples. Pass = strictly above the given percentile of the nulls."""
    vals = [x for x in null if x == x]
    if not vals:
        return {"finding": name, "what": what, "effect": effect, "n_null": 0, "pass": False,
                "why": "no null samples could be drawn (for example the claimed circuit is the whole model)"}
    p = float(np.percentile(vals, percentile))
    return {"finding": name, "what": what, "effect": effect, "null_mean": float(np.mean(vals)),
            f"null_p{percentile:g}": p, "null_max": float(np.max(vals)),
            "percentile_of_effect": float(100.0 * np.mean([x < effect for x in vals])),
            "n_null": len(vals), "null_samples_by_kind": kinds, "pass": bool(effect > p)}


def _null_backends(lab: Lab, n_seeds: int) -> list[tuple[str, Any]]:
    """Null copies of the model (fresh random weights; weights shuffled within each matrix), where the backend can make them."""
    make = getattr(lab.backend, "null_copy", None)
    if make is None:
        return []
    out = []
    for kind in ("random-init", "weight-shuffled"):
        for seed in range(n_seeds):
            out.append((kind, make(kind, seed)))
    return out


def verify(lab: Lab, spec: dict) -> tuple[dict, Any]:
    backend = lab.backend
    th = {**DEFAULT_THRESHOLDS, **spec.get("thresholds", {})}
    pct = float(th["null_percentile"])
    n_null = int(spec.get("null_samples", 30 if getattr(backend, "remote", False) is False else 8))
    ds = tasks.make(backend, spec["dataset"]) if "dataset" in spec else spec
    d = lab.data({k: v for k, v in spec.items() if k in ("dataset", "clean", "corrupt", "metric")}, need_corrupt=False)
    out: dict[str, Any] = {"name": spec.get("name", "hypothesis"), "n_heldout": d.n, "thresholds": th,
                           "provenance": backend.provenance(), "checks": {}, "null_controls": []}
    base = lab.fwd(d.clean, metric=d.metric, keep_logits=True)
    model_pred = base.logits.argmax(-1)
    m_clean = float(base.metric.mean())
    correct = d.metric.correct
    out["model_accuracy"] = float((model_pred == correct).float().mean())
    out["metric_clean"] = m_clean
    floor = metric_floor(d.metric.type, backend.info()["d_vocab"])
    out["metric_floor"] = floor
    passed: list[bool] = []
    rng = random.Random(0)

    # ---- 1. agreement between the written-out algorithm and the model -------------------------
    if spec.get("code"):
        predict = compile_predict(spec["code"], backend)
        prompts = ds["clean"].get("prompts") if isinstance(ds.get("clean"), dict) else None
        rows = d.clean.tolist()
        preds = []
        for i, row in enumerate(rows):
            p = predict(row, prompts[i] if prompts else None)
            preds.append(backend.token_id(p) if isinstance(p, str) else int(p))
        preds_t = torch.tensor(preds)
        agree = float((preds_t == model_pred).float().mean())
        out["checks"]["agreement"] = {
            "agreement_with_model": agree,
            "algorithm_accuracy_vs_ground_truth": float((preds_t == correct).float().mean()),
            "disagreements": [{"input": rows[i] if not prompts else prompts[i],
                               "algorithm": backend.decode([preds[i]])[0], "model": backend.decode([int(model_pred[i])])[0]}
                              for i in (preds_t != model_pred).nonzero()[:5, 0].tolist()],
            "pass": agree >= th["agreement_min"],
        }
        passed.append(agree >= th["agreement_min"])
        # null: the same predictions against the model's outputs for *other* inputs (pairing shuffled)
        gen = torch.Generator().manual_seed(0)
        shuffled = [float((preds_t == model_pred[torch.randperm(d.n, generator=gen)]).float().mean()) for _ in range(max(n_null, 20))]
        out["null_controls"].append(null_test("agreement", agree, shuffled, {"task-shuffled": len(shuffled)}, pct,
                                              "share of held-out inputs where predict() equals the model's output"))

    # ---- 2. causal check -----------------------------------------------------------------------
    circuit = spec.get("circuit") or {}

    def score(edits: list[Edit], on: Lab = lab, tokens=None, metric=None) -> dict:
        o = on.fwd(d.clean if tokens is None else tokens, edits=edits, metric=metric or d.metric, keep_logits=True)
        return {"metric": float(o.metric.mean()), "fraction_of_clean": _frac(float(o.metric.mean()), m_clean, floor),
                "accuracy": float((o.logits.argmax(-1) == correct).float().mean())}

    if circuit.get("components"):
        mode = circuit.get("mode", "mean")
        comps = [parse_component(c) for c in circuit["components"]]
        universe = lab.all_components()
        rest = [c for c in universe if c not in comps]
        use_corrupt = d.corrupt is not None and circuit.get("reference", "corrupt") == "corrupt"
        ref_tokens = d.corrupt if use_corrupt else d.clean
        sites = sorted({component_site(k, l) for k, l, _ in universe})
        ref = lab.fwd(ref_tokens, read=sites).acts if mode != "zero" else None
        without = score(lab.ablation_edits(comps, mode, ref))
        alone = score(lab.ablation_edits(rest, mode, ref))
        ok_alone = alone["fraction_of_clean"] >= th["circuit_alone_min"]
        ok_without = without["fraction_of_clean"] <= th["without_circuit_max"]

        # null: random component sets of the same size, in the same model
        k = len(comps)
        null_drop, null_keep = [], []
        if 0 < k < len(universe):
            for _ in range(n_null):
                rand = rng.sample(universe, k)
                if set(rand) == set(comps):
                    continue
                rand_rest = [c for c in universe if c not in rand]
                null_drop.append(m_clean - score(lab.ablation_edits(rand, mode, ref))["metric"])
                null_keep.append(score(lab.ablation_edits(rand_rest, mode, ref))["metric"] - floor)
        kinds_drop = {"random-same-size-set": len(null_drop)}
        # null: the same circuit in null models (no learned structure), where the backend can build them
        for kind, nb in _null_backends(lab, int(spec.get("null_model_seeds", 2))):
            nlab = Lab(nb, tempfile.mkdtemp(prefix="glassbox-null-"))
            nref = nlab.fwd(ref_tokens, read=sites).acts if mode != "zero" else None
            m0 = float(nlab.fwd(d.clean, metric=d.metric).metric.mean())
            m1 = float(nlab.fwd(d.clean, edits=nlab.ablation_edits(comps, mode, nref), metric=d.metric).metric.mean())
            null_drop.append(m0 - m1)
            kinds_drop[kind] = kinds_drop.get(kind, 0) + 1
            nb.unload()
        out["null_controls"] += [
            null_test("necessity", m_clean - without["metric"], null_drop, kinds_drop, pct,
                      "drop in the metric when the claimed circuit is ablated"),
            null_test("sufficiency", alone["metric"] - floor, null_keep, {"random-same-size-set": len(null_keep)}, pct,
                      "metric (above its no-information floor) with everything except the claimed circuit ablated"),
        ]
        out["checks"]["causal_components"] = {
            "circuit": circuit["components"], "circuit_size": len(comps), "total_components": len(universe),
            "ablation": mode, "reference": "corrupt" if use_corrupt else "clean",
            "circuit_ablated": without, "everything_else_ablated": alone,
            "random_same_size_ablated_fraction": [_frac(m_clean - x, m_clean, floor) for x in null_drop[:3]],
            "pass_breaks_without_circuit": ok_without, "pass_survives_on_circuit_alone": ok_alone,
            "pass": ok_alone and ok_without,
        }
        passed.append(ok_alone and ok_without)

    if circuit.get("subspaces"):
        specs = circuit["subspaces"]
        remove = [lab.subspace_edit({**s, "op": "project_out"}) for s in specs]
        keep = [lab.subspace_edit({**s, "op": "keep_only"}) for s in specs]
        without, alone = score(remove), score(keep)
        gen = torch.Generator().manual_seed(0)

        def random_like(edits: list[Edit], op: str) -> list[Edit]:
            res = []
            for e in edits:
                kk, dim = e.value["basis"].shape
                q = torch.linalg.qr(torch.randn(dim, kk, generator=gen)).Q.T
                res.append(Edit(e.site, op, {"basis": q, "mean": e.value["mean"]}, e.pos))
            return res

        null_drop = [m_clean - score(random_like(remove, "project_out"))["metric"] for _ in range(n_null)]
        null_keep = [score(random_like(remove, "keep_only"))["metric"] - floor for _ in range(n_null)]
        control = score(random_like(remove, "project_out"))
        ok_alone = alone["fraction_of_clean"] >= th["circuit_alone_min"]
        ok_without = without["fraction_of_clean"] <= th["without_circuit_max"]
        out["null_controls"] += [
            null_test("necessity (subspace)", m_clean - without["metric"], null_drop, {"random-same-dim-subspace": n_null}, pct,
                      "drop in the metric when the claimed subspace is removed"),
            null_test("sufficiency (subspace)", alone["metric"] - floor, null_keep, {"random-same-dim-subspace": n_null}, pct,
                      "metric (above its no-information floor) with only the claimed subspace kept"),
        ]
        out["checks"]["causal_subspace"] = {
            "subspaces": [{k: v for k, v in s.items()} for s in specs],
            "dims": [int(e.value["basis"].shape[0]) for e in remove],
            "of_total_dims": [int(e.value["basis"].shape[1]) for e in remove],
            "subspace_removed": without, "only_subspace_kept": alone, "random_subspace_removed": control,
            "pass_breaks_without_circuit": ok_without, "pass_survives_on_circuit_alone": ok_alone,
            "pass": ok_alone and ok_without,
        }
        passed.append(ok_alone and ok_without)

    # ---- verdict -----------------------------------------------------------------------------
    full = bool(backend.spec["verifies"])
    nulls_ok = all(n["pass"] for n in out["null_controls"])
    out["beats_null_controls"] = bool(out["null_controls"]) and nulls_ok
    if not passed:
        status = "nothing-to-check"
    elif all(passed) and nulls_ok:
        status = "verified" if full else "screening-pass"
    else:
        status = "not-verified"
    out["status"] = status
    out["all_checks_passed"] = bool(passed) and all(passed)
    failed_nulls = [n["finding"] for n in out["null_controls"] if not n["pass"]]
    out["note"] = (
        "Passed at full precision, and every effect exceeds its null distribution." if status == "verified" else
        f"Passed, but on a quantized backend ({backend.tag}). This is a screening result; confirm on local-full or ndif."
        if status == "screening-pass" else
        f"The thresholds were met, but these effects do not exceed the {pct:g}th percentile of their null distribution: "
        f"{', '.join(failed_nulls)}. A random control does about as well, so this is not a finding."
        if status == "not-verified" and passed and all(passed) else
        "At least one check failed; the explanation is partial or wrong. See the numbers." if status == "not-verified" else
        "Give 'code' and/or 'circuit' to check."
    )

    # ---- plot ----------------------------------------------------------------------------------
    labels, vals, hi = ["model\naccuracy"], [out["model_accuracy"]], []
    if "agreement" in out["checks"]:
        labels.append("algorithm\n= model")
        vals.append(out["checks"]["agreement"]["agreement_with_model"])
        hi.append(len(vals) - 1)
    for key, a, b, c in (("causal_components", "everything_else_ablated", "circuit_ablated", None),
                         ("causal_subspace", "only_subspace_kept", "subspace_removed", "random_subspace_removed")):
        if key in out["checks"]:
            ch = out["checks"][key]
            labels += ["circuit\nalone", "circuit\nablated"]
            vals += [ch[a]["fraction_of_clean"], ch[b]["fraction_of_clean"]]
            hi.append(len(vals) - 2)
            ctrl = ch[c]["fraction_of_clean"] if c else (float(np.mean(ch["random_same_size_ablated_fraction"]))
                                                         if ch["random_same_size_ablated_fraction"] else None)
            if ctrl is not None:
                labels.append("random\ncontrol")
                vals.append(ctrl)
    fig = plots.bars(lab, labels, vals, f"Verification: {out['name']}  [{status}]", "", "share (1.0 = unablated model)",
                     highlight=hi)
    return out, fig
