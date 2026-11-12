"""Verification: does the recovered algorithm actually describe the model?

Two independent checks, both on inputs the investigation never used:

1. Agreement. The hypothesis is written as an executable ``predict`` function. It is run on
   held-out inputs and compared with what the real model outputs.
2. Causal. The claimed circuit is tested both ways: ablate the circuit and the behaviour
   should break; ablate everything else and the behaviour should survive. A random set of the
   same size is ablated as a control.

A finding is only ``verified`` when it passes on a full-precision backend. On a quantized
backend the best possible status is ``screening``.
"""
from __future__ import annotations

import math
import random
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


def verify(lab: Lab, spec: dict) -> tuple[dict, Any]:
    backend = lab.backend
    th = {**DEFAULT_THRESHOLDS, **spec.get("thresholds", {})}
    ds = tasks.make(backend, spec["dataset"]) if "dataset" in spec else spec
    d = lab.data({k: v for k, v in spec.items() if k in ("dataset", "clean", "corrupt", "metric")}, need_corrupt=False)
    out: dict[str, Any] = {"name": spec.get("name", "hypothesis"), "n_heldout": d.n, "thresholds": th,
                           "provenance": backend.provenance(), "checks": {}}
    base = lab.fwd(d.clean, metric=d.metric, keep_logits=True)
    model_pred = base.logits.argmax(-1)
    m_clean = float(base.metric.mean())
    correct = d.metric.correct
    out["model_accuracy"] = float((model_pred == correct).float().mean())
    out["metric_clean"] = m_clean
    floor = metric_floor(d.metric.type, backend.info()["d_vocab"])
    out["metric_floor"] = floor
    passed: list[bool] = []

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

    # ---- 2. causal check -----------------------------------------------------------------------
    circuit = spec.get("circuit") or {}

    def score(edits: list[Edit]) -> dict:
        o = lab.fwd(d.clean, edits=edits, metric=d.metric, keep_logits=True)
        return {"metric": float(o.metric.mean()), "fraction_of_clean": _frac(float(o.metric.mean()), m_clean, floor),
                "accuracy": float((o.logits.argmax(-1) == correct).float().mean())}

    if circuit.get("components"):
        mode = circuit.get("mode", "mean")
        comps = [parse_component(c) for c in circuit["components"]]
        universe = lab.all_components()
        rest = [c for c in universe if c not in comps]
        ref_tokens = d.corrupt if (d.corrupt is not None and circuit.get("reference", "corrupt") == "corrupt") else d.clean
        sites = sorted({component_site(k, l) for k, l, _ in universe})
        ref = lab.fwd(ref_tokens, read=sites).acts if mode != "zero" else None
        without = score(lab.ablation_edits(comps, mode, ref))
        alone = score(lab.ablation_edits(rest, mode, ref))
        rng = random.Random(0)
        controls = [score(lab.ablation_edits(rng.sample(rest, min(len(comps), len(rest))), mode, ref))["fraction_of_clean"]
                    for _ in range(3)] if rest else []
        ok_alone = alone["fraction_of_clean"] >= th["circuit_alone_min"]
        ok_without = without["fraction_of_clean"] <= th["without_circuit_max"]
        out["checks"]["causal_components"] = {
            "circuit": circuit["components"], "circuit_size": len(comps), "total_components": len(universe),
            "ablation": mode, "reference": "corrupt" if ref_tokens is d.corrupt else "clean",
            "circuit_ablated": without, "everything_else_ablated": alone,
            "random_same_size_ablated_fraction": controls,
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
        rand = []
        for e in remove:
            k, dim = e.value["basis"].shape
            q = torch.linalg.qr(torch.randn(dim, k, generator=gen)).Q.T
            rand.append(Edit(e.site, "project_out", {"basis": q, "mean": e.value["mean"]}, e.pos))
        control = score(rand)
        ok_alone = alone["fraction_of_clean"] >= th["circuit_alone_min"]
        ok_without = without["fraction_of_clean"] <= th["without_circuit_max"]
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
    if not passed:
        status = "nothing-to-check"
    elif all(passed):
        status = "verified" if full else "screening-pass"
    else:
        status = "not-verified"
    out["status"] = status
    out["all_checks_passed"] = bool(passed) and all(passed)
    out["note"] = (
        "Passed at full precision." if status == "verified" else
        f"Passed, but on a quantized backend ({backend.tag}). This is a screening result; confirm on local-full or ndif."
        if status == "screening-pass" else
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
