"""The experiment primitives. Each takes ``(lab, params)`` and returns ``(summary, data, figure)``.

All of them are written against the backend interface only, so they run unchanged on every
backend that has the features they need (see ``glassbox.backends.base.PRIMITIVES``).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .. import tasks
from ..backends.base import Edit
from . import plots
from .core import Lab, component_site, parse_component, parse_head

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _top(matrix: torch.Tensor, rows: list, cols: list, k: int = 8, fmt="{r}.{c}") -> list[dict]:
    m = matrix.detach().float().cpu()
    flat = m.abs().flatten()
    out = []
    for i in flat.argsort(descending=True)[:k].tolist():
        r, c = divmod(i, m.shape[1])
        out.append({"component": fmt.format(r=rows[r], c=cols[c]), "effect": float(m[r, c])})
    return out


def _base_src(lab: Lab, params: dict):
    d = lab.data(params)
    direction = params.get("direction", "denoise")
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' (corrupt run, patch in clean) or 'noise' (clean run, patch in corrupt)")
    base, src = (d.corrupt, d.clean) if direction == "denoise" else (d.clean, d.corrupt)
    m_base = float(lab.fwd(base, metric=d.metric).metric.mean())
    m_src = float(lab.fwd(src, metric=d.metric).metric.mean())
    denom = m_src - m_base
    if abs(denom) < 1e-9:
        raise ValueError("clean and corrupt runs give the same metric; nothing to patch")
    return d, base, src, m_base, m_src, denom, direction


def _layers(lab: Lab, params: dict) -> list[int]:
    n = lab.backend.info()["n_layers"]
    return list(params.get("layers") or range(n))


def _pos_labels(lab: Lab, d, npos: int) -> list[str]:
    strs = d.meta.get("token_strs")
    if strs and len(strs) == npos:
        return [f"{i}:{s.strip() or s!r}" for i, s in enumerate(strs)]
    return [str(i) for i in range(npos)]


# --------------------------------------------------------------------------- #
# dataset / baseline
# --------------------------------------------------------------------------- #

def dataset(lab: Lab, params: dict):
    """Generate a behaviour dataset. params: the task spec, e.g. {"task": "ioi", "n": 64, "seed": 0}."""
    ds = tasks.make(lab.backend, params.get("dataset", params))
    clean = ds["clean"].get("prompts") or ds["clean"]["tokens"]
    corrupt = ds["corrupt"].get("prompts") or ds["corrupt"]["tokens"]
    summary = {
        "task": ds["task"], "description": ds["description"], "n": len(clean),
        "example_clean": clean[:3], "example_corrupt": corrupt[:3], "example_answers": ds["answers"][:3],
        "metric_type": ds["metric"]["type"], "meta": {k: v for k, v in ds["meta"].items() if k != "positions"},
        "positions_example": ds["meta"].get("positions", [None])[:2],
        "usage": "pass the same spec as params['dataset'] to any primitive",
    }
    return summary, ds, None


def baseline(lab: Lab, params: dict):
    """How well does the model do the behaviour? Metric and accuracy on clean (and corrupt) inputs."""
    d = lab.data(params, need_corrupt=False, allow_ragged=True)
    out = lab.fwd(d.clean, metric=d.metric, keep_logits=True)
    pred = out.logits.argmax(-1)
    acc = float((pred == d.metric.correct).float().mean())
    summary: dict[str, Any] = {
        "n": d.n, "metric_type": d.metric.type,
        "clean_metric_mean": float(out.metric.mean()), "clean_metric_std": float(out.metric.std()) if d.n > 1 else 0.0,
        "clean_accuracy": acc,
        "example_predictions": [
            {"predicted": lab.backend.decode([int(pred[i])])[0], "expected": lab.backend.decode([int(d.metric.correct[i])])[0]}
            for i in range(min(4, d.n))
        ],
    }
    data = {"clean_metric": out.metric}
    series = {"clean": out.metric.sort().values.tolist()}
    if d.corrupt is not None:
        c = lab.fwd(d.corrupt, metric=d.metric)
        summary["corrupt_metric_mean"] = float(c.metric.mean())
        data["corrupt_metric"] = c.metric
        series["corrupt"] = c.metric.sort().values.tolist()
    fig = plots.lines(lab, list(range(d.n)), series, f"Baseline: {d.metric.type} per example (sorted)",
                      "example (sorted)", d.metric.type, hline=0.0)
    return summary, data, fig


# --------------------------------------------------------------------------- #
# patching
# --------------------------------------------------------------------------- #

def activation_patching(lab: Lab, params: dict):
    """Exact causal effect of each component: run one input, overwrite one activation from the other run.

    params: dataset | clean+corrupt+metric; component in {resid_pre, resid_post, attn_out, mlp_out, head};
    layers; positions ("all", list, or omitted); direction "denoise" (default) or "noise".
    Effect is normalised: 0 = no change, 1 = fully restores the source run's metric.
    """
    d, base, src, m_base, m_src, denom, direction = _base_src(lab, params)
    comp = params.get("component", "resid_pre")
    info = lab.backend.info()
    layers = _layers(lab, params)
    npos = base.shape[1]

    if comp == "head":
        lab.backend.require("activation_patching", ["read_heads"])
        pos = params.get("positions")
        cache = lab.fwd(src, read=[f"head_out.{l}" for l in layers]).acts
        n_h = [cache[f"head_out.{l}"].shape[2] for l in layers]
        if len(set(n_h)) != 1:
            raise ValueError(f"layers {layers} have different numbers of heads {sorted(set(n_h))}; sweep one layer type at a time "
                             f"(see 'full_attention_layers' in the model info)")
        info = {**info, "n_heads": n_h[0]}
        eff = torch.zeros(len(layers), info["n_heads"])
        for i, l in enumerate(layers):
            for h in range(info["n_heads"]):
                e = Edit(f"head_out.{l}", "replace", cache[f"head_out.{l}"], pos, h)
                eff[i, h] = (float(lab.fwd(base, edits=[e], metric=d.metric).metric.mean()) - m_base) / denom
        cols = list(range(info["n_heads"]))
        top = _top(eff, layers, cols)
        fig = plots.heatmap(lab, eff, f"Activation patching: heads ({direction})", "head", "layer", cols, layers,
                            cbar="normalised effect")
        xl = cols
    elif comp in ("resid_pre", "resid_post", "attn_out", "mlp_out"):
        positions = params.get("positions")
        if positions == "all" or (positions is None and comp.startswith("resid")):
            cols_pos: list[Any] = list(range(npos))
        elif positions is None:
            cols_pos = [None]
        else:
            cols_pos = list(positions)
        cache = lab.fwd(src, read=[f"{comp}.{l}" for l in layers]).acts
        eff = torch.zeros(len(layers), len(cols_pos))
        for i, l in enumerate(layers):
            for j, p in enumerate(cols_pos):
                e = Edit(f"{comp}.{l}", "replace", cache[f"{comp}.{l}"], p)
                eff[i, j] = (float(lab.fwd(base, edits=[e], metric=d.metric).metric.mean()) - m_base) / denom
        labels = _pos_labels(lab, d, npos)
        xl = ["all" if p is None else labels[p] for p in cols_pos]
        top = _top(eff, layers, xl, fmt=comp + ".{r} @ pos {c}")
        fig = plots.heatmap(lab, eff, f"Activation patching: {comp} ({direction})", "position", "layer", xl, layers,
                            cbar="normalised effect")
    else:
        raise ValueError("component must be one of resid_pre, resid_post, attn_out, mlp_out, head")
    summary = {"component": comp, "direction": direction, "metric_base": m_base, "metric_source": m_src,
               "n": d.n, "top": top, "forward_passes": int(eff.numel()) + 3}
    return summary, {"effect": eff, "layers": layers, "columns": xl}, fig


def attribution_patching(lab: Lab, params: dict):
    """Cheap first screen: a first-order estimate of every patching effect from two forward passes and one backward.

    effect ~= (act_source - act_base) . d metric / d act   evaluated on the base run. Normalised like activation patching.
    """
    d, base, src, m_base, m_src, denom, direction = _base_src(lab, params)
    info = lab.backend.info()
    layers = _layers(lab, params)
    kinds = ["resid_pre", "attn_out", "head_out"] + ([] if info.get("attn_only") else ["mlp_out"])
    sites = [f"{k}.{l}" for k in kinds for l in layers]
    src_acts = lab.fwd(src, read=sites).acts
    out = lab.fwd(base, read=sites, metric=d.metric, grad_sites=sites)
    attr = {s: ((src_acts[s] - out.acts[s]) * out.grads[s]).float() for s in sites}
    resid = torch.stack([attr[f"resid_pre.{l}"].sum(-1).mean(0) for l in layers]).cpu() / denom        # [layer, pos]
    attn = torch.stack([attr[f"attn_out.{l}"].sum(-1).mean(0) for l in layers]).cpu() / denom          # [layer, pos]
    heads = torch.stack([attr[f"head_out.{l}"].sum((1, 3)).mean(0) for l in layers]).cpu() / denom     # [layer, head]
    data: dict[str, Any] = {"layers": layers, "resid_pre": resid, "attn_out_by_pos": attn, "heads": heads}
    cols = list(range(info["n_heads"]))
    summary: dict[str, Any] = {
        "direction": direction, "metric_base": m_base, "metric_source": m_src, "n": d.n,
        "top_heads": _top(heads, layers, cols),
        "top_resid": _top(resid, layers, list(range(resid.shape[1])), fmt="resid_pre.{r} @ pos {c}"),
        "attn_by_layer": attn.sum(1).tolist(),
        "note": "first-order estimate; confirm the top components with activation_patching",
        "forward_passes": 4, "backward_passes": 1,
    }
    if "mlp_out" in kinds:
        mlp = torch.stack([attr[f"mlp_out.{l}"].sum(-1).mean(0) for l in layers]).cpu() / denom
        data["mlp_out_by_pos"] = mlp
        summary["mlp_by_layer"] = mlp.sum(1).tolist()
    fig = plots.heatmap(lab, heads, f"Attribution patching: heads ({direction}, first-order)", "head", "layer",
                        cols, layers, cbar="estimated normalised effect")
    return summary, data, fig


def path_patching(lab: Lab, params: dict):
    """Effect of sender heads along one path only.

    receiver "logits": the sender's direct write to the output (every other head and MLP frozen).
    receiver {"heads": ["9.9"], "input": "q"|"k"|"v"}: sender -> those heads' query/key/value input only
    (other heads frozen, MLPs recomputed, as in Wang et al. 2022).
    """
    d, base, src, m_base, m_src, denom, direction = _base_src(lab, params)
    info = lab.backend.info()
    L, H = info["n_layers"], info["n_heads"]
    recv = params.get("receiver", "logits")
    senders = params.get("senders", "all")
    senders = [(l, h) for l in range(L) for h in range(H)] if senders == "all" else [parse_head(s) for s in senders]
    head_sites = [f"head_out.{l}" for l in range(L)]
    mlp_sites = [] if info.get("attn_only") else [f"mlp_out.{l}" for l in range(L)]
    base_cache = lab.fwd(base, read=head_sites + mlp_sites).acts
    src_cache = lab.fwd(src, read=head_sites).acts

    def frozen_heads(l: int, h: int) -> list[Edit]:
        edits = []
        for l2 in range(L):
            v = base_cache[f"head_out.{l2}"]
            if l2 == l:
                v = v.clone()
                v[:, :, h] = src_cache[f"head_out.{l2}"][:, :, h]
            edits.append(Edit(f"head_out.{l2}", "replace", v))
        return edits

    eff = torch.zeros(L, H)
    if recv == "logits":
        frozen_mlps = [Edit(s, "replace", base_cache[s]) for s in mlp_sites]
        for l, h in senders:
            m = lab.fwd(base, edits=frozen_heads(l, h) + frozen_mlps, metric=d.metric).metric.mean()
            eff[l, h] = (float(m) - m_base) / denom
        label = "logits (direct)"
    else:
        lab.backend.require("path_patching", ["head_inputs"])
        receivers = [parse_head(x) for x in recv["heads"]]
        inp = recv.get("input", "q")
        if inp not in ("q", "k", "v"):
            raise ValueError("receiver input must be 'q', 'k' or 'v'")
        r_sites = sorted({f"head_{inp}_in.{rl}" for rl, _ in receivers})
        for l, h in senders:
            if all(rl <= l for rl, _ in receivers):
                continue
            mid = lab.fwd(base, read=r_sites, edits=frozen_heads(l, h)).acts
            edits = [Edit(f"head_{inp}_in.{rl}", "replace", mid[f"head_{inp}_in.{rl}"], None, rh)
                     for rl, rh in receivers if rl > l]
            eff[l, h] = (float(lab.fwd(base, edits=edits, metric=d.metric).metric.mean()) - m_base) / denom
        label = f"{inp}-input of {', '.join(recv['heads'])}"
    cols = list(range(H))
    fig = plots.heatmap(lab, eff, f"Path patching: sender head -> {label}", "sender head", "sender layer",
                        cols, list(range(L)), cbar="normalised effect")
    summary = {"receiver": recv, "direction": direction, "metric_base": m_base, "metric_source": m_src, "n": d.n,
               "top": _top(eff, list(range(L)), cols)}
    return summary, {"effect": eff}, fig


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #

def ablation(lab: Lab, params: dict):
    """Knock components out and see what breaks.

    params: components ["9.6", "mlp0", "attn3"] or sweep "heads" | "mlps" | "attn"; mode "mean" | "zero" | "resample";
    joint (ablate the listed components together); reference "clean" | "corrupt" (where means / resamples come from);
    subspace {"artifact": id, "site": ..., "op": "project_out" | "keep_only", "pos": [...], "groups": [...]}.
    """
    d = lab.data(params, need_corrupt=False)
    mode = params.get("mode", "mean")
    info = lab.backend.info()
    m_clean = float(lab.fwd(d.clean, metric=d.metric).metric.mean())
    summary: dict[str, Any] = {"mode": mode, "n": d.n, "metric_clean": m_clean}

    if "subspace" in params:
        specs = params["subspace"] if isinstance(params["subspace"], list) else [params["subspace"]]
        rows = []
        for spec in specs:
            m = float(lab.fwd(d.clean, edits=[lab.subspace_edit(spec)], metric=d.metric).metric.mean())
            rows.append({"subspace": {k: v for k, v in spec.items()}, "metric": m, "fraction_of_clean": m / m_clean})
        summary["results"] = rows
        fig = plots.bars(lab, [f"{r['subspace'].get('op', 'project_out')}\n{r['subspace'].get('groups', 'all')}" for r in rows],
                         [r["metric"] for r in rows], "Subspace ablation", "edit", d.metric.type)
        return summary, {"results": rows}, fig

    sweep = params.get("sweep")
    if sweep:
        comps = {"heads": lab.all_components(mlps=False), "mlps": lab.all_components(heads=False),
                 "attn": [("attn", l, None) for l in range(info["n_layers"])]}[sweep]
    else:
        comps = [parse_component(c) for c in params["components"]]
    ref_name = params.get("reference", "clean")
    ref_tokens = d.clean if ref_name == "clean" else lab.data(params).corrupt
    sites = sorted({component_site(k, l) for k, l, _ in comps})
    ref = lab.fwd(ref_tokens, read=sites).acts if mode != "zero" else None
    pos = params.get("positions")
    names = [f"{l}.{h}" if k == "head" else f"{k}{l}" for k, l, h in comps]

    if params.get("joint"):
        m = float(lab.fwd(d.clean, edits=lab.ablation_edits(comps, mode, ref, pos), metric=d.metric).metric.mean())
        summary.update({"joint": True, "components": names, "metric_ablated": m, "fraction_of_clean": m / m_clean})
        fig = plots.bars(lab, ["clean", "ablated"], [m_clean, m], f"Joint {mode}-ablation of {len(comps)} components",
                         "", d.metric.type, highlight=[0])
        return summary, {"metric_ablated": m}, fig

    vals = torch.tensor([float(lab.fwd(d.clean, edits=lab.ablation_edits([c], mode, ref, pos), metric=d.metric).metric.mean())
                         for c in comps])
    change = vals - m_clean
    order = change.abs().argsort(descending=True)[:10].tolist()
    summary.update({"reference": ref_name,
                    "top": [{"component": names[i], "metric": float(vals[i]), "change": float(change[i]),
                             "fraction_of_clean": float(vals[i] / m_clean)} for i in order]})
    if sweep == "heads":
        mat = change.reshape(info["n_layers"], info["n_heads"])
        fig = plots.heatmap(lab, mat, f"{mode}-ablation of each head: change in {d.metric.type}", "head", "layer",
                            list(range(info["n_heads"])), list(range(info["n_layers"])), cbar="metric change")
    else:
        fig = plots.bars(lab, names, change.tolist(), f"{mode}-ablation: change in {d.metric.type}", "component", "metric change",
                         highlight=order[:3])
    return summary, {"components": names, "metric": vals, "change": change}, fig


# --------------------------------------------------------------------------- #
# reading the residual stream
# --------------------------------------------------------------------------- #

def logit_lens(lab: Lab, params: dict):
    """Decode the residual stream after every layer through the final norm and unembedding."""
    d = lab.data(params, need_corrupt=False, need_metric=False, allow_ragged=True)
    pos = params.get("pos", -1)
    layers = _layers(lab, params)
    sites = [f"resid_post.{l}" for l in layers]
    acts = lab.fwd(d.clean, read=sites, read_pos=pos).acts
    idx = torch.arange(d.n)
    k = int(params.get("top_k", 5))
    metric_by_layer, rank_by_layer, prob_by_layer, tops = [], [], [], []
    for s in sites:
        logits = lab.backend.unembed(acts[s]).float()
        tops.append([lab.backend.decode(logits[i].topk(k).indices.tolist()) for i in range(min(3, d.n))])
        if d.metric is not None:
            metric_by_layer.append(float(d.metric(logits, idx).mean()))
            c = d.metric.correct.to(logits.device)
            correct_logit = logits.gather(1, c[:, None])
            rank_by_layer.append(float((logits > correct_logit).sum(-1).float().median()))
            prob_by_layer.append(float(logits.softmax(-1).gather(1, c[:, None]).mean()))
    summary: dict[str, Any] = {"pos": pos, "n": d.n, "layers": layers,
                               "top_tokens_example0": {str(l): t[0] for l, t in zip(layers, tops)}}
    data: dict[str, Any] = {"layers": layers, "top_tokens": tops}
    if d.metric is not None:
        summary.update({"metric_type": d.metric.type, "metric_by_layer": metric_by_layer,
                        "median_rank_of_answer": rank_by_layer, "prob_of_answer": prob_by_layer})
        half = metric_by_layer[-1] / 2
        summary["first_layer_reaching_half_final_metric"] = next(
            (l for l, m in zip(layers, metric_by_layer) if (m >= half if half >= 0 else m <= half)), None)
        data.update({"metric": metric_by_layer, "rank": rank_by_layer, "prob": prob_by_layer})
        fig = plots.lines(lab, layers, {d.metric.type: metric_by_layer}, f"Logit lens at position {pos}",
                          "layer (residual stream after)", d.metric.type, hline=0.0)
    else:
        fig = plots.lines(lab, layers, {"max logit": [float(lab.backend.unembed(acts[s]).float().max(-1).values.mean()) for s in sites]},
                          f"Logit lens at position {pos}", "layer", "max logit")
    return summary, data, fig


def direct_logit_attribution(lab: Lab, params: dict):
    """How much each head writes directly along the answer direction (no downstream effects)."""
    d = lab.data(params, need_corrupt=False)
    if d.metric.type != "logit_diff":
        raise ValueError("direct_logit_attribution needs a logit_diff metric")
    info = lab.backend.info()
    L, H = info["n_layers"], info["n_heads"]
    pos = params.get("pos", -1)
    acts = lab.fwd(d.clean, read=[f"head_out.{l}" for l in range(L)] + [f"resid_post.{L - 1}"]).acts
    W_U = lab.backend.weight("W_U").float()
    direction = (W_U[:, d.metric.correct.to(W_U.device)] - W_U[:, d.metric.wrong.to(W_U.device)]).T   # [n, d]
    final = acts[f"resid_post.{L - 1}"][:, pos].float()
    scale = lab.backend.final_scale(final)                                                             # [n, 1]
    out = torch.zeros(L, H)
    for l in range(L):
        write = lab.backend.head_write(l, acts[f"head_out.{l}"][:, [pos]])[:, 0].float()    # [n, head, d]
        write = write - write.mean(-1, keepdim=True)
        out[l] = torch.einsum("nhd,nd->nh", write / scale[:, None], direction).mean(0).cpu()
    total = float(lab.fwd(d.clean, metric=d.metric).metric.mean())
    cols = list(range(H))
    fig = plots.heatmap(lab, out, "Direct logit attribution per head", "head", "layer", cols, list(range(L)),
                        cbar="logit difference contributed")
    summary = {"metric_clean": total, "sum_of_head_contributions": float(out.sum()), "n": d.n,
               "top": _top(out, list(range(L)), cols)}
    return summary, {"contribution": out}, fig


def linear_probe(lab: Lab, params: dict):
    """Train a logistic-regression probe per layer: where is a property linearly readable?

    params: data {"prompts": [...]} (or dataset), labels [...], pos (default -1), sites (default resid_post of every layer).
    Reports held-out accuracy and a shuffled-label control. Saves the probe directions as an artifact.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    spec = params.get("data") or params.get("clean")
    tokens = lab.tokens(spec, allow_ragged=True) if spec is not None else lab.data(params, False, False, True).clean
    labels = params["labels"]
    classes = sorted(set(labels), key=str)
    y = np.array([classes.index(v) for v in labels])
    if len(y) != len(tokens):
        raise ValueError(f"{len(y)} labels for {len(tokens)} inputs")
    pos = params.get("pos", -1)
    sites = params.get("sites") or [f"resid_post.{l}" for l in _layers(lab, params)]
    acts = lab.fwd(tokens, read=sites, read_pos=pos).acts
    seed = int(params.get("seed", 0))
    tr, te = train_test_split(np.arange(len(y)), test_size=float(params.get("test_frac", 0.3)), random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    acc, ctrl, dirs = [], [], {}
    for s in sites:
        X = acts[s].float().cpu().numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=float(params.get("C", 0.1))).fit(Xn[tr], y[tr])
        acc.append(float(clf.score(Xn[te], y[te])))
        yc = rng.permutation(y[tr])
        ctrl.append(float(LogisticRegression(max_iter=2000, C=float(params.get("C", 0.1))).fit(Xn[tr], yc).score(Xn[te], y[te])))
        if len(classes) == 2:
            dirs[s] = torch.tensor(clf.coef_[0] / sd, dtype=torch.float32)
    aid = lab.save_artifact("probe", {"directions": dirs, "classes": classes, "pos": pos}) if dirs else None
    best = int(np.argmax(acc))
    summary = {"classes": [str(c) for c in classes], "n_train": len(tr), "n_test": len(te), "pos": pos,
               "sites": sites, "test_accuracy": acc, "shuffled_label_control": ctrl,
               "best_site": sites[best], "best_accuracy": acc[best], "chance": float(np.bincount(y).max() / len(y)),
               "first_site_above_0.9": next((s for s, a in zip(sites, acc) if a >= 0.9), None),
               "direction_artifact": aid}
    fig = plots.lines(lab, list(range(len(sites))), {"probe (held-out)": acc, "shuffled labels": ctrl},
                      f"Linear probe accuracy by layer ({' vs '.join(map(str, classes))})", "layer", "accuracy",
                      hline=summary["chance"])
    return summary, {"sites": sites, "accuracy": acc, "control": ctrl}, fig


def diff_of_means(lab: Lab, params: dict):
    """Find a direction as the difference between mean activations of two prompt sets.

    params: positive {"prompts": [...]}, negative {"prompts": [...]}, pos (default -1), sites.
    The direction is fitted on half the prompts and scored on the other half. Saved as an artifact for steering.
    """
    pos = params.get("pos", -1)
    sites = params.get("sites") or [f"resid_post.{l}" for l in _layers(lab, params)]
    P = lab.fwd(lab.tokens(params["positive"], True), read=sites, read_pos=pos).acts
    N = lab.fwd(lab.tokens(params["negative"], True), read=sites, read_pos=pos).acts
    gen = torch.Generator().manual_seed(int(params.get("seed", 0)))
    n_p, n_n = len(P[sites[0]]), len(N[sites[0]])
    pp, pn = torch.randperm(n_p, generator=gen), torch.randperm(n_n, generator=gen)
    fp, hp, fn, hn = pp[: n_p // 2], pp[n_p // 2:], pn[: n_n // 2], pn[n_n // 2:]
    dirs, dprime, acc, norms, rel = {}, [], [], [], []
    for s in sites:
        a, b = P[s].float().cpu(), N[s].float().cpu()
        fit = a[fp].mean(0) - b[fn].mean(0)
        unit = fit / fit.norm()
        ya, yb = a[hp] @ unit, b[hn] @ unit
        pooled = math.sqrt((float(ya.var()) + float(yb.var())) / 2) + 1e-9
        dprime.append(float((ya.mean() - yb.mean()) / pooled))
        mid = float((a[fp] @ unit).mean() + (b[fn] @ unit).mean()) / 2
        acc.append(float(((ya > mid).float().sum() + (yb <= mid).float().sum()) / (len(ya) + len(yb))))
        dirs[s] = a.mean(0) - b.mean(0)
        norms.append(float(dirs[s].norm()))
        rel.append(norms[-1] / float(torch.cat([a, b]).norm(dim=-1).mean()))
    aid = lab.save_artifact("diffmeans", {"directions": dirs, "pos": pos})
    best = int(np.argmax(acc))
    summary = {"n_positive": n_p, "n_negative": n_n, "pos": pos, "sites": sites,
               "heldout_separation_dprime": dprime, "heldout_accuracy": acc, "direction_norm": norms,
               "direction_norm_relative_to_residual": rel,
               "best_site": sites[int(np.argmax(dprime))], "first_site_accuracy_above_0.9": next((s for s, a in zip(sites, acc) if a >= 0.9), None),
               "direction_artifact": aid, "usage": "steering: {'direction': {'artifact': '%s'}, 'sites': ['<site>']}" % aid}
    fig = plots.lines(lab, list(range(len(sites))), {"held-out accuracy": acc},
                      "Difference of means: held-out separation by layer", "layer", "accuracy", hline=0.5)
    return summary, {"sites": sites, "dprime": dprime, "accuracy": acc, "norm": norms}, fig


def steering(lab: Lab, params: dict):
    """Add a direction to the residual stream and measure the change in behaviour.

    params: direction {"artifact": id} or raw vector; sites ["resid_post.12", ...]; alphas [..];
    scale "relative" (alpha is a fraction of the typical residual norm at that site, default) or "absolute";
    data/metric to score (optional); generate {"prompts": [...], "max_new_tokens": 30, "alphas": [...]} (optional).
    """
    sites = params["sites"]
    alphas = [float(a) for a in params.get("alphas", [-8, -4, 0, 4, 8])]
    scale_mode = params.get("scale", "relative")
    vecs = {s: lab.direction(params["direction"], s) for s in sites}
    units = {s: v / v.norm() for s, v in vecs.items()}
    have_data = "dataset" in params or "clean" in params
    d = lab.data(params, need_corrupt=False, allow_ragged=True) if have_data else None
    gen = params.get("generate")
    probe_tokens = d.clean if d is not None else lab.tokens(gen["prompts"], True)
    scales = {s: 1.0 for s in sites}
    if scale_mode == "relative":
        norms = lab.fwd(probe_tokens, read=sites, read_pos=-1).acts
        scales = {s: float(norms[s].float().norm(dim=-1).mean()) for s in sites}

    def edits(alpha: float) -> list[Edit]:
        return [Edit(s, "add", units[s] * alpha * scales[s]) for s in sites] if alpha != 0 else []

    summary: dict[str, Any] = {"sites": sites, "alphas": alphas, "scale": scale_mode, "residual_norm_at_site": scales}
    data: dict[str, Any] = {}
    fig = None
    if d is not None and d.metric is not None:
        vals = [float(lab.fwd(d.clean, edits=edits(a), metric=d.metric).metric.mean()) for a in alphas]
        summary.update({"metric_type": d.metric.type, "metric_by_alpha": dict(zip(map(str, alphas), vals)), "n": d.n})
        if len(alphas) > 2:
            summary["slope_metric_per_alpha"] = float(np.polyfit(alphas, vals, 1)[0])
        data["metric"] = vals
        fig = plots.lines(lab, alphas, {d.metric.type: vals}, f"Steering at {', '.join(sites)}",
                          f"alpha ({scale_mode})", d.metric.type, hline=0.0)
    if gen:
        g_alphas = [float(a) for a in gen.get("alphas", [alphas[0], 0.0, alphas[-1]])]
        outs = []
        for p in gen["prompts"]:
            outs.append({"prompt": p, "outputs": {str(a): lab.backend.generate(p, int(gen.get("max_new_tokens", 30)), edits(a))
                                                  for a in g_alphas}})
        summary["generations"] = outs
        data["generations"] = outs
    return summary, data, fig


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #

def attention_patterns(lab: Lab, params: dict):
    """Where does each head look?

    params: data; query_pos (default -1); key_pos (int or one per example) -> score = attention from query to key;
    offset (e.g. -1) -> score = mean attention from every position to position+offset;
    head "L.H" -> also return that head's full mean pattern.
    """
    d = lab.data(params, need_corrupt=False, need_metric=False)
    info = lab.backend.info()
    layers = _layers(lab, params)
    H = info["n_heads"]
    q = params.get("query_pos", -1)
    key_pos, offset = params.get("key_pos"), params.get("offset")
    pats = lab.fwd(d.clean, read=[f"pattern.{l}" for l in layers]).acts
    npos = d.clean.shape[1]
    score = torch.zeros(len(layers), H)
    top_key = torch.zeros(len(layers), H, dtype=torch.long)
    for i, l in enumerate(layers):
        p = pats[f"pattern.{l}"].float()                       # [n, head, q, k]
        from_q = p[:, :, q]                                     # [n, head, k]
        top_key[i] = from_q.mean(0).argmax(-1).cpu()
        if key_pos is not None:
            kp = torch.tensor(key_pos if isinstance(key_pos, list) else [key_pos] * d.n, device=p.device)
            score[i] = from_q.gather(2, kp[:, None, None].expand(-1, H, 1))[:, :, 0].mean(0).cpu()
        elif offset is not None:
            qs = torch.arange(max(0, -offset), min(npos, npos - offset), device=p.device)
            score[i] = p[:, :, qs, qs + offset].mean((0, 2)).cpu()
        else:
            score[i] = from_q.mean(0).max(-1).values.cpu()
    what = (f"attention from pos {q} to given key positions" if key_pos is not None
            else f"mean attention to offset {offset}" if offset is not None
            else f"max mean attention from pos {q}")
    cols = list(range(H))
    summary = {"score": what, "n": d.n, "top": [{"head": t["component"], "score": t["effect"]} for t in _top(score, layers, cols)],
               "most_attended_key_from_query": {f"{l}.{h}": int(top_key[i, h]) for i, l in enumerate(layers) for h in range(H)}
               if len(layers) * H <= 48 else "omitted (many heads); see data"}
    data: dict[str, Any] = {"score": score, "top_key": top_key, "layers": layers}
    if params.get("head"):
        l, h = parse_head(params["head"])
        full = pats[f"pattern.{l}"][:, h].float().mean(0).cpu()
        data["pattern"] = full
        fig = plots.heatmap(lab, full, f"Mean attention pattern of head {l}.{h}", "key position", "query position",
                            _pos_labels(lab, d, npos), list(range(npos)), diverging=False, cbar="attention")
    else:
        fig = plots.heatmap(lab, score, f"Attention: {what}", "head", "layer", cols, layers, diverging=False, cbar="attention")
    return summary, data, fig


# --------------------------------------------------------------------------- #
# structure in weights and activations
# --------------------------------------------------------------------------- #

def periodicity(lab: Lab, params: dict):
    """Discrete Fourier analysis over an integer-valued input.

    source "W_E" / "W_U": rows 0..period-1 of the (un)embedding, transformed along the token index.
    source "<site>" (e.g. "mlp_act.0", "resid_post.0"): activations on the full period x period input grid
    (params.dataset must enumerate it in row-major order), transformed in 2D; reports how much of each unit's
    variance sits on a single frequency.
    Saves the subspace spanned by the dominant frequencies as an artifact (usable in ablation.subspace).
    """
    p = int(params["period"])
    source = params.get("source", "W_E")
    half = p // 2
    freqs = list(range(1, half + 1))
    if source in ("W_E", "W_U"):
        W = lab.backend.weight(source).float()
        M = (W[:p] if source == "W_E" else W[:, :p].T).cpu()                # [p, d]
        F = torch.fft.fft(M - M.mean(0), dim=0)                             # [p, d]
        power = (F.abs() ** 2).sum(1)[1:half + 1] * 2
        share = power / power.sum()
        # a frequency is "key" if it holds more than a uniform share and stands well clear of the typical one
        thresh = max(1.0 / len(freqs), float(params.get("threshold", 5.0)) * float(share.median()))
        key = [f for f, s in zip(freqs, share.tolist()) if s > thresh]
        idx = torch.arange(p, dtype=torch.float32)
        raw, groups = [], {}
        for f in key:
            c, s = torch.cos(2 * math.pi * f * idx / p), torch.sin(2 * math.pi * f * idx / p)
            groups[str(f)] = [len(raw), len(raw) + 1]
            raw += [c @ M, s @ M]
        aid = None
        if raw:
            raw_t = torch.stack(raw)
            basis = torch.linalg.qr(raw_t.T).Q.T
            site = "embed" if source == "W_E" else f"resid_post.{lab.backend.info()['n_layers'] - 1}"
            aid = lab.save_artifact("fourier_subspace", {"basis": basis, "raw": raw_t, "mean": M.mean(0), "groups": groups,
                                                         "site": site, "period": p, "source": source})
        order = share.argsort(descending=True)[:10].tolist()
        summary = {"source": source, "period": p, "key_frequencies": key,
                   "share_of_power_in_key_frequencies": float(sum(share[f - 1] for f in key)),
                   "top_frequencies": [{"frequency": freqs[i], "share": float(share[i])} for i in order],
                   "uniform_share_would_be": 1 / len(freqs), "subspace_artifact": aid,
                   "subspace_dim": 2 * len(key)}
        fig = plots.bars(lab, freqs, share.tolist(), f"Fourier power of {source} over the token index (period {p})",
                         "frequency k  (waves per period)", "share of power", highlight=[f - 1 for f in key])
        return summary, {"frequencies": freqs, "share": share}, fig

    d = lab.data(params, need_corrupt=False, need_metric=False)
    if d.n != p * p:
        raise ValueError(f"a site source needs the full {p}x{p} grid ({p * p} inputs); got {d.n}")
    A = lab.fwd(d.clean, read=[source], read_pos=params.get("pos", -1), batch_size=4096).acts[source].float()
    A = A.reshape(p, p, -1)
    F2 = torch.fft.fft2(A - A.mean((0, 1)), dim=(0, 1))
    P2 = (F2.abs() ** 2)                                                   # [p, p, units]
    total = P2.sum((0, 1)) + 1e-12
    frac = torch.zeros(half, A.shape[-1], device=A.device)
    for f in freqs:
        ks = torch.tensor([0, f, p - f], device=A.device)
        frac[f - 1] = P2[ks][:, ks].sum((0, 1)) / total
    best_frac, best = frac.max(0)
    best = (best + 1).cpu()
    best_frac = best_frac.cpu()
    var = total.cpu()
    alive = var > var.max() * 1e-4
    counts = torch.bincount(best[alive], minlength=half + 1)[1:]
    key = [f for f in freqs if counts[f - 1] >= max(2, 0.02 * int(alive.sum()))]
    summary = {"source": source, "period": p, "units": int(A.shape[-1]), "active_units": int(alive.sum()),
               "mean_fraction_of_variance_on_one_frequency": float(best_frac[alive].mean()),
               "units_above_0.85_single_frequency": int((best_frac[alive] > 0.85).sum()),
               "frequencies_with_units": {str(f): int(counts[f - 1]) for f in key},
               "key_frequencies": key}
    fig = plots.bars(lab, freqs, counts.tolist(), f"{source}: number of units tuned to each frequency (period {p})",
                     "frequency k", "units", highlight=[f - 1 for f in key])
    return summary, {"unit_best_frequency": best, "unit_fraction": best_frac, "counts": counts}, fig


def weights_svd(lab: Lab, params: dict):
    """Singular value spectrum of a weight matrix: how many dimensions does it really use?"""
    name = params.get("weight", "W_E")
    W = lab.backend.weight(name).float().cpu()
    if params.get("rows"):
        W = W[: int(params["rows"])]
    s = torch.linalg.svdvals(W - W.mean(0))
    var = s ** 2 / (s ** 2).sum()
    cum = var.cumsum(0)
    summary = {"weight": name, "shape": list(W.shape), "top_singular_values": s[:12].tolist(),
               "dims_for_90pct_variance": int((cum < 0.9).sum()) + 1, "dims_for_99pct_variance": int((cum < 0.99).sum()) + 1,
               "participation_ratio": float((s ** 2).sum() ** 2 / (s ** 4).sum())}
    k = min(40, len(s))
    fig = plots.bars(lab, list(range(1, k + 1)), var[:k].tolist(), f"Singular value spectrum of {name}", "component",
                     "share of variance")
    return summary, {"singular_values": s}, fig


REGISTRY = {
    "dataset": dataset,
    "baseline": baseline,
    "attribution_patching": attribution_patching,
    "activation_patching": activation_patching,
    "path_patching": path_patching,
    "ablation": ablation,
    "logit_lens": logit_lens,
    "direct_logit_attribution": direct_logit_attribution,
    "linear_probe": linear_probe,
    "attention_patterns": attention_patterns,
    "diff_of_means": diff_of_means,
    "steering": steering,
    "periodicity": periodicity,
    "weights_svd": weights_svd,
}
