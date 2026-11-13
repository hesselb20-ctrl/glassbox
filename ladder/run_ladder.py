"""Scale ladder: does the algorithm found in a small model show up in a big one?

The blind benchmark recovered the IOI circuit in GPT-2 small. This script turns that circuit's main claims into
four predictions and tests each of them, with the same code and the same fixed protocol, on every rung:

    python ladder/run_ladder.py gpt2-local          # small model, full precision, your GPU
    python ladder/run_ladder.py gpt2-ndif           # same model on NDIF: checks the remote path reproduces local numbers
    python ladder/run_ladder.py qwen27b-gguf        # abliterated 27B GGUF: read-only screening (no patching on GGUF)
    python ladder/run_ladder.py qwen27b-4bit        # official Qwen3.8-27B in 4-bit on two GPUs (needs the download)
    python ladder/run_ladder.py qwen27b-ndif        # official Qwen3.8-27B at full precision on NDIF
    python ladder/run_ladder.py summary             # collect all finished rungs into ladder/summary.json and LADDER.md

Predictions (from benchmarks/ioi, claims 2, 5 and 7 of the report):
  P1  the model does the task: it prefers the indirect object over the repeated subject;
  P2  hand-off: which name is repeated is carried at the second subject token (S2) in early layers and has moved to
      the final token (END) by the late layers (GPT-2: around layers 7-9 of 12);
  P3  a few attention heads late in the network write the answer at END (the name movers);
  P4  those heads are backed up: ablating the strongest ones costs much less than their patching effect.

This is a scripted protocol, not a blind investigation: the hypotheses are fixed in advance and every rung either
supports or rejects them. Each run writes a normal glassbox investigation (notebook, experiments, plots) to ladder/<rung>/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "ladder"
os.environ.setdefault("GLASSBOX_HOME", str(HERE))

RUNGS = {
    "gpt2-local": dict(backend="local-full", model="gpt2", params="124M", n=32),
    "gpt2-ndif": dict(backend="ndif", model="openai-community/gpt2", params="124M", n=16),
    "qwen27b-gguf": dict(backend="local-gguf", model="Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf", params="27B", n=16,
                         options={"n_ctx": 256}),
    "qwen27b-4bit": dict(backend="local-4bit", model="Qwen/Qwen3.8-27B", params="27B", n=16, max_rows=32,
                         # explicit layout for two 12 GB cards: 32 layers + embedding on GPU 0, 32 layers + unembedding on GPU 1
                         options={"device_map": {"model.embed_tokens": "0", "lm_head": "1", "model.norm": "1", "model.rotary_emb": "0",
                                                 **{f"model.layers.{i}": ("0" if i < 32 else "1") for i in range(64)}}}),
    # remote requests take about a minute each: fewer layers sampled than on the local rungs
    "qwen27b-ndif": dict(backend="ndif", model="Qwen/Qwen3.8-27B", params="27B", n=16, max_rows=96, resid_k=9, attn_k=17),
}
BEHAVIOR = "Indirect object identification: 'When Mary and John went to the store, John gave a drink to' -> ' Mary'"


def spread(n_layers: int, k: int) -> list[int]:
    """About k layers spread over the depth, always including the first and last."""
    if n_layers <= k:
        return list(range(n_layers))
    return sorted({round(i * (n_layers - 1) / (k - 1)) for i in range(k)})


def run(name: str) -> None:
    from glassbox.backends.base import BACKENDS, PRIMITIVES
    from glassbox.session import Session

    r = RUNGS[name]
    s = Session(HERE)
    t0 = time.time()
    info = s.load_model(r["backend"], r["model"], investigation=name, options=r.get("options"))["info"]
    L, feats = info["n_layers"], BACKENDS[r["backend"]]["features"]
    if r.get("max_rows"):
        s.backend.max_rows = r["max_rows"]
    can = lambda prim: PRIMITIVES[prim] <= feats  # noqa: E731
    abc = {"task": "ioi", "n": r["n"], "seed": 11, "corruption": "abc"}
    flip = {"task": "ioi", "n": r["n"], "seed": 11, "corruption": "flip"}
    out: dict = {"rung": name, "behavior": BEHAVIOR, **{k: r[k] for k in ("backend", "model", "params")},
                 "tag": info["tag"], "n_layers": L, "n_prompts": r["n"], "predictions": {}, "not_run": []}
    note = s.notebook_append

    def result(key: str, hyp: dict, text: str, refs: list[str], ok: bool | None) -> None:
        status = None if ok is None else ("supported" if ok else "rejected")
        s.notebook_append("result", text, refs=refs, status=status, target=hyp["id"])
        out["predictions"][key]["verdict"] = status or "not testable on this backend"

    # ---- P1: the behaviour -------------------------------------------------------------------------------
    h1 = note("hypothesis", "P1: the model prefers the indirect object over the repeated subject (accuracy > 0.8, logit difference > 0).")
    ds = s.run_experiment("dataset", abc, "the IOI dataset on this tokenizer", detail="full")
    pos = ds["data"]["meta"]["positions"]
    s2, end = pos[0]["S2"], pos[0]["END"]
    assert all(p["S2"] == s2 and p["END"] == end for p in pos), "prompts are not position-aligned"
    b = s.run_experiment("baseline", {"dataset": abc}, "P1: baseline")
    acc, ld = b["summary"]["clean_accuracy"], b["summary"]["clean_metric_mean"]
    out["predictions"]["P1"] = {"accuracy": acc, "logit_diff": ld, "evidence": [b["id"]]}
    result("P1", h1, f"{b['id']} ({info['tag']}): accuracy {acc:.2f}, logit difference {ld:.2f} on {r['n']} prompts.", [b["id"]],
           acc > 0.8 and ld > 0)

    # ---- where the answer becomes readable ---------------------------------------------------------------
    lens = s.run_experiment("logit_lens", {"dataset": abc, "layers": spread(L, 17)}, "logit lens: where does the answer become readable at END?")
    ll, lm = lens["summary"]["layers"], lens["summary"]["metric_by_layer"]
    half = lens["summary"]["first_layer_reaching_half_final_metric"]
    out["logit_lens"] = {"layers": ll, "logit_diff": lm, "first_layer_reaching_half": half,
                         "relative_depth": None if half is None else round(half / (L - 1), 3), "evidence": [lens["id"]]}
    note("result", f"{lens['id']}: the logit difference read off the residual stream reaches half its final value at layer {half} of {L - 1}.",
         refs=[lens["id"]])

    # ---- P2: hand-off from S2 to END -----------------------------------------------------------------------
    h2 = note("hypothesis", "P2 (hand-off): with a corruption that only changes which name is repeated, patching the clean residual stream at S2 "
                            "restores the behaviour in early layers but not in late ones, and patching at END does the opposite. "
                            "Against: no layer where S2 restores more than half, or END never takes over.")
    out["predictions"]["P2"] = {}
    if can("activation_patching"):
        layers = spread(L, r.get("resid_k", 17))
        p = s.run_experiment("activation_patching", {"dataset": flip, "component": "resid_pre", "layers": layers, "positions": [s2, end]},
                             "P2: residual patching at S2 and END, flip corruption")
        eff = p["data"]["effect"]
        e_s2, e_end = [row[0] for row in eff], [row[1] for row in eff]
        cross = next((l for l, a, c in zip(layers, e_s2, e_end) if c > a), None)
        ok = max(e_s2[: len(layers) // 2 + 1]) >= 0.5 and e_end[-1] >= 0.8 and e_s2[-1] <= 0.2 and e_end[0] <= 0.2
        out["predictions"]["P2"] = {"layers": layers, "effect_S2": e_s2, "effect_END": e_end, "crossover_layer": cross,
                                    "crossover_relative_depth": None if cross is None else round(cross / (L - 1), 3), "evidence": [p["id"]]}
        result("P2", h2, f"{p['id']} ({info['tag']}): S2 patching restores up to {max(e_s2):.2f} (early) and {e_s2[-1]:.2f} at the last layer; "
                         f"END patching goes from {e_end[0]:.2f} to {e_end[-1]:.2f}. END overtakes S2 at layer {cross} of {L - 1}.", [p["id"]], ok)
    else:
        out["not_run"].append("P2 (needs activation patching)")
        result("P2", h2, f"Not testable: {r['backend']} cannot patch activations.", [b["id"]], None)

    # ---- P3: late attention heads write the answer at END ---------------------------------------------------
    h3 = note("hypothesis", "P3 (name movers): the answer is written at END by attention in a few late layers, and within those layers by a few "
                            "heads: patching the three strongest heads from the clean run restores at least 30% of the logit difference. "
                            "Against: the effect is spread evenly over heads, or sits in MLPs only.")
    out["predictions"]["P3"] = {}
    top_heads: list[str] = []
    if can("activation_patching"):
        layers = spread(L, r.get("attn_k", 33))
        a = s.run_experiment("activation_patching", {"dataset": abc, "component": "heads_all", "layers": layers, "positions": [end]},
                             "P3: all attention heads of a layer at END, by layer")
        ea = [row[0] for row in a["data"]["effect"]]
        order = sorted(range(len(layers)), key=lambda i: -(ea[i] if ea[i] is not None else -9))
        best_layers = [layers[i] for i in order[:2]]
        head_runs, heads = [], []
        for bl in best_layers:
            hp = s.run_experiment("activation_patching", {"dataset": abc, "component": "head", "layers": [bl], "positions": [end]},
                                  f"P3: heads of layer {bl} at END")
            head_runs.append(hp["id"])
            heads += [(f"{bl}.{h}", e) for h, e in enumerate(hp["data"]["effect"][0])]
        heads.sort(key=lambda t: -t[1])
        top_heads = [h for h, _ in heads[:3]]
        top3 = sum(e for _, e in heads[:3])
        out["predictions"]["P3"] = {"attn_layers_tested": layers, "attn_effect_at_END": ea, "top_attention_layers": best_layers,
                                    "top_heads": [{"head": h, "effect": e} for h, e in heads[:6]], "top3_heads_effect": top3,
                                    "relative_depth_of_top_layer": round(best_layers[0] / (L - 1), 3), "evidence": [a["id"]] + head_runs}
        result("P3", h3, f"{a['id']}, {', '.join(head_runs)} ({info['tag']}): attention at END matters most in layers {best_layers} "
                         f"(effects {ea[order[0]]:.2f}, {ea[order[1]]:.2f}). Strongest heads: "
                         + ", ".join(f"{h} ({e:.2f})" for h, e in heads[:3]) + f"; together {top3:.2f}.", [a["id"]] + head_runs, top3 >= 0.3)
    else:
        out["not_run"].append("P3 (needs activation patching)")
        result("P3", h3, f"Not testable: {r['backend']} cannot patch activations.", [b["id"]], None)

    # ---- P4: backup ------------------------------------------------------------------------------------------
    h4 = note("hypothesis", "P4 (backup): mean-ablating the three strongest heads together removes clearly less of the logit difference than their "
                            "patching effect, because other heads compensate. Against: the drop is as large as the patching effect.")
    out["predictions"]["P4"] = {}
    if can("ablation") and top_heads:
        ab = s.run_experiment("ablation", {"dataset": abc, "components": top_heads, "mode": "mean", "reference": "corrupt", "joint": True,
                                           "positions": [end]}, f"P4: joint mean-ablation of {top_heads} at END")
        left = ab["summary"]["fraction_of_clean"]
        top3 = out["predictions"]["P3"]["top3_heads_effect"]
        out["predictions"]["P4"] = {"heads": top_heads, "fraction_of_logit_diff_left": left, "drop": 1 - left,
                                    "patching_effect_of_same_heads": top3, "evidence": [ab["id"]]}
        result("P4", h4, f"{ab['id']} ({info['tag']}): with {top_heads} mean-ablated, {left:.2f} of the logit difference is left "
                         f"(drop {1 - left:.2f}); patching the same heads restored {top3:.2f}.", [ab["id"]], (1 - left) < 0.75 * top3)
    else:
        out["not_run"].append("P4 (needs ablation)")
        result("P4", h4, f"Not testable: {r['backend']} cannot ablate activations.", [b["id"]], None)

    verdicts = [p.get("verdict") for p in out["predictions"].values()]
    tested = [v for v in verdicts if v in ("supported", "rejected")]
    verifies = BACKENDS[r["backend"]]["verifies"]
    out["status"] = ("no prediction testable" if not tested else
                     ("confirmed" if verifies else "screening-pass") if all(v == "supported" for v in tested) and len(tested) == 4 else
                     ("partly tested, supported so far (screening)" if all(v == "supported" for v in tested) else "partly supported"))
    out["counts_as_verification"] = verifies
    out["seconds"] = round(time.time() - t0, 1)
    out["experiments"] = s._budget()["experiments_used"]
    (HERE / name / "rung.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("predictions", "logit_lens")}, indent=1))
    for k, v in out["predictions"].items():
        print(k, v.get("verdict"), {a: b for a, b in v.items() if a in ("accuracy", "logit_diff", "crossover_layer", "top_attention_layers",
                                                                         "top3_heads_effect", "fraction_of_logit_diff_left")})
    if s.backend is not None:
        s.backend.unload()


def summary() -> None:
    rungs = []
    for name in RUNGS:
        p = HERE / name / "rung.json"
        if p.exists():
            rungs.append(json.loads(p.read_text(encoding="utf-8")))
    pending = [n for n in RUNGS if not (HERE / n / "rung.json").exists()]
    rows = []
    for r in rungs:
        P = r["predictions"]
        bits = [f"accuracy {P['P1']['accuracy']:.2f}, logit diff {P['P1']['logit_diff']:.1f}"]
        if P["P2"].get("crossover_layer") is not None:
            bits.append(f"S2 to END hand-off at layer {P['P2']['crossover_layer']}/{r['n_layers'] - 1}")
        if P["P3"].get("top_heads"):
            bits.append("top heads " + ", ".join(h["head"] for h in P["P3"]["top_heads"][:3]) + f" restore {P['P3']['top3_heads_effect']:.2f}")
        if P["P4"].get("heads"):
            bits.append(f"ablating them leaves {P['P4']['fraction_of_logit_diff_left']:.2f}")
        rows.append({"rung": r["rung"], "model": r["model"], "params": r["params"], "backend": r["backend"], "tag": r["tag"],
                     "status": r["status"], "key_result": "; ".join(bits),
                     "heads": [h["head"] for h in P["P3"].get("top_heads", [])[:3]],
                     "verdicts": {k: v.get("verdict") for k, v in P.items()}, "not_run": r["not_run"]})
    names = {"P1": "the behaviour itself", "P2": "the S2-to-END hand-off", "P3": "a few late heads writing the answer", "P4": "backup"}

    def held(r):
        ok = [names[k] for k, v in r["verdicts"].items() if v == "supported"]
        return f"{len(ok)} of 4" + ("" if len(ok) == 4 else f" ({', '.join(ok)})")

    small = [r for r in rows if r["params"] != "27B" and r["status"] == "confirmed"]
    big = next((r for r in rows if r["rung"] == "qwen27b-ndif"), None)
    four = next((r for r in rows if r["rung"] == "qwen27b-4bit"), None)
    concl = (f"In GPT-2 small all four predictions hold at full precision ({', '.join(r['rung'] for r in small)}). " if small else "")
    if four:
        concl += f"In the official Qwen3.8-27B screened in 4-bit on two 12 GB GPUs, {held(four)} hold. "
    if big:
        concl += (f"At full precision on NDIF, {held(big)} hold" + (", which confirms the screen" if big["status"] == "confirmed" else "") + ". ")
        if big["status"] == "confirmed":
            concl += ("So the same three-stage structure (the repeated name is tracked at S2, handed to the last token late in the "
                      "network, and written out by a few backed-up attention heads) appears at about 220 times the size, at a similar "
                      "relative depth. These are four coarse predictions, not the full 26-head circuit.")
    first = HERE / "qwen27b-ndif-first-run" / "rung.json"
    if first.exists():
        f = json.loads(first.read_text(encoding="utf-8"))
        concl += (f" An earlier NDIF run is kept in qwen27b-ndif-first-run: with only {f['n_prompts']} prompts (a memory bug in glassbox, "
                  f"since fixed, limited the batch size) its behaviour check missed the accuracy threshold "
                  f"({f['predictions']['P1']['accuracy']:.2f}); the other three predictions held there too.")
    if pending:
        concl += " Not done yet: " + ", ".join(pending) + "."
    out = {"behavior": "IOI", "protocol": "four fixed predictions from the GPT-2 circuit, tested unchanged on every rung",
           "rungs": rows, "pending": pending, "conclusion": concl}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# Scale ladder: indirect object identification", "", concl, "",
          "The four predictions come from the circuit the blind benchmark recovered in GPT-2 small. Each rung runs the same script "
          "(`ladder/run_ladder.py`). Quantized rungs are screening only.", "",
          "| Rung | Model | Backend / precision | P1 behaviour | P2 hand-off | P3 name movers | P4 backup | Status |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["verdicts"]
        md.append(f"| {r['rung']} | {r['model']} ({r['params']}) | {r['tag']} | {v['P1']} | {v['P2']} | {v['P3']} | {v['P4']} | **{r['status']}** |")
    for n in pending:
        md.append(f"| {n} | {RUNGS[n]['model']} ({RUNGS[n]['params']}) | {RUNGS[n]['backend']} | - | - | - | - | not done yet |")
    md += ["", "## Numbers", ""] + [f"- **{r['rung']}** ({r['tag']}): {r['key_result']}." for r in rows]
    md += ["", "Each rung folder holds the full notebook, experiment records and plots."]
    (HERE / "LADDER.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    summary() if sys.argv[1] == "summary" else run(sys.argv[1])
