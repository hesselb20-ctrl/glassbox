"""Score the blind benchmark runs against published ground truth and write the scorecard.

    python benchmarks/score.py

Reads benchmarks/<name>/ (notebook, experiments, report, session.json) and benchmarks/ground_truth.json.
Writes benchmarks/scorecard.json and benchmarks/SCORECARD.md.

Two kinds of scoring, kept apart on purpose:
  * mechanical: counted by this script from the run's own records (heads in the verified circuit,
    key frequencies, verification numbers, experiment counts, wall-clock time);
  * rubric: mechanism-level items from the papers, judged by a reader of the blind report. The
    judgement and the evidence ids it rests on are stored in ground_truth.json so anyone can disagree.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    d = HERE / name
    recs = {}
    for p in sorted((d / "experiments").glob("[EV]*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        recs[r["id"]] = r
    nb = [json.loads(l) for l in (d / "notebook.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    session = json.loads((d / "session.json").read_text(encoding="utf-8"))
    report = json.loads((d / "report" / "report.json").read_text(encoding="utf-8"))
    return {"recs": recs, "notebook": nb, "session": session, "report": report}


def common(run: dict) -> dict:
    nb, recs = run["notebook"], run["recs"]
    hyps = [e for e in nb if e["kind"] == "hypothesis" and not e.get("target")]
    updates = [e for e in nb if e.get("target") and e.get("status")]
    verifs = [r for i, r in recs.items() if i.startswith("V")]
    last = verifs[-1] if verifs else None
    return {
        "experiments": sum(1 for i in recs if i.startswith("E")),
        "verification_runs": len(verifs),
        "wall_clock_seconds": run["session"]["wall_clock_seconds"],
        "scientist_turns": run["session"]["turns"],
        "scientist_model": run["session"]["model_usage"],
        "hypotheses": len(hyps),
        "hypotheses_rejected_or_revised": sum(e["status"] in ("rejected", "revised") for e in updates),
        "final_verification": None if last is None else {
            "id": last["id"], "status": last["status"], "n_heldout": last["n_heldout"],
            "model_accuracy": last["model_accuracy"],
            "agreement_with_model": last["checks"].get("agreement", {}).get("agreement_with_model"),
        },
        "report_status": run["report"]["overall_status"],
        "claims": len(run["report"].get("claims", [])),
        "backend": sorted({r["provenance"]["tag"] for r in recs.values()}),
    }


def rubric(name: str) -> dict:
    items = GT[name]["rubric"]
    found = [i for i in items if i["verdict"] == "found"]
    partial = [i for i in items if i["verdict"] == "partial"]
    missed = [i for i in items if i["verdict"] == "missed"]
    return {"items": items, "found": len(found), "partial": len(partial), "missed": len(missed), "total": len(items)}


def score_ioi(run: dict) -> dict:
    last = [r for i, r in run["recs"].items() if i.startswith("V")][-1]
    circuit = {c for c in last["spec"]["circuit"]["components"] if not c.startswith(("mlp", "attn"))}
    mlps = sorted(c for c in last["spec"]["circuit"]["components"] if c.startswith("mlp"))
    classes = {}
    published = set()
    for cls, heads in GT["ioi"]["published_circuit"].items():
        published |= set(heads)
        classes[cls] = {"published": heads, "found": [h for h in heads if h in circuit], "missed": [h for h in heads if h not in circuit]}
    extra = sorted(circuit - published, key=lambda s: tuple(map(int, s.split("."))))
    k = last["checks"]["causal_components"]
    return {
        "published_heads": len(published), "heads_in_recovered_circuit": len(circuit),
        "published_heads_found": len(published & circuit), "published_heads_missed": len(published - circuit),
        "extra_heads_not_in_published_circuit": extra, "mlps_in_recovered_circuit": mlps, "by_class": classes,
        "recall": round(len(published & circuit) / len(published), 3),
        "precision": round(len(published & circuit) / len(circuit), 3),
        "circuit_alone_fraction_of_logit_diff": k["everything_else_ablated"]["fraction_of_clean"],
        "circuit_alone_accuracy": k["everything_else_ablated"]["accuracy"],
        "circuit_ablated_fraction_of_logit_diff": k["circuit_ablated"]["fraction_of_clean"],
        "random_same_size_ablation": k["random_same_size_ablated_fraction"],
    }


def score_modadd(run: dict) -> dict:
    import torch

    recs = run["recs"]
    found = sorted({f for r in recs.values() if r.get("primitive") == "periodicity" and r["params"].get("source", "W_E") == "W_E"
                    for f in r["summary"]["key_frequencies"]})
    out = {"key_frequencies_reported_for_W_E": found}
    ckpt = HERE / "modadd" / "model" / "model.pt"
    if ckpt.exists():   # independent recomputation straight from the weights, without the lab
        W = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]["embed.W_E"][:113]
        power = (torch.fft.fft(W - W.mean(0), dim=0).abs() ** 2).sum(1)[1:57]
        share = power / power.sum()
        top5 = sorted((share.argsort(descending=True)[:5] + 1).tolist())
        out["top5_frequencies_recomputed_from_weights"] = top5
        out["top5_share_of_embedding_power"] = round(float(share.sort(descending=True).values[:5].sum()), 3)
        out["top5_all_reported"] = all(f in found for f in top5)
    v = [r for i, r in recs.items() if i.startswith("V")][-1]["checks"]["causal_subspace"]
    out.update({"subspace_dims": f"{sum(v['dims'])} of {v['of_total_dims'][0]}",
                "accuracy_only_subspace": v["only_subspace_kept"]["accuracy"],
                "accuracy_subspace_removed": v["subspace_removed"]["accuracy"],
                "accuracy_random_subspace_removed": v["random_subspace_removed"]["accuracy"]})
    return out


def score_induction(run: dict) -> dict:
    v = [r for i, r in run["recs"].items() if i.startswith("V")]
    out = {"verification_runs": [{"id": r["id"], "status": r["status"], "metric": r["spec"].get("metric", {}).get("type", "dataset default"),
                                  "agreement": r["checks"].get("agreement", {}).get("agreement_with_model"),
                                  "model_accuracy": r["model_accuracy"]} for r in v]}
    k = v[-1]["checks"].get("causal_components")
    if k:
        out.update({"circuit": k["circuit"], "circuit_alone": k["everything_else_ablated"]["fraction_of_clean"],
                    "circuit_ablated": k["circuit_ablated"]["fraction_of_clean"],
                    "random_same_size_ablation": k["random_same_size_ablated_fraction"]})
    return out


def main() -> None:
    card = {"note": GT["blind_protocol"], "benchmarks": {}}
    for name, fn in (("modadd", score_modadd), ("induction", score_induction), ("ioi", score_ioi)):
        run = load(name)
        card["benchmarks"][name] = {"title": GT[name]["title"], "ground_truth": GT[name]["source"], **common(run),
                                    "mechanical": fn(run), "rubric": rubric(name)}
    (HERE / "scorecard.json").write_text(json.dumps(card, indent=1), encoding="utf-8")

    md = ["# Benchmark scorecard", "", GT["blind_protocol"], "",
          "| Benchmark | Status reported | Rubric: found / partial / missed | Experiments | Wall clock | Hypotheses (rejected or revised) |",
          "|---|---|---|---|---|---|"]
    for name, b in card["benchmarks"].items():
        r = b["rubric"]
        md.append(f"| {b['title']} | {b['report_status']} | {r['found']} / {r['partial']} / {r['missed']} of {r['total']} | "
                  f"{b['experiments']} (+{b['verification_runs']} verification) | {b['wall_clock_seconds'] / 60:.1f} min | "
                  f"{b['hypotheses']} ({b['hypotheses_rejected_or_revised']}) |")
    for name, b in card["benchmarks"].items():
        md += ["", f"## {b['title']}", "", f"Ground truth: {b['ground_truth']}", "",
               f"Backend: {', '.join(b['backend'])}. Scientist: headless Claude Code ({', '.join(b['scientist_model'])}), "
               f"{b['scientist_turns']} turns, MCP tools only.", "",
               "| Published mechanism | Verdict | Evidence in the blind run | Comment |", "|---|---|---|---|"]
        md += [f"| {i['item']} | **{i['verdict']}** | {', '.join(i['evidence']) or '-'} | {i['comment']} |" for i in b["rubric"]["items"]]
        m = b["mechanical"]
        md += ["", "Counted from the run's records:", ""]
        if name == "ioi":
            md += [f"- Recovered circuit: {m['heads_in_recovered_circuit']} heads plus {', '.join(m['mlps_in_recovered_circuit'])}. "
                   f"Published circuit: {m['published_heads']} heads. Overlap: {m['published_heads_found']} found, "
                   f"{m['published_heads_missed']} missed (recall {m['recall']}, precision {m['precision']}).",
                   "", "| Class (Wang et al. 2022) | Published | Found | Missed |", "|---|---|---|---|"]
            md += [f"| {c} | {', '.join(v['published'])} | {', '.join(v['found']) or '-'} | {', '.join(v['missed']) or '-'} |"
                   for c, v in m["by_class"].items()]
            md += ["", f"- Heads in the recovered circuit that are not in the published one: {', '.join(m['extra_heads_not_in_published_circuit']) or 'none'}.",
                   f"- Held-out causal check: circuit alone keeps {m['circuit_alone_fraction_of_logit_diff']:.2f} of the logit difference "
                   f"(top-1 accuracy {m['circuit_alone_accuracy']:.2f}); circuit ablated leaves {m['circuit_ablated_fraction_of_logit_diff']:.2f}; "
                   f"random same-size ablations leave {', '.join(f'{x:.2f}' for x in m['random_same_size_ablation'])}."]
        elif name == "modadd":
            md += [f"- Key frequencies the run reported for the embedding: {m['key_frequencies_reported_for_W_E']}.",
                   f"- Top 5 frequencies recomputed independently from the weights: {m.get('top5_frequencies_recomputed_from_weights')} "
                   f"({m.get('top5_share_of_embedding_power')} of embedding power). All reported: {m.get('top5_all_reported')}.",
                   f"- Held-out causal check on the {m['subspace_dims']} dim key-frequency subspace of the embedding: accuracy "
                   f"{m['accuracy_only_subspace']:.3f} with only the subspace, {m['accuracy_subspace_removed']:.3f} with it removed, "
                   f"{m['accuracy_random_subspace_removed']:.3f} with a random subspace of the same size removed."]
        else:
            for r in m["verification_runs"]:
                md.append(f"- {r['id']}: {r['status']} (agreement with model {r['agreement']}, model accuracy {r['model_accuracy']}).")
            if "circuit" in m:
                md.append(f"- Circuit {m['circuit']}: alone keeps {m['circuit_alone']:.2f} of the metric, ablated leaves {m['circuit_ablated']:.2f}; "
                          f"random same-size ablations leave {', '.join(f'{x:.2f}' for x in m['random_same_size_ablation'])}.")
        fv = b["final_verification"]
        md += ["", f"Final verification {fv['id']}: **{fv['status']}** on {fv['n_heldout']} held-out inputs. "
                   f"Report: [`{name}/report/report.md`]({name}/report/report.md). Notebook: [`{name}/notebook.jsonl`]({name}/notebook.jsonl)."]
    md += ["", "## What this does not show", ""] + [f"- {x}" for x in GT["caveats"]]
    (HERE / "SCORECARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
