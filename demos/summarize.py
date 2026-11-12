"""Collect the numbers of the 27B demo run into demos/qwen27b-language/summary.json (used by the README image and the video).

Nothing is computed here that the run did not measure: this only picks fields out of the experiment records.

    python demos/summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parent / "qwen27b-language"


def rec(eid: str) -> dict:
    return json.loads(next((D / "experiments").glob(f"{eid}_*.json")).read_text(encoding="utf-8"))


def main() -> None:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((D / "experiments").glob("E*.json"))]
    probe = next(r for r in recs if r["primitive"] == "linear_probe")
    dom = [r for r in recs if r["primitive"] == "diff_of_means"]
    steer = [r for r in recs if r["primitive"] == "steering"]
    main_art = dom[0]["summary"]["direction_artifact"]

    def pick(site: str, sign: int, artifact: str) -> dict:
        """The steering run at ``site`` with the most dose levels, in the given direction, using the given direction artifact."""
        c = [r for r in steer if r["summary"]["sites"] == [site] and r["params"]["direction"]["artifact"] == artifact
             and all(a * sign >= 0 for a in r["summary"]["alphas"])]
        return max(c, key=lambda r: len(r["summary"]["alphas"]))

    fwd, back = pick("resid_post.28", 1, main_art), pick("resid_post.28", -1, main_art)
    control = next((r for r in steer if r["params"]["direction"]["artifact"] != main_art), None)

    def gens(r: dict, k: int) -> list[dict]:
        out = []
        for g in r["summary"]["generations"][:k]:
            alphas = sorted(g["outputs"], key=lambda a: abs(float(a)))
            out.append({"prompt": g["prompt"], "before": g["outputs"][alphas[0]].strip(), "after": g["outputs"][alphas[-1]].strip(),
                        "alpha": float(alphas[-1]), "evidence": r["id"]})
        return out

    nb = [json.loads(l) for l in (D / "notebook.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    info = nb[0]["data"]["info"]
    session = json.loads((D / "session.json").read_text(encoding="utf-8"))
    by_site = {}
    for r in steer:
        if r["params"]["direction"]["artifact"] == main_art and all(a >= 0 for a in r["summary"]["alphas"]):
            layer = int(r["summary"]["sites"][0].split(".")[1])
            by_site.setdefault(layer, []).append(r["id"])
    out = {
        "title": "A language direction found and steered in a 27B model on a home PC",
        "model": info["file"], "label": "quantized, abliterated community variant, screening result", "tag": info["tag"],
        "n_layers": info["n_layers"], "gpus": [f"{k.replace('NVIDIA GeForce ', '')} ({v} GB used)" for k, v in info["vram_used_gb"].items()],
        "device": info["device"],
        "separation_name": "probe accuracy (held-out)",
        "separation_by_layer": probe["summary"]["test_accuracy"], "control_by_layer": probe["summary"]["shuffled_label_control"],
        "lens_name": "direction d' (held-out)", "lens_by_layer": dom[0]["summary"]["heldout_separation_dprime"],
        "best_layer": int(dom[0]["summary"]["best_site"].split(".")[1]),
        "steer_layers": sorted(by_site), "steer_sites": fwd["summary"]["sites"],
        "steering": {"alphas": fwd["summary"]["alphas"], "metric": list(fwd["summary"]["metric_by_alpha"].values()),
                     "metric_name": "metric: logit(' the') - logit(' la') on 8 held-out English prompts; below 0 = French",
                     "metric_short": "logit(the) - logit(la)",
                     "control_metric": list(control["summary"]["metric_by_alpha"].values()) if control else None,
                     "control_alphas": control["summary"]["alphas"] if control else None,
                     "evidence": [fwd["id"]] + ([control["id"]] if control else [])},
        "generations": gens(fwd, 2) + gens(back, 1),
        "experiments": len(recs), "wall_clock_seconds": session["wall_clock_seconds"],
        "evidence": {"probe": probe["id"], "direction": dom[0]["id"], "steer_forward": fwd["id"], "steer_back": back["id"],
                     "control_direction": control["id"] if control else None},
    }
    (D / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_by_layer")}, indent=1, ensure_ascii=False)[:2200])


if __name__ == "__main__":
    main()
