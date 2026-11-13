"""Grade bench runs against the sealed answer keys.

This is the only code that reads the sealed folder. The MCP server never imports this module, and a
headless scientist has no file access at all. It is still an honour-system seal on one PC, not
cryptographic isolation: anyone with a shell on the machine can open the folder.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..notebook import Notebook
from . import truth as T
from .generate import ROOT, build_model, sample_inputs, sealed_dir

HELDOUT_SEED, HELDOUT_N = 424242, 600


def _predictor(code: str, unified: list[str]):
    ns = {"math": math, "np": np, "torch": torch, "tok": unified.index, "detok": lambda i: unified[int(i)]}
    exec(compile(code, "<recovered algorithm>", "exec"), ns)
    return ns["predict"]


def grade_task(run: Path, task_dir: Path, key: dict) -> dict:
    public = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    unified, n = public["vocab"], public["seq_len"]
    out: dict = {"id": key["id"], "track": key["track"], "family": key["family"], "label": key["label"]}
    exp = run / "experiments"
    verifs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(exp.glob("V*.json"))] if exp.exists() else []
    session = json.loads((run / "session.json").read_text(encoding="utf-8")) if (run / "session.json").exists() else {}
    out.update({"ran": bool(session), "experiments": len(list(exp.glob("E*.json"))) if exp.exists() else 0,
                "wall_clock_seconds": session.get("wall_clock_seconds"), "verification_runs": len(verifs)})
    if (run / "notebook.jsonl").exists():
        nb = Notebook(run / "notebook.jsonl")
        pr = nb.prediction_record()
        out["prereg"] = {k: pr[k] for k in ("preregistered", "run", "confirmed", "refuted", "inconclusive", "hit_rate")}
        out["notebook_chain_ok"] = nb.verify()["ok"]
    final = verifs[-1] if verifs else None
    out["final_verification"] = final["status"] if final else None
    out["beats_null_controls"] = bool(final and final.get("beats_null_controls"))
    spec = (final or {}).get("spec", {})

    # ---- circuit: recall and precision against the true components (compiled track only)
    claimed = sorted(set(((spec.get("circuit") or {}).get("components")) or []))
    out["claimed_circuit"] = claimed
    if key.get("true_components") is not None:
        true = set(key["true_components"])
        hit = true & set(claimed)
        out["circuit"] = {"true": sorted(true), "found": sorted(hit), "missed": sorted(true - set(claimed)),
                          "extra": sorted(set(claimed) - true),
                          "recall": round(len(hit) / len(true), 3) if true else None,
                          "precision": round(len(hit) / len(claimed), 3) if claimed else None}
    else:
        out["circuit"] = None                       # trained track: mechanism unknown, never scored on circuit match

    # ---- the recovered algorithm on held-out inputs: against the true function, and against the model
    xs = sample_inputs(public["input_token_ids"], n, HELDOUT_N, HELDOUT_SEED)
    ckpt = torch.load(task_dir / "model.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        model_out = build_model(ckpt["cfg"], ckpt["state_dict"])(torch.tensor(xs))[:, -1].argmax(-1).tolist()
    want = [unified.index(T.truth(key["family"], key["params"], [unified[i] for i in x[1:]])) for x in xs]
    out["model_accuracy_on_true_function"] = float(np.mean([a == b for a, b in zip(model_out, want)]))
    out["heldout_inputs"] = len(xs)
    code = spec.get("code")
    if code:
        try:
            predict = _predictor(code, unified)
            got = []
            for x in xs:
                p = predict(list(x), None)
                got.append(unified.index(p) if isinstance(p, str) else int(p))
            out["mechanism_match"] = float(np.mean([a == b for a, b in zip(got, want)]))       # recovered algorithm == true algorithm
            out["behavior_prediction"] = float(np.mean([a == b for a, b in zip(got, model_out)]))  # recovered algorithm == model
        except Exception as e:  # noqa: BLE001
            out.update({"mechanism_match": 0.0, "behavior_prediction": 0.0, "algorithm_error": f"{type(e).__name__}: {e}"[:200]})
    else:
        out.update({"mechanism_match": None, "behavior_prediction": None, "algorithm_error": "no algorithm was submitted to verify_hypothesis"})
    return out


def grade(seed: int, out_root: Path | str | None = None, reveal: bool = False) -> dict:
    out_root = Path(out_root) if out_root else ROOT / "benchmarks" / "glassbox-bench"
    base, seal = out_root / f"seed-{seed}", sealed_dir(seed)
    if not seal.exists():
        raise FileNotFoundError(f"no sealed answer keys for seed {seed} in {seal.parent}")
    rows = []
    for kp in sorted(seal.glob("t*.json"), key=lambda p: int(p.stem[1:])):
        key = json.loads(kp.read_text(encoding="utf-8"))
        rows.append(grade_task(base / "runs" / key["id"], base / "tasks" / key["id"], key))
    comp = [r for r in rows if r["track"] == "compiled"]
    mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None  # noqa: E731
    card = {
        "seed": seed, "tasks": rows,
        "summary": {
            "tasks": len(rows), "ran": sum(r["ran"] for r in rows),
            "compiled_circuit_recall": mean([r["circuit"]["recall"] or 0.0 for r in comp]),
            "compiled_circuit_precision": mean([r["circuit"]["precision"] or 0.0 for r in comp]),
            "compiled_mechanism_match": mean([r["mechanism_match"] or 0.0 for r in comp]),
            "compiled_solved": sum((r["mechanism_match"] or 0) >= 0.98 and (r["circuit"]["recall"] or 0) == 1.0 for r in comp),
            "behavior_prediction_all": mean([r["behavior_prediction"] or 0.0 for r in rows]),
            "verified": sum(r["final_verification"] == "verified" for r in rows),
            "beats_null_controls": sum(r["beats_null_controls"] for r in rows),
            "experiments_mean": mean([r["experiments"] for r in rows]),
            "wall_clock_minutes_total": round(sum((r["wall_clock_seconds"] or 0) for r in rows) / 60, 1),
            "prereg_confirmed": sum((r.get("prereg") or {}).get("confirmed", 0) for r in rows),
            "prereg_refuted": sum((r.get("prereg") or {}).get("refuted", 0) for r in rows),
        },
        "definitions": {
            "mechanism_match": "share of held-out inputs where the recovered predict() equals the true algorithm of the answer key",
            "behavior_prediction": "share of held-out inputs where the recovered predict() equals the model's own output",
            "circuit": "heads and MLP layers named in the final verify_hypothesis call, against the components that carry the compiled program",
            "compiled_solved": "mechanism match >= 0.98 and every true component found",
        },
    }
    s = card["summary"]
    dec = s["prereg_confirmed"] + s["prereg_refuted"]
    s["prereg_hit_rate"] = round(s["prereg_confirmed"] / dec, 3) if dec else None
    (base / "scorecard.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    md = [f"# glassbox-bench, seed {seed}", "",
          f"{s['ran']} of {s['tasks']} tasks run. Compiled track: {s['compiled_solved']} of {len(comp)} solved "
          f"(mean circuit recall {s['compiled_circuit_recall']}, precision {s['compiled_circuit_precision']}, mechanism match "
          f"{s['compiled_mechanism_match']}). Prediction hit rate {s['prereg_hit_rate']}. "
          f"{s['beats_null_controls']} of {s['tasks']} final findings beat their null controls.", "",
          "| Task | Track | Family (from the key) | Circuit recall | Circuit precision | Mechanism match | Behavior prediction | Experiments | Minutes | Predictions (confirmed / refuted) | Beats nulls | Final status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    for r in rows:
        c, p = r["circuit"] or {}, r.get("prereg") or {}
        md.append(f"| {r['id']} | {r['track']} | {r['family']} | {f(c.get('recall')) if r['circuit'] else 'not scored'} | "
                  f"{f(c.get('precision')) if r['circuit'] else 'not scored'} | {f(r['mechanism_match'])} | {f(r['behavior_prediction'])} | "
                  f"{r['experiments']} | {f((r['wall_clock_seconds'] or 0) / 60)} | {p.get('confirmed', 0)} / {p.get('refuted', 0)} | "
                  f"{'yes' if r['beats_null_controls'] else 'no'} | {r['final_verification'] or 'none'} |")
    md += ["", "Trained-track tasks are behaviorally specified with an unknown mechanism: they are scored on behavior prediction only.",
           "", "Missed and extra components:", ""]
    md += [f"- {r['id']} ({r['family']}): missed {r['circuit']['missed'] or 'none'}, extra {r['circuit']['extra'] or 'none'}" for r in comp]
    (base / "SCORECARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if reveal:
        rev = out_root / "revealed" / f"seed-{seed}"
        rev.mkdir(parents=True, exist_ok=True)
        for kp in seal.glob("t*.json"):
            (rev / kp.name).write_text(kp.read_text(encoding="utf-8"), encoding="utf-8")
        card["revealed_to"] = str(rev.relative_to(out_root.parent.parent)) if out_root.is_relative_to(ROOT) else str(rev)
    return card
