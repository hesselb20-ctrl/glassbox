"""Collect the real numbers the tutorial video animates into tutorial/src/data.json.

Everything comes from committed run records (benchmarks/, demos/, ladder/). Sections whose runs do not
exist yet are written as {"available": false} and the video shows them as "not run yet".

    python tutorial/make_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tutorial" / "src" / "data.json"


def rec(folder: Path, eid: str) -> dict:
    return json.loads(next((folder / "experiments").glob(f"{eid}_*.json")).read_text(encoding="utf-8"))


def notebook(folder: Path) -> list[dict]:
    return [json.loads(l) for l in (folder / "notebook.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def modadd() -> dict:
    d = ROOT / "benchmarks" / "modadd"
    e3, v1 = rec(d, "E0003"), rec(d, "V0001")
    nb = {e["id"]: e for e in notebook(d)}
    sub = v1["checks"]["causal_subspace"]
    report = json.loads((d / "report" / "report.json").read_text(encoding="utf-8"))
    session = json.loads((d / "session.json").read_text(encoding="utf-8"))
    training = json.loads((d / "model" / "training.json").read_text(encoding="utf-8"))
    hyp = next(e for e in nb.values() if e["kind"] == "hypothesis" and "Fourier" in e["text"])
    res = next(e for e in nb.values() if e["kind"] == "result" and "E0003" in e.get("refs", []))
    revised = [e for e in notebook(d) if e.get("status") == "revised"]
    return {
        "available": True,
        "p": 113,
        "frequencies": e3["data"]["frequencies"],
        "share": [round(x, 5) for x in e3["data"]["share"]],
        "key_frequencies": e3["summary"]["key_frequencies"],
        "top": e3["summary"]["top_frequencies"][:5],
        "hypothesis": hyp["text"],
        "result": res["text"],
        "revised_example": {"text": revised[0]["text"], "target": revised[0]["target"]} if revised else None,
        "pseudocode": report["pseudocode"],
        "verification": {
            "n": v1["n_heldout"], "agreement": v1["checks"]["agreement"]["agreement_with_model"],
            "subspace_dims": sum(sub["dims"]), "total_dims": sub["of_total_dims"][0],
            "acc_only_subspace": sub["only_subspace_kept"]["accuracy"],
            "acc_subspace_removed": sub["subspace_removed"]["accuracy"],
            "acc_random_removed": sub["random_subspace_removed"]["accuracy"],
        },
        "experiments": sum(1 for _ in (d / "experiments").glob("E*.json")),
        "wall_clock_seconds": session["wall_clock_seconds"],
        "training": {k: training["meta"][k] for k in ("epochs", "train_seconds", "final_test_acc", "device")},
        "tag": e3["provenance"]["tag"],
    }


def scorecard() -> dict:
    p = ROOT / "benchmarks" / "scorecard.json"
    if not p.exists():
        return {"available": False}
    card = json.loads(p.read_text(encoding="utf-8"))["benchmarks"]
    rows = []
    for name, b in card.items():
        rows.append({"name": name, "title": b["title"], "found": b["rubric"]["found"], "partial": b["rubric"]["partial"],
                     "missed": b["rubric"]["missed"], "total": b["rubric"]["total"], "experiments": b["experiments"],
                     "minutes": round(b["wall_clock_seconds"] / 60, 1), "status": b["final_verification"]["status"]})
    ioi = card["ioi"]["mechanical"]
    return {"available": True, "rows": rows,
            "ioi": {"found": ioi["published_heads_found"], "published": ioi["published_heads"],
                    "by_class": {k: {"found": v["found"], "missed": v["missed"]} for k, v in ioi["by_class"].items()}}}


def optional(path: Path) -> dict:
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {"available": False}


def demo27b() -> dict:
    """The demo summary, with labels shortened and numbers rounded so they fit on a video frame."""
    d = optional(ROOT / "demos" / "qwen27b-language" / "summary.json")
    if d["available"]:
        d["model"] = "Qwen3.8-27B abliterated, Q4_K_M GGUF"
        st = d["steering"]
        st["metric"] = [round(x, 1) for x in st["metric"]]
        st["metric_name"] = st.get("metric_short", "metric")
        d["gpus"] = [g.replace(" used)", ")") for g in d["gpus"]]
        d["tag"] = "screening (gguf quantized)"
    return d


def ladder() -> dict:
    d = optional(ROOT / "ladder" / "summary.json")
    if d["available"]:
        names = {"gpt2-local": "GPT-2 small", "gpt2-ndif": "GPT-2 small", "qwen27b-gguf": "Qwen3.8-27B GGUF", "qwen27b-ndif": "Qwen3.8-27B official"}
        tags = {"gpt2-local": "full precision", "gpt2-ndif": "full precision", "qwen27b-gguf": "screening", "qwen27b-ndif": "full precision"}
        for r in d["rungs"]:
            v = r["verdicts"]
            ok = sum(x == "supported" for x in v.values())
            tested = sum(x in ("supported", "rejected") for x in v.values())
            r["model"], r["tag"] = names.get(r["rung"], r["model"]), tags.get(r["rung"], r["tag"])
            r["status"] = f"{ok} of {tested} hold"
            r["key_result"] = r["key_result"].replace("; ", ". ")
    return d


def main() -> None:
    data = {
        "modadd": modadd(),
        "scorecard": scorecard(),
        "demo27b": demo27b(),
        "ladder": ladder(),
        "repo": "github.com/hesselb20-ctrl/glassbox",
        "install": "pip install -e . && claude mcp add glassbox -- python -m glassbox.mcp",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), {k: (v.get("available") if isinstance(v, dict) else v) for k, v in data.items()})


if __name__ == "__main__":
    main()
