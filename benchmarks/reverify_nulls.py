"""Re-check the findings of the three original benchmarks against null distributions.

The original blind runs predate null-calibrated verification. This script takes the final verification spec of each
run (the same circuit, the same code, the same held-out seed) and runs it again with the null comparison switched on,
keeping every null sample so the distributions can be plotted. It is a re-verification of an existing claim, not a
new blind run.

    python benchmarks/reverify_nulls.py            # writes benchmarks/<name>/null_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import glassbox.verify as V  # noqa: E402
from glassbox.backends import get_backend  # noqa: E402
from glassbox.lab import Lab  # noqa: E402
from glassbox.lab.core import _jsonable  # noqa: E402

SAMPLES: dict[str, list[float]] = {}
_orig = V.null_test


def recording(name, effect, null, kinds, percentile, what):
    SAMPLES[name] = [float(x) for x in null]
    return _orig(name, effect, null, kinds, percentile, what)


V.null_test = recording


def main() -> None:
    for name in ("modadd", "induction", "ioi"):
        d = ROOT / "benchmarks" / name
        state = json.loads((d / "state.json").read_text(encoding="utf-8"))
        last = sorted((d / "experiments").glob("V*.json"))[-1]
        spec = json.loads(last.read_text(encoding="utf-8"))["spec"]
        b = get_backend(state["backend"])
        b.load(state["model"])
        SAMPLES.clear()
        res, fig = V.verify(Lab(b, d), {**spec, "null_samples": 60, "null_model_seeds": 3})
        import matplotlib.pyplot as plt

        plt.close(fig)
        out = {"benchmark": name, "reverified_spec_from": last.name, "status": res["status"], "beats_null_controls": res["beats_null_controls"],
               "metric_clean": res["metric_clean"], "metric_unit": res["metric_unit"], "null_controls": res["null_controls"],
               "null_samples": dict(SAMPLES), "provenance": res["provenance"],
               "note": "same circuit, code and held-out data as the run's final verification; only the null comparison is new"}
        (d / "null_calibration.json").write_text(json.dumps(_jsonable(out), indent=1), encoding="utf-8")
        print(name, res["status"], "beats nulls:", res["beats_null_controls"])
        for n in res["null_controls"]:
            print("   ", n["finding"], "effect", round(n["effect"], 3), "null mean", round(n.get("null_mean", float("nan")), 3),
                  "p95", round(n.get("null_p95", float("nan")), 3), "n", n["n_null"], "pass", n["pass"])
        b.unload()


if __name__ == "__main__":
    main()
