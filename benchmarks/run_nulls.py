"""Rerun a benchmark investigation on the three null controls and report the false discovery rate.

    python benchmarks/run_nulls.py ioi

Same behaviour description, same budget, same scientist instructions as the real run; the null is applied
inside the MCP server and the scientist is not told. Results: benchmarks/<name>__null-<kind>/ and
benchmarks/<name>/null_runs.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from glassbox import headless, nulls  # noqa: E402
from run_blind import BENCHMARKS  # noqa: E402


def main(name: str, seed: int = 0) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    b = BENCHMARKS[name]
    home = ROOT / "benchmarks"
    for kind in nulls.KINDS:
        run = f"{name}__null-{kind}"
        if (home / run / "session.json").exists():
            continue
        prompt, args = headless.build_prompt(b["backend"], b["model"], b["behavior"], b["budget"], run)
        s = headless.run(prompt, home, run, prompt_args=args, env_extra={"GLASSBOX_NULL": f"{kind}:{seed}"})
        print(run, s["wall_clock_seconds"], "s", s["turns"], "turns")
    out = nulls.summarize_null_runs(home, name)
    (home / name / "null_runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "false_discoveries", "false_discovery_rate")}))
    for r in out["null_runs"]:
        print(" ", r["run"], r["experiments"], "experiments", r["verifications"], r["report_status"], "->", r["discovered_a_circuit"])


if __name__ == "__main__":
    main(sys.argv[1])
