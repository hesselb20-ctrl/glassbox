"""Command line access to the same tools the MCP server exposes.

    glassbox tool list_models
    glassbox tool load_model '{"backend": "local-full", "model": "gpt2", "investigation": "ioi"}'
    glassbox tool run_experiment '{"primitive": "baseline", "params": {"dataset": {"task": "ioi"}}}' -i ioi
    glassbox train-modadd benchmarks/modadd/model
    glassbox notebook verify ioi            # recompute the hash chain of an investigation's notebook

Each call is a fresh process: the model named in the investigation's state.json is reloaded automatically.
"""
from __future__ import annotations

import argparse
import json
import sys

from .session import Session

TOOLS = ("list_models", "load_model", "backend_capabilities", "preregister", "run_experiment", "get_plot", "notebook_append",
         "notebook_read", "verify_hypothesis", "write_report")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glassbox", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tool", help="call one lab tool")
    t.add_argument("name", choices=TOOLS)
    t.add_argument("args", nargs="?", default="{}", help="JSON object of arguments, or @file.json")
    t.add_argument("-i", "--investigation", default=None)
    tr = sub.add_parser("train-modadd", help="train the modular addition toy model")
    tr.add_argument("out_dir")
    tr.add_argument("--p", type=int, default=113)
    tr.add_argument("--seed", type=int, default=0)
    nb = sub.add_parser("notebook", help="notebook utilities")
    nb.add_argument("action", choices=["verify"])
    nb.add_argument("investigation", help="investigation name under GLASSBOX_HOME, or a path to a notebook.jsonl")
    nr = sub.add_parser("null-run", help="rerun an investigation on null controls and report the false discovery rate")
    nr.add_argument("investigation")
    nr.add_argument("--home", default=None, help="folder holding the investigation (default: GLASSBOX_HOME)")
    nr.add_argument("--kinds", default="random-init,weight-shuffled,task-shuffled")
    nr.add_argument("--seed", type=int, default=0)
    nr.add_argument("--summary-only", action="store_true", help="do not run anything, just collect existing null runs")
    be = sub.add_parser("bench", help="glassbox-bench: planted circuits with sealed answer keys")
    be.add_argument("action", choices=["generate", "run", "grade"])
    be.add_argument("seed_pos", nargs="?", type=int, default=None, help="seed (positional form, e.g. `glassbox bench grade 3`)")
    be.add_argument("--seed", type=int, default=None)
    be.add_argument("--tasks", default=None, help="comma-separated task ids for `run` (default: all)")
    be.add_argument("--budget", type=int, default=25)
    be.add_argument("--reveal", action="store_true", help="grade: also copy the answer keys of this seed into the repo")
    a = ap.parse_args(argv)

    if a.cmd == "bench":
        seed = a.seed if a.seed is not None else a.seed_pos
        if seed is None:
            print("give a seed: glassbox bench generate --seed 1")
            return 1
        if a.action == "generate":
            from .bench.generate import generate

            print(json.dumps(generate(seed), indent=1))
        elif a.action == "run":
            from . import bench

            print(json.dumps(bench.run(seed, a.tasks.split(",") if a.tasks else None, a.budget), indent=1))
        else:
            from .bench.grade import grade

            card = grade(seed, reveal=a.reveal)
            print(json.dumps(card["summary"], indent=1))
        return 0

    if a.cmd == "null-run":
        from pathlib import Path

        from . import headless, nulls
        from .session import home as default_home

        home = Path(a.home).resolve() if a.home else default_home()
        src = home / a.investigation
        if not (src / "prompt.md").exists():
            print(f"{src} has no prompt.md: null-run repeats a headless investigation and needs its original prompt")
            return 1
        prompt = (src / "prompt.md").read_text(encoding="utf-8")
        for kind in ([] if a.summary_only else [k for k in a.kinds.split(",") if k]):
            name = f"{a.investigation}__null-{kind}"
            if (home / name / "session.json").exists():
                print(f"{name}: already run, skipping")
                continue
            # same prompt, same budget, same instructions; only the investigation name differs, and the null is
            # applied inside the MCP server so the scientist is not told
            p = prompt.replace(f"investigation={a.investigation}", f"investigation={name}")
            s = headless.run(p, home, name, label=name, env_extra={"GLASSBOX_NULL": f"{kind}:{a.seed}"})
            print(f"{name}: {s['wall_clock_seconds']} s, {s['turns']} turns")
        out = nulls.summarize_null_runs(home, a.investigation)
        (src / "null_runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
        return 0

    if a.cmd == "notebook":
        from pathlib import Path

        from .notebook import Notebook

        path = Path(a.investigation)
        res = Notebook(path).verify() if path.suffix == ".jsonl" and path.exists() else Session().notebook_verify(a.investigation)
        print(json.dumps(res, indent=1))
        if res["ok"]:
            print(f"OK: {res['chained']} chained entries" + (f", {res['legacy_unchained_prefix']} older entries from before the chain" if res["legacy_unchained_prefix"] else ""))
        else:
            fp = res["first_problem"]
            print(f"TAMPERED: first problem at entry {fp['id']} (position {fp['index'] + 1}): {fp['reason']}")
        return 0 if res["ok"] else 2

    if a.cmd == "train-modadd":
        from .toy import train_modadd

        print(json.dumps(train_modadd(a.out_dir, p=a.p, seed=a.seed), indent=1))
        return 0

    raw = open(a.args[1:], encoding="utf-8").read() if a.args.startswith("@") else a.args
    kwargs = json.loads(raw)
    s = Session()
    if a.investigation and a.name != "load_model":
        s._open(a.investigation)
    elif a.investigation:
        kwargs.setdefault("investigation", a.investigation)
    try:
        if a.name in ("verify_hypothesis", "write_report") and "spec" not in kwargs and "report" not in kwargs:
            kwargs = {"spec" if a.name == "verify_hypothesis" else "report": kwargs}
        out = getattr(s, a.name)(**kwargs)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": type(e).__name__, "message": str(e)}, indent=1))
        return 1
    print(out if isinstance(out, str) else json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
