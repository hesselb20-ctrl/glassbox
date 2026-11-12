"""Command line access to the same tools the MCP server exposes.

    glassbox tool list_models
    glassbox tool load_model '{"backend": "local-full", "model": "gpt2", "investigation": "ioi"}'
    glassbox tool run_experiment '{"primitive": "baseline", "params": {"dataset": {"task": "ioi"}}}' -i ioi
    glassbox train-modadd benchmarks/modadd/model

Each call is a fresh process: the model named in the investigation's state.json is reloaded automatically.
"""
from __future__ import annotations

import argparse
import json
import sys

from .session import Session

TOOLS = ("list_models", "load_model", "backend_capabilities", "run_experiment", "get_plot", "notebook_append",
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
    a = ap.parse_args(argv)

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
