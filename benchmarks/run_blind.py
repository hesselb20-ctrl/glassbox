"""Run one benchmark investigation blind.

A fresh headless Claude Code session (your subscription, no API key) gets the /investigate prompt,
the behaviour description below, and the glassbox MCP tools. Nothing else: no file access, no web,
no shell, no hints about the mechanism. Everything it learns it has to measure.

    python benchmarks/run_blind.py modadd
    python benchmarks/run_blind.py induction
    python benchmarks/run_blind.py ioi

Results land in benchmarks/<name>/ (notebook.jsonl, experiments/, report/, session.json).
The runner itself lives in glassbox/headless.py; `glassbox null-run <name> --home benchmarks` repeats a run on null controls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glassbox import headless  # noqa: E402

BENCHMARKS = {
    "modadd": dict(
        backend="local-full", model="checkpoint:benchmarks/modadd/model/model.pt", budget=40,
        behavior="This 1-layer transformer was trained from scratch. Its input is three tokens [a, b, =] with a and b "
                 "integers from 0 to 112 (the '=' token has id 113), and at the last position it outputs "
                 "(a + b) mod 113. What algorithm did it learn to compute the answer? "
                 "The dataset task is 'modadd' (n='all' gives the full 113x113 grid).",
    ),
    "induction": dict(
        backend="local-full", model="attn-only-2l", budget=40,
        behavior="When this 2-layer attention-only model is shown a sequence of random tokens that then repeats, it "
                 "predicts the next token of the repeat correctly. How does it do that? The dataset task is 'induction'.",
    ),
    "ioi": dict(
        backend="local-full", model="gpt2", budget=60,
        behavior="Given a sentence like 'When Mary and John went to the store, John gave a drink to', GPT-2 small "
                 "continues with ' Mary' rather than ' John'. How does it work out which name to output? "
                 "The dataset task is 'ioi'.",
    ),
}


def main(name: str, spec: dict | None = None, home: str = "benchmarks") -> int:
    """Run one investigation headless. ``spec`` defaults to the benchmark of that name; ``home`` is the results folder."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    b = spec or BENCHMARKS[name]
    prompt, args = headless.build_prompt(b["backend"], b["model"], b["behavior"], b["budget"], name)
    session = headless.run(prompt, ROOT / home, name, prompt_args=args)
    print(json.dumps({k: v for k, v in session.items() if k != "final_message"}, indent=1))
    print((session.get("final_message") or "")[:3000])
    return session["exit_code"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
