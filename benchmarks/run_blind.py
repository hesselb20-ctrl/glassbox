"""Run one benchmark investigation blind.

A fresh headless Claude Code session (your subscription, no API key) gets the /investigate prompt,
the behaviour description below, and the glassbox MCP tools. Nothing else: no file access, no web,
no shell, no hints about the mechanism. Everything it learns it has to measure.

    python benchmarks/run_blind.py modadd
    python benchmarks/run_blind.py induction
    python benchmarks/run_blind.py ioi

Results land in benchmarks/<name>/ (notebook.jsonl, experiments/, report/, session.json).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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


def _span(nb_path: Path) -> float | None:
    """Seconds between the first and last notebook entry (covers every session of a resumed investigation)."""
    if not nb_path.exists():
        return None
    ts = [json.loads(l)["ts"] for l in nb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(ts) < 2:
        return None
    fmt = "%Y-%m-%dT%H:%M:%S"
    return time.mktime(time.strptime(ts[-1], fmt)) - time.mktime(time.strptime(ts[0], fmt))


def main(name: str, spec: dict | None = None, home: str = "benchmarks") -> int:
    """Run one investigation headless. ``spec`` defaults to the benchmark of that name; ``home`` is the results folder."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    b = spec or BENCHMARKS[name]
    command = (ROOT / ".claude" / "commands" / "investigate.md").read_text(encoding="utf-8")
    command = command.split("---", 2)[2]                       # drop the frontmatter
    args = f'{b["backend"]} {b["model"]} "{b["behavior"]}" budget={b["budget"]} investigation={name}'
    prompt = command.replace("$ARGUMENTS", args)

    cfg = {"mcpServers": {"glassbox": {"command": sys.executable, "args": ["-m", "glassbox.mcp"],
                                       "env": {"GLASSBOX_HOME": str(ROOT / home)}}}}
    cfg_path = Path(tempfile.mkdtemp()) / "mcp.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    cmd = [shutil.which("claude"), "-p", "--output-format", "json",
           "--mcp-config", str(cfg_path), "--strict-mcp-config",
           "--tools", "",                                       # no built-in tools: no files, no shell, no web
           "--allowedTools", "mcp__glassbox",
           "--setting-sources", "project",                      # no personal memory or instructions
           "--no-session-persistence"]
    out_dir = ROOT / home / name
    nb_path = out_dir / "notebook.jsonl"
    resumed = nb_path.exists() and nb_path.stat().st_size > 0      # an earlier session was interrupted: continue it
    t0 = time.time()
    env = {**os.environ, "MCP_TOOL_TIMEOUT": "3600000"}       # big local models: one experiment can take minutes
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env)
    wall = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = {"raw_stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]}
    root_variants = [str(ROOT) + os.sep, str(ROOT).replace(os.sep, "/") + "/", str(ROOT)]

    def clean(text):
        for v in root_variants:                                  # keep machine paths out of committed records
            text = text.replace(v, "") if isinstance(text, str) else text
        return text

    res["result"] = clean(res.get("result"))
    session = {
        "benchmark": name, "prompt_arguments": args, "wall_clock_seconds": round(wall, 1),
        "resumed_from_interrupted_session": resumed, "notebook_span_seconds": _span(nb_path),
        "scientist": "Claude Code (headless, subscription), MCP tools only",
        "model_usage": sorted((res.get("modelUsage") or {}).keys()),
        "turns": res.get("num_turns"), "is_error": res.get("is_error"), "exit_code": proc.returncode,
        "final_message": res.get("result"),
    }
    if "raw_stdout" in res:
        session.update(res)
    (out_dir / "session.json").write_text(json.dumps(session, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    print(json.dumps({k: v for k, v in session.items() if k != "final_message"}, indent=1))
    print((session.get("final_message") or "")[:3000])
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
