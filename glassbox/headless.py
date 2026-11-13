"""Run an investigation with a headless Claude Code session as the scientist.

The session gets the /investigate prompt, the behaviour description and the glassbox MCP tools. Nothing
else: no file access, no shell, no web. It runs on the Claude Code subscription; no API key is used.
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
COMMAND = ROOT / ".claude" / "commands" / "investigate.md"


def build_prompt(backend: str, model: str, behavior: str, budget: int, investigation: str) -> tuple[str, str]:
    command = COMMAND.read_text(encoding="utf-8").split("---", 2)[2]          # drop the frontmatter
    args = f'{backend} {model} "{behavior}" budget={budget} investigation={investigation}'
    return command.replace("$ARGUMENTS", args), args


def notebook_span(nb_path: Path) -> float | None:
    """Seconds between the first and last notebook entry (covers every session of a resumed investigation)."""
    if not nb_path.exists():
        return None
    ts = [json.loads(l)["ts"] for l in nb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(ts) < 2:
        return None
    fmt = "%Y-%m-%dT%H:%M:%S"
    return time.mktime(time.strptime(ts[-1], fmt)) - time.mktime(time.strptime(ts[0], fmt))


def run(prompt: str, home: Path, investigation: str, label: str = "", env_extra: dict | None = None,
        prompt_args: str = "", cwd: Path | None = None) -> dict:
    """Run one headless investigation and write ``session.json`` and ``prompt.md`` into its folder."""
    home = Path(home).resolve()
    out_dir = home / investigation
    nb_path = out_dir / "notebook.jsonl"
    resumed = nb_path.exists() and nb_path.stat().st_size > 0
    cfg = {"mcpServers": {"glassbox": {"command": sys.executable, "args": ["-m", "glassbox.mcp"],
                                       "env": {"GLASSBOX_HOME": str(home), **(env_extra or {})}}}}
    cfg_path = Path(tempfile.mkdtemp()) / "mcp.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    cmd = [shutil.which("claude"), "-p", "--output-format", "json",
           "--mcp-config", str(cfg_path), "--strict-mcp-config",
           "--tools", "",                                       # no built-in tools: no files, no shell, no web
           "--allowedTools", "mcp__glassbox",
           "--setting-sources", "project",
           "--no-session-persistence"]
    t0 = time.time()
    env = {**os.environ, "MCP_TOOL_TIMEOUT": "3600000"}
    env.pop("GLASSBOX_NULL", None)
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", cwd=cwd or ROOT, env=env)
    wall = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = json.loads(proc.stdout)
    except json.JSONDecodeError:
        res = {"raw_stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]}

    def clean(text):
        for v in (str(ROOT) + os.sep, str(ROOT).replace(os.sep, "/") + "/", str(ROOT), str(Path.home())):
            text = text.replace(v, "") if isinstance(text, str) else text
        return text

    session = {
        "benchmark": label or investigation, "prompt_arguments": prompt_args, "wall_clock_seconds": round(wall, 1),
        "resumed_from_interrupted_session": resumed, "notebook_span_seconds": notebook_span(nb_path),
        "scientist": "Claude Code (headless, subscription), MCP tools only",
        "model_usage": sorted((res.get("modelUsage") or {}).keys()),
        "turns": res.get("num_turns"), "is_error": res.get("is_error"), "exit_code": proc.returncode,
        "final_message": clean(res.get("result")),
    }
    if "raw_stdout" in res:
        session.update({k: clean(v) for k, v in res.items()})
    (out_dir / "session.json").write_text(json.dumps(session, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    return session
