"""Remove remarks about the repo owner's personal setup from the closing messages of headless runs.

The headless scientist loads the owner's personal Claude Code instructions (see Limitations in the README), and its
closing message sometimes says that it could not follow a note-keeping habit from them. Those sentences say nothing
about the investigation and name private folders, so they are replaced by a marker before the records are committed.
Notebooks, experiment records and reports are not touched.

    python benchmarks/scrub_sessions.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = re.compile(r"vault|standing instruction|method note", re.I)
MARK = "[A sentence about the owner's personal note-keeping instructions was removed.]"


def scrub(text: str) -> str:
    parts = re.split(r"(?<=[.!?])(\s+)|(\n+)", text)
    out, dropped = [], False
    for p in parts:
        if p is None:
            continue
        if PRIVATE.search(p):
            if not dropped:
                out.append(MARK)
            dropped = True
        else:
            out.append(p)
    return "".join(out)


def main() -> None:
    n = 0
    for f in sorted(ROOT.glob("**/session.json")):
        if "node_modules" in f.parts:
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        changed = False
        for k, v in s.items():
            if isinstance(v, str) and PRIVATE.search(v):
                s[k], changed = scrub(v), True
        if changed:
            f.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
            n += 1
    print("scrubbed", n, "session records")


if __name__ == "__main__":
    main()
