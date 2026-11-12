"""The lab notebook: an append-only ``notebook.jsonl`` per investigation.

Hypotheses, experiments, results, decisions. Nothing is ever edited in place; a hypothesis
changes status by appending an update that points at it. Because the whole state of an
investigation lives in this file (plus the experiment records next to it), a long
investigation can be stopped and resumed in a later session.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

KINDS = ("hypothesis", "experiment", "result", "decision", "verification", "note", "report")
STATUSES = ("open", "supported", "rejected", "revised")


class Notebook:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, kind: str, text: str, refs: list[str] | None = None, status: str | None = None,
               target: str | None = None, data: dict | None = None) -> dict:
        """Add an entry.

        ``refs`` are experiment ids (E0007) or notebook ids (N0003) this entry rests on.
        ``target`` + ``status`` on any entry updates the status of an earlier hypothesis.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if status is not None and status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        existing = self.entries()
        ids = {e["id"] for e in existing}
        if target is not None and target not in ids:
            raise KeyError(f"target '{target}' is not a notebook entry")
        entry: dict[str, Any] = {
            "id": f"N{len(existing) + 1:04d}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "text": text,
            "refs": list(refs or []),
        }
        if kind == "hypothesis" and status is None and target is None:
            status = "open"
        if status:
            entry["status"] = status
        if target:
            entry["target"] = target
        if data:
            entry["data"] = data
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def view(self) -> list[dict]:
        """Entries with each hypothesis carrying its latest status and the history of how it got there."""
        entries = self.entries()
        by_id = {e["id"]: dict(e) for e in entries}
        for e in entries:
            t = e.get("target")
            if t and e.get("status") and t in by_id and by_id[t]["kind"] == "hypothesis":
                by_id[t].setdefault("history", []).append({"by": e["id"], "status": e["status"], "why": e["text"]})
                by_id[t]["status"] = e["status"]
        return list(by_id.values())

    def summary(self) -> dict:
        v = self.view()
        hyps = [e for e in v if e["kind"] == "hypothesis"]
        return {
            "entries": len(v),
            "experiments": sum(e["kind"] == "experiment" for e in v),
            "hypotheses": {s: [f"{h['id']}: {h['text'][:140]}" for h in hyps if h.get("status") == s] for s in STATUSES},
            "last_decision": next((e["text"] for e in reversed(v) if e["kind"] == "decision"), None),
            "verifications": [{"id": e["id"], "status": e.get("data", {}).get("status"), "text": e["text"][:140]}
                              for e in v if e["kind"] == "verification"],
        }
