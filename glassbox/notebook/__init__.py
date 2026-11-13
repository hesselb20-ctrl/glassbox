"""The lab notebook: an append-only, hash-chained ``notebook.jsonl`` per investigation.

Hypotheses, pre-registrations, experiments, results, decisions. Nothing is ever edited in place; a
hypothesis changes status by appending an update that points at it. Because the whole state of an
investigation lives in this file (plus the experiment records next to it), a long investigation can
be stopped and resumed in a later session.

Hash chain. Every entry stores ``prev`` (the hash of the entry before it) and
``hash = sha256(prev + canonical JSON of the entry)``. Editing, deleting or reordering any earlier
entry breaks every hash after it, and :meth:`Notebook.verify` reports the first entry where that
happens. This makes after-the-fact edits detectable; it does not stop someone who rewrites the whole
file and recomputes every hash. It is a tamper-evidence seal, not a signature.

Pre-registration. A ``prereg`` entry fixes, before an experiment runs, the hypothesis, the exact
experiment, a quantitative prediction and the competing hypotheses the experiment separates. When the
experiment runs, its result is scored against the prediction by :func:`score_prediction`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

KINDS = ("hypothesis", "prereg", "experiment", "result", "decision", "verification", "note", "report")
STATUSES = ("open", "supported", "rejected", "revised")
OPS = (">=", "<=", ">", "<", "between", "outside", "==")
GENESIS = "0" * 64


def canonical(entry: dict) -> str:
    """Canonical JSON of an entry without its own hash: sorted keys, no whitespace."""
    return json.dumps({k: v for k, v in entry.items() if k != "hash"}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_hash(prev: str, entry: dict) -> str:
    return hashlib.sha256((prev + canonical(entry)).encode("utf-8")).hexdigest()


def _link(prev_entry: dict | None) -> str:
    """The value the next entry must carry in ``prev``."""
    if prev_entry is None:
        return GENESIS
    # entries written before the chain existed have no hash: they are anchored by the hash of their content
    return prev_entry.get("hash") or hashlib.sha256(canonical(prev_entry).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #

def resolve(record: dict, path: str) -> Any:
    """Follow a path such as ``summary.fraction_of_clean``, ``data.effect[9][6]`` or
    ``summary.top[component=9.9].effect`` into an experiment record. Raises KeyError if it does not exist."""
    cur: Any = record
    for part in re.findall(r"[^.\[\]]+|\[[^\]]+\]", path):
        if part.startswith("["):
            inner = part[1:-1]
            if "=" in inner:
                key, val = inner.split("=", 1)
                hit = next((d for d in cur if isinstance(d, dict) and str(d.get(key)) == val), None)
                if hit is None:
                    raise KeyError(f"no element with {key}={val} at '{part}' in '{path}'")
                cur = hit
            else:
                cur = cur[int(inner)]
        else:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"'{part}' not found while resolving '{path}'")
            cur = cur[part]
    return cur


def validate_prediction(pred: dict) -> None:
    if not isinstance(pred, dict) or "metric" not in pred or "op" not in pred or "threshold" not in pred:
        raise ValueError('a prediction needs "metric" (a path into the result, e.g. "summary.fraction_of_clean"), '
                         f'"op" (one of {OPS}) and "threshold" (a number, or [low, high] for between/outside)')
    if pred["op"] not in OPS:
        raise ValueError(f'prediction op must be one of {OPS}')
    th = pred["threshold"]
    if pred["op"] in ("between", "outside"):
        if not (isinstance(th, (list, tuple)) and len(th) == 2 and all(isinstance(x, (int, float)) for x in th) and th[0] <= th[1]):
            raise ValueError("for 'between' / 'outside' the threshold must be [low, high]")
    elif not isinstance(th, (int, float)) or isinstance(th, bool):
        raise ValueError("prediction threshold must be a number")
    if not str(pred["metric"]).startswith(("summary.", "data.")):
        raise ValueError('prediction metric must be a path starting with "summary." or "data."')


def score_prediction(pred: dict, record: dict) -> dict:
    """Score an experiment record against a pre-registered prediction: confirmed / refuted / inconclusive.

    Inconclusive means the predicted quantity could not be read (missing, not a number) or fell within the
    optional ``margin`` of the threshold.
    """
    out = {"metric": pred["metric"], "op": pred["op"], "threshold": pred["threshold"], "statement": pred.get("statement")}
    try:
        v = resolve(record, pred["metric"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return {**out, "verdict": "inconclusive", "observed": None, "why": f"metric not found in the result: {e}"}
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return {**out, "verdict": "inconclusive", "observed": v if isinstance(v, (int, float, str, type(None))) else str(type(v)),
                "why": "metric is not a number"}
    op, th, margin = pred["op"], pred["threshold"], float(pred.get("margin", 0) or 0)
    edges = list(th) if op in ("between", "outside") else [th]
    if margin and any(abs(v - e) < margin for e in edges):
        return {**out, "verdict": "inconclusive", "observed": v, "why": f"within the margin of {margin} around the threshold"}
    ok = {">=": lambda: v >= th, "<=": lambda: v <= th, ">": lambda: v > th, "<": lambda: v < th,
          "between": lambda: th[0] <= v <= th[1], "outside": lambda: v < th[0] or v > th[1],
          "==": lambda: abs(v - th) <= max(margin, 1e-9)}[op]()
    return {**out, "verdict": "confirmed" if ok else "refuted", "observed": v}


# --------------------------------------------------------------------------- #
# notebook
# --------------------------------------------------------------------------- #

class Notebook:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _write(self, entry: dict, existing: list[dict]) -> dict:
        entry["prev"] = _link(existing[-1] if existing else None)
        entry["hash"] = entry_hash(entry["prev"], entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def append(self, kind: str, text: str, refs: list[str] | None = None, status: str | None = None,
               target: str | None = None, data: dict | None = None) -> dict:
        """Add an entry.

        ``refs`` are experiment ids (E0007) or notebook ids (N0003) this entry rests on.
        ``target`` + ``status`` on any entry updates the status of an earlier hypothesis.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if kind == "prereg":
            raise ValueError("use Notebook.prereg() to pre-register an experiment")
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
        return self._write(entry, existing)

    def prereg(self, hypothesis: str, experiment_spec: dict, prediction: dict, alternatives: list[str] | None = None) -> dict:
        """Pre-register one experiment: what will be run, what is predicted, and what else could be true."""
        if not hypothesis or not str(hypothesis).strip():
            raise ValueError("a pre-registration needs a hypothesis")
        if not isinstance(experiment_spec, dict) or "primitive" not in experiment_spec:
            raise ValueError('experiment_spec must be {"primitive": ..., "params": {...}}')
        validate_prediction(prediction)
        if not alternatives:
            raise ValueError("list at least one alternative: a competing hypothesis this experiment could favour instead")
        existing = self.entries()
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry = {
            "id": f"N{len(existing) + 1:04d}", "ts": now, "kind": "prereg", "text": hypothesis, "refs": [],
            "hypothesis": hypothesis,
            "experiment_spec": {"primitive": experiment_spec["primitive"], "params": experiment_spec.get("params") or {}},
            "prediction": dict(prediction), "alternatives": list(alternatives), "created_at": now,
        }
        return self._write(entry, existing)

    # -- integrity ---------------------------------------------------------- #
    def verify(self) -> dict:
        """Recompute the hash chain. Reports the first tampered, missing or reordered entry."""
        entries = self.entries()
        chained = legacy = 0
        problem = None
        prev: dict | None = None
        for i, e in enumerate(entries):
            expect_id = f"N{i + 1:04d}"
            if e.get("id") != expect_id:
                problem = {"index": i, "id": e.get("id"), "reason": f"expected id {expect_id} at this position: an entry was removed, inserted or reordered"}
                break
            if "hash" not in e:
                if chained:
                    problem = {"index": i, "id": e["id"], "reason": "entry has no hash although earlier entries are chained"}
                    break
                legacy += 1
            else:
                if e.get("prev") != _link(prev):
                    culprit = prev["id"] if prev else e["id"]
                    problem = {"index": i, "id": e["id"], "reason": f"does not link to the entry before it: {culprit} was changed, removed or reordered"}
                    break
                if entry_hash(e["prev"], e) != e["hash"]:
                    problem = {"index": i, "id": e["id"], "reason": "content does not match its hash: this entry was edited"}
                    break
                chained += 1
            prev = e
        return {"ok": problem is None, "entries": len(entries), "chained": chained,
                "legacy_unchained_prefix": legacy, "first_problem": problem,
                "head": entries[-1].get("hash") if entries and problem is None else None,
                "note": "tamper-evident, not tamper-proof: a full rewrite with recomputed hashes is not detectable from the file alone"}

    # -- views -------------------------------------------------------------- #
    def view(self) -> list[dict]:
        """Entries with each hypothesis carrying its latest status and the history of how it got there."""
        entries = self.entries()
        by_id = {e["id"]: {k: v for k, v in e.items() if k not in ("hash", "prev")} for e in entries}
        for e in entries:
            t = e.get("target")
            if t and e.get("status") and t in by_id and by_id[t]["kind"] == "hypothesis":
                by_id[t].setdefault("history", []).append({"by": e["id"], "status": e["status"], "why": e["text"]})
                by_id[t]["status"] = e["status"]
            p = (e.get("data") or {}).get("prereg_id")
            if e["kind"] == "experiment" and p in by_id:
                by_id[p]["run"] = {"experiment": e["refs"][0] if e["refs"] else None, **(e["data"].get("score") or {})}
        return list(by_id.values())

    def prediction_record(self) -> dict:
        """Counts over pre-registrations: how many, how many ran, and how they scored."""
        v = self.view()
        pre = [e for e in v if e["kind"] == "prereg"]
        ran = [e for e in pre if e.get("run")]
        count = lambda verdict: sum(e["run"].get("verdict") == verdict for e in ran)  # noqa: E731
        decided = count("confirmed") + count("refuted")
        return {
            "preregistered": len(pre), "run": len(ran), "not_run": [e["id"] for e in pre if not e.get("run")],
            "confirmed": count("confirmed"), "refuted": count("refuted"), "inconclusive": count("inconclusive"),
            "hit_rate": round(count("confirmed") / decided, 3) if decided else None,
            "refuted_predictions": [{"prereg": e["id"], "experiment": e["run"].get("experiment"), "hypothesis": e["hypothesis"],
                                     "prediction": e["prediction"], "observed": e["run"].get("observed")}
                                    for e in ran if e["run"].get("verdict") == "refuted"],
        }

    def summary(self) -> dict:
        v = self.view()
        hyps = [e for e in v if e["kind"] == "hypothesis"]
        rec = self.prediction_record()
        return {
            "entries": len(v),
            "experiments": sum(e["kind"] == "experiment" for e in v),
            "hypotheses": {s: [f"{h['id']}: {h['text'][:140]}" for h in hyps if h.get("status") == s] for s in STATUSES},
            "predictions": {k: rec[k] for k in ("preregistered", "run", "confirmed", "refuted", "inconclusive", "hit_rate", "not_run")},
            "last_decision": next((e["text"] for e in reversed(v) if e["kind"] == "decision"), None),
            "verifications": [{"id": e["id"], "status": e.get("data", {}).get("status"), "text": e["text"][:140]}
                              for e in v if e["kind"] == "verification"],
        }
