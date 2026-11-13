"""The lab session: the nine tools the scientist drives, as plain Python.

The MCP server, the CLI and the optional API agent are all thin wrappers over this class,
so every interface runs exactly the same code and writes to the same notebook.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .backends import BACKENDS, capability_table, get_backend
from .backends.base import PRIMITIVES, PreflightError
from .backends.preflight import BUSY_PORT, gpu_report, port_in_use
from .notebook import Notebook

DATA_LIMIT = 6000  # characters of raw 'data' returned inline; the full record is always on disk


def home() -> Path:
    return Path(os.environ.get("GLASSBOX_HOME", "glassbox_runs")).resolve()


class Session:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root else home()
        self.investigation: str | None = None
        self.backend = None
        self.lab = None
        self.notebook: Notebook | None = None

    # -- investigation plumbing --------------------------------------------- #
    def _dir(self, name: str | None = None) -> Path:
        return self.root / (name or self.investigation or "default")

    def _state_path(self) -> Path:
        return self._dir() / "state.json"

    def _state(self) -> dict:
        p = self._state_path()
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _save_state(self, **kw: Any) -> None:
        st = {**self._state(), **kw}
        self._state_path().parent.mkdir(parents=True, exist_ok=True)
        self._state_path().write_text(json.dumps(st, indent=1), encoding="utf-8")

    def _open(self, investigation: str | None) -> None:
        """Switch to an investigation (creating it if new)."""
        name = investigation or self.investigation or "default"
        if name != self.investigation:
            self.investigation = name
            self.notebook = Notebook(self._dir() / "notebook.jsonl")
            self.lab = None
            self.backend = None
        elif self.notebook is None:
            self.notebook = Notebook(self._dir() / "notebook.jsonl")

    def _ensure_lab(self) -> None:
        """Lazily reload the model recorded in state.json, so a resumed session just works."""
        from .lab import Lab

        self._open(None)
        if self.lab is not None:
            return
        st = self._state()
        if not st.get("backend"):
            raise RuntimeError("no model loaded in this investigation. Call load_model first.")
        self.backend = get_backend(st["backend"])
        self.backend.load(st["model"], **st.get("options", {}))
        self.lab = Lab(self.backend, self._dir())

    # -- tools -------------------------------------------------------------- #
    def list_models(self, backend: str | None = None) -> dict:
        """Which models can be loaded right now, per backend, plus the state of the machine."""
        out: dict[str, Any] = {"backends": {}}
        for name in ([backend] if backend else list(BACKENDS)):
            try:
                out["backends"][name] = {"tag": BACKENDS[name]["tag"], "models": get_backend(name).list_models()}
            except Exception as e:  # a backend that cannot list still reports why
                out["backends"][name] = {"tag": BACKENDS[name]["tag"], "available": False, "reason": str(e)}
        if backend in (None, "ndif"):
            from .backends import ndif_status

            live = ndif_status.live_models()
            key = ndif_status.have_key()
            out["ndif"] = {
                "reachable": live["reachable"],
                "api_key_present": key,
                "api_key_recognised": ndif_status.key_recognised() if key else False,
                "ready_now": [m["id"] for m in live["models"] if m["status"] == "ready"],
                "note": None if key else ndif_status.MISSING_KEY,
            }
        busy = port_in_use()
        out["machine"] = {
            "gpus": gpu_report(),
            f"port_{BUSY_PORT}_in_use": busy,
            "note": (f"A local model server is listening on port {BUSY_PORT} and is probably using the GPUs. "
                     "Close it yourself before any local GPU run.") if busy else "GPUs are free for local runs.",
        }
        return out

    def load_model(self, backend: str, model: str, investigation: str | None = None, options: dict | None = None,
                   budget: int | None = None) -> dict:
        """Load a model on a backend and open (or resume) an investigation."""
        self._open(investigation)
        from .lab import Lab

        options = options or {}
        b = get_backend(backend)
        info = b.load(model, **options)
        self.backend, self.lab = b, Lab(b, self._dir())
        st = self._state()
        self._save_state(backend=backend, model=model, options=options,
                         budget=budget if budget is not None else st.get("budget"),
                         started=st.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.notebook.append("decision", f"Loaded {model} on {backend} ({b.tag}).", data={"info": info})
        return {"investigation": self.investigation, "info": info, "capabilities": self.backend_capabilities()["this_backend"],
                "notebook": self.notebook.summary(), "budget": self._budget()}

    def backend_capabilities(self) -> dict:
        """Which primitive runs on which backend."""
        table = capability_table()
        if self.backend is not None:
            feats = BACKENDS[self.backend.name]["features"]
            table["this_backend"] = {
                "name": self.backend.name, "tag": self.backend.tag,
                "supported": sorted(p for p, need in PRIMITIVES.items() if need <= feats),
                "unsupported": sorted(p for p, need in PRIMITIVES.items() if not need <= feats),
                "counts_as_verification": bool(BACKENDS[self.backend.name]["verifies"]),
            }
        return table

    def _budget(self) -> dict:
        st = self._state()
        used = len(list((self._dir() / "experiments").glob("E*.json"))) if (self._dir() / "experiments").exists() else 0
        b = st.get("budget")
        return {"experiments_used": used, "budget": b, "remaining": None if b is None else max(0, b - used)}

    def preregister(self, hypothesis: str, experiment_spec: dict, prediction: dict, alternatives: list[str] | None = None) -> dict:
        """Pre-register one experiment before it runs: hypothesis, exact experiment, quantitative prediction, alternatives."""
        from .lab import REGISTRY

        self._open(None)
        prim = (experiment_spec or {}).get("primitive")
        if prim not in REGISTRY:
            raise KeyError(f"unknown primitive '{prim}'. Known: {sorted(REGISTRY)}")
        entry = self.notebook.prereg(hypothesis, experiment_spec, prediction, alternatives)
        return {"prereg_id": entry["id"], "created_at": entry["created_at"], "prediction": entry["prediction"],
                "next": f"run_experiment(prereg_id='{entry['id']}') runs exactly this experiment and scores it against the prediction"}

    def _prereg_entry(self, prereg_id: str) -> dict:
        hit = next((e for e in self.notebook.entries() if e["id"] == prereg_id), None)
        if hit is None or hit["kind"] != "prereg":
            raise KeyError(f"'{prereg_id}' is not a pre-registration in this notebook")
        if any(e["kind"] == "experiment" and (e.get("data") or {}).get("prereg_id") == prereg_id for e in self.notebook.entries()):
            raise RuntimeError(f"pre-registration {prereg_id} has already been run. A prediction is tested once; "
                               "write a new pre-registration for a new experiment.")
        return hit

    def run_experiment(self, primitive: str | None = None, params: dict | None = None, note: str = "", detail: str = "auto",
                       prereg_id: str | None = None, exploratory: bool = False) -> dict:
        """Run one lab primitive. The result is written to disk and logged in the notebook automatically.

        By default a run needs ``prereg_id``: the experiment that was pre-registered is run exactly as specified and its
        result is scored against the prediction. ``exploratory=True`` allows a run without one; exploratory runs are
        listed separately in the report and can never be the only support for a claim. Looking at a dataset is exempt.
        """
        from .notebook import score_prediction

        self._ensure_lab()
        pre = None
        if prereg_id:
            pre = self._prereg_entry(prereg_id)
            spec = pre["experiment_spec"]
            if (primitive and primitive != spec["primitive"]) or (params and params != spec["params"]):
                raise ValueError(f"{prereg_id} pre-registered {spec['primitive']} with fixed params. Run it with prereg_id only "
                                 "(omit primitive and params); to run something else, pre-register that.")
            primitive, params, exploratory = spec["primitive"], spec["params"], False
        elif primitive == "dataset":
            exploratory = True                      # inspecting inputs is not an experiment on the model
        elif not exploratory:
            raise ValueError("this run has no pre-registration. Call preregister(hypothesis, experiment_spec, prediction, "
                             "alternatives) first and pass its prereg_id, or pass exploratory=true (exploratory runs are "
                             "reported separately and cannot be the only support for a claim).")
        if not primitive:
            raise ValueError("give a prereg_id, or a primitive with exploratory=true")
        bud = self._budget()
        if bud["remaining"] == 0:
            raise RuntimeError(f"experiment budget of {bud['budget']} is used up. Write the report with what you have, "
                               "or ask the user to raise the budget.")
        rec = self.lab.run(primitive, params or {})
        score = score_prediction(pre["prediction"], rec) if pre else None
        rec.update({"prereg_id": prereg_id, "exploratory": bool(exploratory), "score": score})
        (self._dir() / "experiments" / f"{rec['id']}_{primitive}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        self.notebook.append("experiment", note or (pre["hypothesis"][:120] if pre else primitive), refs=[rec["id"]],
                             data={"primitive": primitive, "seconds": rec["seconds"], "tag": rec["provenance"]["tag"],
                                   "prereg_id": prereg_id, "exploratory": bool(exploratory), "score": score})
        out = {k: rec[k] for k in ("id", "primitive", "provenance", "summary", "plot", "seconds")}
        out["exploratory"] = bool(exploratory)
        if score:
            out["prediction_score"] = score
        raw = json.dumps(rec["data"])
        if detail == "full" or (detail == "auto" and len(raw) <= DATA_LIMIT):
            out["data"] = rec["data"]
        else:
            out["data"] = f"omitted ({len(raw)} chars); full record on disk: experiments/{rec['id']}_{primitive}.json"
        out["budget"] = self._budget()
        return out

    def notebook_verify(self, investigation: str | None = None) -> dict:
        """Recompute the notebook's hash chain and report the first tampered or reordered entry."""
        self._open(investigation)
        return {"investigation": self.investigation, **self.notebook.verify()}

    def get_plot(self, experiment_id: str) -> str:
        """Path of the PNG for an experiment (the MCP tool returns the image itself)."""
        self._open(None)
        hits = sorted((self._dir() / "experiments").glob(f"{experiment_id}_*.png"))
        if not hits:
            raise KeyError(f"no plot for '{experiment_id}'")
        return str(hits[0])

    def notebook_append(self, kind: str, text: str, refs: list[str] | None = None, status: str | None = None,
                        target: str | None = None) -> dict:
        """Write a hypothesis, result, decision or note. ``target`` + ``status`` updates an earlier hypothesis."""
        self._open(None)
        if kind == "prereg":
            raise ValueError("use the preregister tool for pre-registrations")
        known = self._known_ids()
        missing = [r for r in (refs or []) if r not in known]
        if missing:
            raise KeyError(f"refs {missing} do not exist. Cite experiment ids (E0001) or notebook ids (N0001) that are real.")
        if kind in ("result",) and not refs:
            raise ValueError("a result must cite the experiment(s) it comes from in 'refs'")
        return self.notebook.append(kind, text, refs, status, target)

    def _known_ids(self) -> set[str]:
        ids = {e["id"] for e in self.notebook.entries()}
        exp = self._dir() / "experiments"
        if exp.exists():
            ids |= {p.name[:5] for p in exp.glob("[EV]*.json")}
        return ids

    def notebook_read(self, investigation: str | None = None, last_n: int | None = None, kinds: list[str] | None = None) -> dict:
        """Read the notebook. Passing ``investigation`` switches to it, which is how a session resumes."""
        if investigation is None and self.investigation is None:
            names = sorted(p.parent.name for p in self.root.glob("*/notebook.jsonl"))
            if len(names) != 1:
                return {"investigations": names, "note": "pass 'investigation' to open one, or call load_model to start"}
            investigation = names[0]
        self._open(investigation)
        view = self.notebook.view()
        if kinds:
            view = [e for e in view if e["kind"] in kinds]
        if last_n:
            view = view[-last_n:]
        st = self._state()
        return {"investigation": self.investigation, "model": st.get("model"), "backend": st.get("backend"),
                "model_in_memory": self.lab is not None, "budget": self._budget(),
                "summary": self.notebook.summary(), "entries": view,
                "resume_hint": "The model reloads automatically on the next run_experiment." if st.get("model") else
                               "No model yet: call load_model."}

    def verify_hypothesis(self, spec: dict) -> dict:
        """Check a written-out algorithm and/or a claimed circuit on held-out inputs. See glassbox.verify."""
        from .verify import verify

        self._ensure_lab()
        result, fig = verify(self.lab, spec)
        vid = self.lab._next("experiments", "V")
        plot = f"experiments/{vid}_verify.png"
        fig.savefig(self._dir() / plot)
        import matplotlib.pyplot as plt

        plt.close(fig)
        from .lab.core import _jsonable

        result = _jsonable({"id": vid, **result, "plot": plot, "spec": {k: v for k, v in spec.items()},
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        (self._dir() / "experiments" / f"{vid}_verify.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        self.notebook.append("verification", f"{result['name']}: {result['status']}. {result['note']}", refs=[vid],
                             data={"status": result["status"], "tag": result["provenance"]["tag"]})
        return {k: v for k, v in result.items() if k != "spec"}

    def write_report(self, report: dict) -> dict:
        """Write report.md and report.html. Every claim must cite experiments that exist."""
        from .report import write_report

        self._open(None)
        paths = write_report(self._dir(), report, self.notebook, self._state())
        self.notebook.append("report", f"Report written: {paths['markdown']}", refs=paths["cited"])
        return {**paths, "folder": str(self._dir())}
