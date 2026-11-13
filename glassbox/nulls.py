"""Null controls: versions of an investigation where there is, by construction, nothing to find.

Three nulls, all with the same architecture as the target model:

``random-init``      fresh random weights. No training, no learned structure.
``weight-shuffled``  the trained weights, permuted within each matrix. Every weight statistic is kept
                     (mean, variance, norm per matrix); the learned structure is destroyed.
``task-shuffled``    the real, unchanged model, but the answers of the behaviour are shuffled across the
                     prompts, so the "behaviour" being explained does not exist.

They are used in two ways. ``verify_hypothesis`` compares each effect with what the same measurement
gives in null models (see ``glassbox.verify``). And ``glassbox null-run <investigation>`` reruns a whole
investigation, with the same prompt, budget and scientist instructions, on each null: how often the
pipeline still reports a verified circuit is its false discovery rate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

KINDS = ("random-init", "weight-shuffled", "task-shuffled")
WEIGHT_KINDS = ("random-init", "weight-shuffled")


def parse(spec: Any) -> dict | None:
    """``"weight-shuffled:3"`` or ``{"kind": ..., "seed": ...}`` -> ``{"kind", "seed"}``."""
    if not spec:
        return None
    if isinstance(spec, str):
        kind, _, seed = spec.partition(":")
        spec = {"kind": kind, "seed": int(seed or 0)}
    if spec["kind"] not in KINDS:
        raise ValueError(f"null kind must be one of {KINDS}")
    return {"kind": spec["kind"], "seed": int(spec.get("seed", 0))}


def from_env() -> dict | None:
    """A null requested through the environment (how ``null-run`` applies it without telling the scientist)."""
    return parse(os.environ.get("GLASSBOX_NULL"))


def apply_to_hooked_transformer(model, kind: str, seed: int = 0) -> None:
    """Turn a TransformerLens model into a weight null, in place."""
    if kind not in WEIGHT_KINDS:
        raise ValueError(f"'{kind}' is not a weight null")
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        if kind == "random-init":
            was = model.cfg.seed
            torch.manual_seed(1000 + seed)
            model.cfg.seed = None
            model.init_weights()
            model.cfg.seed = was
            for name, p in model.named_parameters():          # biases and norm gains: back to their neutral values
                if name.split(".")[-1].startswith("b"):
                    p.zero_()
        else:
            for p in model.parameters():
                flat = p.detach().flatten()
                perm = torch.randperm(flat.numel(), generator=gen).to(flat.device)
                p.copy_(flat[perm].reshape(p.shape))


def summarize_null_runs(home: Path, investigation: str) -> dict:
    """Collect the outcomes of ``<investigation>__null-*`` runs and compute the false discovery rate."""
    rows = []
    for d in sorted(Path(home).glob(f"{investigation}__null-*")):
        state = json.loads((d / "state.json").read_text(encoding="utf-8")) if (d / "state.json").exists() else {}
        verifs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "experiments").glob("V*.json"))] if (d / "experiments").exists() else []
        report = json.loads((d / "report" / "report.json").read_text(encoding="utf-8")) if (d / "report" / "report.json").exists() else None
        final = verifs[-1]["status"] if verifs else None
        status = (report or {}).get("overall_status", "")
        discovered = final in ("verified", "screening-pass") or status.startswith(("verified", "screening"))
        rows.append({
            "run": d.name, "null": state.get("null"),
            "experiments": len(list((d / "experiments").glob("E*.json"))) if (d / "experiments").exists() else 0,
            "verifications": [v["status"] for v in verifs], "report_status": status or None,
            "discovered_a_circuit": bool(discovered),
        })
    n = len(rows)
    k = sum(r["discovered_a_circuit"] for r in rows)
    return {"investigation": investigation, "null_runs": rows, "n": n, "false_discoveries": k,
            "false_discovery_rate": (k / n) if n else None,
            "definition": "a null run counts as a false discovery if its final verification is 'verified' or "
                          "'screening-pass', or its report status says verified / screening"}
