"""NDIF service status: which models are hosted right now, and whether a key is present.

The key is read only from the ``NDIF_API_KEY`` environment variable of this process.
It is never printed, logged, returned or written anywhere.
"""
from __future__ import annotations

import json
import os
from typing import Any

MISSING_KEY = (
    "NDIF_API_KEY is not set in this process. Sign up at https://login.ndif.us to get a free key, then start "
    "the glassbox MCP server with that variable set (see the README, 'Runs for $0'). "
    "Glassbox reads the key from the environment only and never stores it."
)


def have_key() -> bool:
    return bool(os.environ.get("NDIF_API_KEY"))


def key_recognised() -> bool | None:
    """Ask NDIF whether it knows this key. Returns None if the service cannot be reached."""
    if not have_key():
        return False
    try:
        from nnsight import ndif

        return ndif.whoami(os.environ["NDIF_API_KEY"]).get("email") is not None
    except Exception:
        return None


def live_models() -> dict[str, Any]:
    """Query NDIF's live model list. Needs no key."""
    try:
        from nnsight import ndif

        raw = ndif.status(raw=True)
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "reason": str(e), "models": []}
    if not raw:
        return {"reachable": False, "reason": "NDIF did not answer", "models": []}
    best: dict[str, dict] = {}
    rank = {"ready": 0, "starting": 1, "deployable": 2}
    cold = 0
    for v in raw.get("deployments", {}).values():
        level, state = v.get("deployment_level"), v.get("application_state")
        if level == "COLD":
            cold += 1
            continue
        status = "ready" if (level == "HOT" and state == "RUNNING") else "starting" if level == "HOT" else "deployable"
        cfg = json.loads(v.get("config") or "{}")
        entry = {
            "id": v.get("repo_id"),
            "status": status,
            "params_b": round((v.get("n_params") or 0) / 1e9, 1),
            "dtype": v.get("dtype"),
            "model_type": cfg.get("model_type"),
            "architecture": "MoE" if "moe" in str(cfg.get("model_type", "")).lower() or cfg.get("num_experts") else "dense",
            "backend": "ndif",
        }
        held = best.get(entry["id"])
        if held is None or rank[status] < rank[held["status"]]:
            best[entry["id"]] = entry
    models = sorted(best.values(), key=lambda m: (rank[m["status"]], -m["params_b"]))
    return {"reachable": True, "models": models, "cold_models_not_listed": cold,
            "legend": {"ready": "running now", "deployable": "hosted, starts on first request (can take minutes)"}}
