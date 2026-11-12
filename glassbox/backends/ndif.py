"""ndif: nnsight remote execution on the National Deep Inference Fabric, at full precision.

NDIF hosts large open-weight models and runs your interventions next to them, free for research.
The API key is read only from the ``NDIF_API_KEY`` environment variable of this process (nnsight
picks it up itself). Glassbox never stores, logs or returns it.
"""
from __future__ import annotations

from typing import Any

from . import ndif_status
from .nnsight_common import NNsightBackend


class NDIF(NNsightBackend):
    name = "ndif"
    remote = True

    def list_models(self) -> list[dict[str, Any]]:
        live = ndif_status.live_models()
        if not live["reachable"]:
            raise RuntimeError(f"NDIF is not reachable: {live['reason']}")
        return live["models"]

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        """Bind to a model hosted on NDIF. Nothing is downloaded except the config and tokenizer."""
        if not ndif_status.have_key():
            raise RuntimeError(ndif_status.MISSING_KEY)
        hosted = {m["id"]: m for m in self.list_models()}
        if model_id not in hosted:
            ready = [m["id"] for m in hosted.values() if m["status"] == "ready"]
            raise RuntimeError(f"NDIF does not host '{model_id}' right now. Ready models: {ready}")
        from nnsight.modeling.transformers import TransformersModel

        self.unload()
        self.model = TransformersModel(model_id, task="text-generation", **kwargs)
        self.model_id = model_id
        self._adapt()
        self.hosted = hosted[model_id]
        return self.info()

    def info(self) -> dict[str, Any]:
        out = super().info()
        h = getattr(self, "hosted", {})
        out.update({"ndif_status": h.get("status"), "params_b": h.get("params_b"), "served_dtype": h.get("dtype"),
                    "note": None if h.get("status") == "ready" else
                    "This model is hosted but not running; the first request starts it and can take several minutes."})
        return out
