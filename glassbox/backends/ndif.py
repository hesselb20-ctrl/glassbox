"""ndif: nnsight remote execution on the National Deep Inference Fabric, full precision."""
from __future__ import annotations

from typing import Any

from . import ndif_status
from .base import Backend


class NDIF(Backend):
    name = "ndif"

    def list_models(self) -> list[dict[str, Any]]:
        live = ndif_status.live_models()
        if not live["reachable"]:
            raise RuntimeError(f"NDIF is not reachable: {live['reason']}")
        return live["models"]

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("ndif backend: remote execution lands in phase 5")

    info = encode = token_id = decode = forward = load
