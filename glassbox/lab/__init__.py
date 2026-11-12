"""Experiment primitives. ``Lab(backend, out_dir).run(primitive, params)`` returns structured JSON plus a plot."""
from .core import Lab, Metric
from .primitives import REGISTRY

__all__ = ["Lab", "Metric", "REGISTRY"]
