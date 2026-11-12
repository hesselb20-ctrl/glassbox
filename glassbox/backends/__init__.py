"""Four backends behind one interface: local-full, local-gguf, local-4bit, ndif."""
from __future__ import annotations

from .base import BACKENDS, PRIMITIVES, Backend, Edit, PreflightError, Run, UnsupportedPrimitive, capability_table, require

__all__ = ["BACKENDS", "PRIMITIVES", "Backend", "Edit", "Run", "PreflightError", "UnsupportedPrimitive",
           "capability_table", "require", "get_backend"]


def get_backend(name: str, **kwargs) -> Backend:
    """Instantiate a backend by name. Imports lazily so unused backends cost nothing."""
    if name == "local-full":
        from .local_full import LocalFull
        return LocalFull(**kwargs)
    if name == "local-gguf":
        from .local_gguf import LocalGGUF
        return LocalGGUF(**kwargs)
    if name == "local-4bit":
        from .local_4bit import Local4Bit
        return Local4Bit(**kwargs)
    if name == "ndif":
        from .ndif import NDIF
        return NDIF(**kwargs)
    raise KeyError(f"unknown backend '{name}'. Known: {sorted(BACKENDS)}")
