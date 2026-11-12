"""The single interface every glassbox backend implements.

A backend exposes a model as a set of named *sites* (places in the forward pass)
that can be read, edited, or differentiated through. Lab primitives are written
once against this interface, so the same experiment runs on a 1-layer toy model
on your GPU and on a 70B model on NDIF.

Site names
----------
``embed``            token (+pos) embedding output          [batch, pos, d_model]
``resid_pre.L``      residual stream entering layer L       [batch, pos, d_model]
``resid_mid.L``      residual stream between attn and MLP   [batch, pos, d_model]
``resid_post.L``     residual stream leaving layer L        [batch, pos, d_model]
``attn_out.L``       attention block output of layer L      [batch, pos, d_model]
``mlp_out.L``        MLP output of layer L                  [batch, pos, d_model]
``mlp_act.L``        MLP hidden (post-nonlinearity)         [batch, pos, d_mlp]
``head_out.L``       per-head mixed values ("z")            [batch, pos, n_heads, d_head]
``pattern.L``        attention pattern                      [batch, n_heads, query, key]
``head_q_in.L`` / ``head_k_in.L`` / ``head_v_in.L``
                     per-head query/key/value inputs        [batch, pos, n_heads, d_model]
"""
from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Sequence

import torch

# --------------------------------------------------------------------------- #
# Capability table
# --------------------------------------------------------------------------- #

#: Atomic features a backend may offer.
FEATURES = (
    "read",         # read residual / attn / MLP outputs
    "read_heads",   # read per-head outputs
    "pattern",      # read attention patterns
    "edit",         # replace / ablate activations at any site (true patching)
    "steer",        # add a direction to the residual stream
    "grad",         # gradients of a metric w.r.t. activations
    "head_inputs",  # split q/k/v inputs per head (path patching to heads)
    "unembed",      # map a residual vector to vocabulary logits
    "weights",      # direct access to weight matrices
    "generate",     # sample text
)

#: Static description of every backend, usable without loading anything.
BACKENDS: dict[str, dict[str, Any]] = {
    "local-full": {
        "precision": "full",
        "tag": "full precision (local)",
        "verifies": True,
        "features": set(FEATURES),
        "summary": "TransformerLens on your own GPU. Small models at fp32/bf16.",
    },
    "local-gguf": {
        "precision": "gguf-quantized",
        "tag": "screening (gguf quantized)",
        "verifies": False,
        "features": {"read", "steer", "unembed", "generate"},
        "summary": "llama.cpp on a quantized GGUF file. Read activations and steer; no patching in v1.",
    },
    "local-4bit": {
        "precision": "4bit",
        "tag": "screening (4-bit)",
        "verifies": False,
        "features": {"read", "read_heads", "edit", "steer", "grad", "unembed", "generate"},
        "summary": "nnsight + bitsandbytes 4-bit across both GPUs. Full patching, quantization noise.",
    },
    "ndif": {
        "precision": "full",
        "tag": "full precision (NDIF remote)",
        "verifies": True,
        "features": {"read", "read_heads", "edit", "steer", "unembed", "generate"},
        "summary": "nnsight remote execution on NDIF. Large models at full precision, free for research.",
    },
}

#: Features each lab primitive needs.
PRIMITIVES: dict[str, set[str]] = {
    "dataset": set(),
    "baseline": set(),
    "attribution_patching": {"read", "grad"},
    "activation_patching": {"read", "edit"},
    "path_patching": {"read", "read_heads", "edit"},
    "ablation": {"read", "edit"},
    "logit_lens": {"read", "unembed"},
    "direct_logit_attribution": {"read", "read_heads", "weights", "unembed"},
    "linear_probe": {"read"},
    "attention_patterns": {"pattern"},
    "diff_of_means": {"read"},
    "steering": {"steer"},
    "periodicity": {"read", "weights"},
    "weights_svd": {"weights"},
}


class UnsupportedPrimitive(RuntimeError):
    """Raised when a primitive is requested on a backend that cannot run it."""


class PreflightError(RuntimeError):
    """Raised when the machine is not in a state where a GPU run is safe."""


def backends_supporting(primitive: str) -> list[str]:
    need = PRIMITIVES[primitive]
    return [name for name, b in BACKENDS.items() if need <= b["features"]]


def require(backend_name: str, primitive: str, extra: Sequence[str] = ()) -> None:
    """Raise a clear error if ``backend_name`` cannot run ``primitive``."""
    if primitive not in PRIMITIVES:
        raise KeyError(f"unknown primitive '{primitive}'. Known: {sorted(PRIMITIVES)}")
    need = PRIMITIVES[primitive] | set(extra)
    have = BACKENDS[backend_name]["features"]
    missing = need - have
    if missing:
        ok = [n for n, b in BACKENDS.items() if need <= b["features"]]
        raise UnsupportedPrimitive(
            f"'{primitive}' is not supported on backend '{backend_name}' "
            f"(missing: {', '.join(sorted(missing))}). "
            f"Backends that support it: {', '.join(ok) if ok else 'none'}."
        )


def capability_table() -> dict[str, Any]:
    """Primitive x backend support matrix, as plain JSON."""
    table = {
        prim: {name: need <= b["features"] for name, b in BACKENDS.items()}
        for prim, need in PRIMITIVES.items()
    }
    meta = {
        name: {k: (sorted(v) if isinstance(v, set) else v) for k, v in b.items()}
        for name, b in BACKENDS.items()
    }
    return {"backends": meta, "primitives": table}


# --------------------------------------------------------------------------- #
# Edits and run results
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Edit:
    """One intervention on the forward pass.

    op:
      ``replace``      overwrite with ``value``
      ``copy``         overwrite with the same activation taken from other batch rows of this forward pass
                       (``value`` = the source row indices); nothing has to be uploaded to a remote backend
      ``add``          add ``value`` (steering)
      ``zero``         set to zero
      ``project_out``  remove the component inside a subspace
      ``keep_only``    remove everything *outside* a subspace
    For the two subspace ops ``value`` is ``{"basis": [k, d] orthonormal, "mean": [d] or None}``;
    components are measured relative to ``mean`` so the edit is a mean-ablation of the subspace.
    """

    site: str
    op: str = "replace"
    value: Any = None
    pos: Any = None          # int, list[int] or None for all positions
    head: int | None = None  # head index for per-head sites
    rows: Any = None         # batch rows the edit applies to (None = all); lets one forward pass carry many variants


@dataclasses.dataclass
class Run:
    logits: torch.Tensor                 # [batch, vocab] at the requested position
    acts: dict[str, torch.Tensor]
    grads: dict[str, torch.Tensor]


def apply_edit(x: torch.Tensor, e: Edit) -> torch.Tensor:
    """Apply ``e`` to activation ``x`` ([batch, pos, d] or [batch, pos, head, d])."""
    if e.op == "copy":
        e = dataclasses.replace(e, op="replace", value=x[list(e.value)])
    if e.rows is not None:
        rows = list(e.rows)
        out = x.clone()
        out[rows] = apply_edit(x[rows], dataclasses.replace(e, rows=None))
        return out
    pos = e.pos
    if isinstance(pos, int):
        pos = [pos]
    idx: list[Any] = [slice(None), slice(None) if pos is None else list(pos)]
    if e.head is not None:
        if x.dim() != 4:
            raise ValueError(f"edit on {e.site} names a head but the site has no head axis")
        idx.append(e.head)
    idx_t = tuple(idx)
    target = x[idx_t]

    v = e.value
    if isinstance(v, torch.Tensor):
        v = v.to(device=x.device, dtype=x.dtype)
        if e.head is not None and v.dim() == 4:
            v = v[:, :, e.head]
        if pos is not None and v.dim() == target.dim() and v.shape[1] == x.shape[1] != target.shape[1]:
            v = v[:, list(pos)]

    if e.op == "replace":
        new = v.expand_as(target)
    elif e.op == "add":
        new = target + v
    elif e.op == "zero":
        new = torch.zeros_like(target)
    elif e.op in ("project_out", "keep_only"):
        basis = v["basis"].to(device=x.device, dtype=x.dtype)          # [k, d]
        mean = v.get("mean")
        mean = torch.zeros(basis.shape[1], device=x.device, dtype=x.dtype) if mean is None else mean.to(x.device, x.dtype)
        centred = target - mean
        inside = (centred @ basis.T) @ basis
        new = target - inside if e.op == "project_out" else mean + inside
    else:
        raise ValueError(f"unknown edit op '{e.op}'")
    out = x.clone()
    out[idx_t] = new
    return out


def parse_site(site: str) -> tuple[str, int | None]:
    kind, _, layer = site.partition(".")
    return kind, (int(layer) if layer else None)


# --------------------------------------------------------------------------- #
# Backend ABC
# --------------------------------------------------------------------------- #

class Backend(abc.ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.model_id: str | None = None

    # -- static description ------------------------------------------------- #
    @property
    def spec(self) -> dict[str, Any]:
        return BACKENDS[self.name]

    @property
    def precision(self) -> str:
        return self.spec["precision"]

    @property
    def tag(self) -> str:
        return self.spec["tag"]

    def provenance(self) -> dict[str, Any]:
        """Stamp attached to every result: who produced this number."""
        return {
            "backend": self.name,
            "precision": self.precision,
            "tag": self.tag,
            "model": self.model_id,
            "counts_as_verification": bool(self.spec["verifies"]),
        }

    def require(self, primitive: str, extra: Sequence[str] = ()) -> None:
        require(self.name, primitive, extra)

    # -- lifecycle ---------------------------------------------------------- #
    @abc.abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """Models this backend can load right now."""

    @abc.abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        """Load a model and return :meth:`info`."""

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Architecture facts: n_layers, n_heads, d_model, d_vocab, attn_only, ..."""

    # -- tokens ------------------------------------------------------------- #
    @abc.abstractmethod
    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        """Tokenise prompts into a rectangular [batch, pos] tensor (must be equal length)."""

    @abc.abstractmethod
    def token_id(self, tok: str | int) -> int:
        """Resolve a single-token string to its id."""

    @abc.abstractmethod
    def decode(self, ids: Sequence[int]) -> list[str]:
        """Token ids -> token strings, one per id."""

    def heads_in_layers(self, layers: Sequence[int]) -> list[int]:
        """Number of attention heads in each of the given layers (hybrid models have layers of different kinds)."""
        return [self.info()["n_heads"]] * len(layers)

    def bos_id(self) -> int | None:
        """Id of the token the model expects at position 0, or None."""
        return None

    # -- the forward pass --------------------------------------------------- #
    @abc.abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        read: Sequence[str] = (),
        edits: Sequence[Edit] = (),
        read_pos: int | None = None,
        logit_pos: int = -1,
        grad_metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
        grad_sites: Sequence[str] = (),
    ) -> Run:
        """Run the model with optional reads, edits and gradients."""

    # -- optional features -------------------------------------------------- #
    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        """[..., d_model] residual -> [..., vocab] logits through the final norm."""
        raise UnsupportedPrimitive(f"backend '{self.name}' cannot unembed")

    def final_scale(self, resid: torch.Tensor) -> torch.Tensor:
        """Scale the final norm divides by at this residual, [..., 1] (for frozen-norm attribution)."""
        raise UnsupportedPrimitive(f"backend '{self.name}' has no weight access")

    def head_write(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        """[batch, pos, n_heads, d_head] -> each head's write to the residual [batch, pos, n_heads, d_model]."""
        raise UnsupportedPrimitive(f"backend '{self.name}' has no weight access")

    def weight(self, name: str) -> torch.Tensor:
        """Named weight matrix: W_E [vocab, d], W_U [d, vocab], W_pos."""
        raise UnsupportedPrimitive(f"backend '{self.name}' has no weight access")

    def generate(self, prompt: str, max_new_tokens: int = 40, edits: Sequence[Edit] = ()) -> str:
        raise UnsupportedPrimitive(f"backend '{self.name}' cannot generate")

    def unload(self) -> None:
        pass
