"""local-full: TransformerLens on a local GPU, full precision.

Loads either a TransformerLens pretrained model (``gpt2``, ``pythia-70m``, ``attn-only-2l``, ...)
or a glassbox checkpoint (``checkpoint:<path>.pt`` holding ``{"cfg": {...}, "state_dict": ...}``),
or a model object handed over directly (used by the tests).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable, Sequence

import torch

from .base import Backend, Edit, Run, apply_edit, parse_site
from .preflight import gpu_preflight

_HOOKS = {
    "embed": "hook_embed",
    "resid_pre": "blocks.{L}.hook_resid_pre",
    "resid_mid": "blocks.{L}.hook_resid_mid",
    "resid_post": "blocks.{L}.hook_resid_post",
    "attn_out": "blocks.{L}.hook_attn_out",
    "mlp_out": "blocks.{L}.hook_mlp_out",
    "mlp_act": "blocks.{L}.mlp.hook_post",
    "head_out": "blocks.{L}.attn.hook_z",
    "pattern": "blocks.{L}.attn.hook_pattern",
    "head_q_in": "blocks.{L}.hook_q_input",
    "head_k_in": "blocks.{L}.hook_k_input",
    "head_v_in": "blocks.{L}.hook_v_input",
}

#: Small open models that run comfortably at full precision on a 12 GB card.
SUGGESTED = [
    ("gpt2", "GPT-2 small, 12 layers, 85M"),
    ("attn-only-2l", "2-layer attention-only toy model (induction heads)"),
    ("attn-only-1l", "1-layer attention-only toy model"),
    ("gelu-2l", "2-layer toy model with MLPs"),
    ("pythia-70m", "Pythia 70M, 6 layers"),
    ("pythia-160m", "Pythia 160M, 12 layers"),
    ("gpt2-medium", "GPT-2 medium, 24 layers, 345M"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5 0.5B, 24 layers"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5 1.5B, 28 layers"),
]


class LocalFull(Backend):
    name = "local-full"

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.device = device or os.environ.get("GLASSBOX_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None

    # -- lifecycle ---------------------------------------------------------- #
    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": m, "note": note, "backend": self.name} for m, note in SUGGESTED] + [
            {"id": "checkpoint:<path-to.pt>", "note": "a glassbox toy-model checkpoint", "backend": self.name}
        ]

    def load(self, model_id: str | None = None, model: Any = None, **kwargs: Any) -> dict[str, Any]:
        from transformer_lens import HookedTransformer, HookedTransformerConfig

        if self.device.startswith("cuda"):
            gpu_preflight()
        self.unload()
        from ..nulls import WEIGHT_KINDS, parse

        kwargs_null = parse(kwargs.pop("null", None))
        if kwargs_null and kwargs_null["kind"] not in WEIGHT_KINDS:
            kwargs_null = None                      # task-shuffled is applied to the labels, not the weights
        self.vocab = None
        if model is not None:
            self.model = model
            self.model_id = model_id or "in-memory"
        elif model_id.startswith("checkpoint:"):
            ckpt = torch.load(model_id.split(":", 1)[1], map_location="cpu", weights_only=False)
            c = dict(ckpt["cfg"])
            if isinstance(c.get("dtype"), str):
                c["dtype"] = getattr(torch, c["dtype"])
            cfg = HookedTransformerConfig(**c)
            self.model = HookedTransformer(cfg)
            self.model.load_state_dict({k: v.to(cfg.dtype) for k, v in ckpt["state_dict"].items()}, strict=False)
            self.model_id = model_id
            self.vocab = ckpt.get("vocab")           # token strings for models without a tokenizer
        else:
            dtype = kwargs.pop("dtype", None)
            extra = {"torch_dtype": getattr(torch, dtype)} if dtype else {}
            self.model = HookedTransformer.from_pretrained(model_id, device="cpu", **extra, **kwargs)
            self.model_id = model_id
        null = kwargs_null
        if null:
            from ..nulls import apply_to_hooked_transformer

            apply_to_hooked_transformer(self.model, null["kind"], null["seed"])
        self.null = null
        self.model.to(self.device, print_details=False)
        self.model.eval()
        self.model.requires_grad_(False)
        return self.info()

    def null_copy(self, kind: str, seed: int = 0) -> "LocalFull":
        """A null version of the loaded model (same architecture): fresh random weights, or weights shuffled within each matrix."""
        import copy

        from ..nulls import apply_to_hooked_transformer

        other = LocalFull(device=self.device)
        other.model = copy.deepcopy(self.model)
        other.model_id, other.vocab = f"{self.model_id} [{kind} null, seed {seed}]", self.vocab
        apply_to_hooked_transformer(other.model, kind, seed)
        other.model.eval()
        other.model.requires_grad_(False)
        other.null = {"kind": kind, "seed": seed}
        return other

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> dict[str, Any]:
        c = self.model.cfg
        return {
            **self.provenance(),
            "n_layers": c.n_layers,
            "n_heads": c.n_heads,
            "d_model": c.d_model,
            "d_head": c.d_head,
            "d_mlp": None if c.attn_only else c.d_mlp,
            "d_vocab": c.d_vocab,
            "n_ctx": c.n_ctx,
            "attn_only": bool(c.attn_only),
            "architecture": "dense",
            "has_tokenizer": self.model.tokenizer is not None,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
        }

    # -- tokens ------------------------------------------------------------- #
    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        if self.model.tokenizer is None:
            raise ValueError("this model has no tokenizer; pass token ids instead of prompts")
        rows = [self.model.to_tokens(p)[0] for p in prompts]
        lengths = {len(r) for r in rows}
        if len(lengths) != 1:
            raise ValueError(
                f"prompts tokenise to different lengths {sorted(lengths)}; "
                "patching needs position-aligned prompts"
            )
        return torch.stack(rows)

    def token_id(self, tok: str | int) -> int:
        if isinstance(tok, int):
            return tok
        if self.model.tokenizer is None:
            v = getattr(self, "vocab", None)
            if v and tok in v:
                return v.index(tok)
            return int(tok)
        ids = self.model.tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{tok!r} is {len(ids)} tokens, expected exactly one")
        return ids[0]

    def bos_id(self) -> int | None:
        tok = self.model.tokenizer
        if tok is None or not self.model.cfg.default_prepend_bos:
            return None
        return tok.bos_token_id

    def decode(self, ids: Sequence[int]) -> list[str]:
        if self.model.tokenizer is None:
            v = getattr(self, "vocab", None)
            return [str(v[int(i)]) if v and 0 <= int(i) < len(v) else str(int(i)) for i in ids]
        return [self.model.tokenizer.decode([int(i)]) for i in ids]

    # -- forward ------------------------------------------------------------ #
    def _hook_name(self, site: str) -> str:
        kind, layer = parse_site(site)
        if kind not in _HOOKS:
            raise KeyError(f"unknown site '{site}'. Known kinds: {sorted(_HOOKS)}")
        if kind.startswith("mlp") and self.model.cfg.attn_only:
            raise KeyError(f"site '{site}' does not exist: this model is attention-only")
        return _HOOKS[kind].format(L=layer)

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
        model = self.model
        tokens = tokens.to(self.device)
        want = {self._hook_name(s): s for s in list(read) + list(grad_sites)}
        edits_by_hook: dict[str, list[Edit]] = defaultdict(list)
        for e in edits:
            edits_by_hook[self._hook_name(e.site)].append(e)
        split = any(n.endswith(("q_input", "k_input", "v_input")) for n in set(want) | set(edits_by_hook))
        model.cfg.use_split_qkv_input = split
        grad_set = set(grad_sites)
        stash: dict[str, torch.Tensor] = {}

        def make(hook_name: str):
            def fn(x, hook):
                for e in edits_by_hook.get(hook_name, ()):
                    x = apply_edit(x, e)
                site = want.get(hook_name)
                if site is not None:
                    if site in grad_set and not x.requires_grad:
                        x.requires_grad_(True)
                    stash[site] = x
                return x
            return fn

        hooks = [(n, make(n)) for n in set(want) | set(edits_by_hook)]
        with torch.set_grad_enabled(grad_metric is not None):
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")[:, logit_pos]
            grads: dict[str, torch.Tensor] = {}
            if grad_metric is not None:
                total = grad_metric(logits).sum()
                order = list(grad_sites)
                got = torch.autograd.grad(total, [stash[s] for s in order], allow_unused=True)
                grads = {s: (torch.zeros_like(stash[s]) if g is None else g).detach() for s, g in zip(order, got)}
        model.cfg.use_split_qkv_input = False
        acts = {}
        for s in read:
            a = stash[s].detach()
            if read_pos is not None and not s.startswith("pattern"):
                a = a[:, read_pos]
            acts[s] = a
        if read_pos is not None:
            grads = {s: g[:, read_pos] for s, g in grads.items()}
        return Run(logits=logits.detach(), acts=acts, grads=grads)

    # -- optional features -------------------------------------------------- #
    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        shape = resid.shape
        x = resid.reshape(-1, 1, shape[-1]).to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "ln_final"):
                x = self.model.ln_final(x)
            out = self.model.unembed(x)
        return out.reshape(*shape[:-1], -1)

    def final_scale(self, resid: torch.Tensor) -> torch.Tensor:
        if self.model.cfg.normalization_type in ("LN", "LNPre"):
            x = resid - resid.mean(-1, keepdim=True)
            return (x.pow(2).mean(-1, keepdim=True) + self.model.cfg.eps).sqrt()
        return torch.ones_like(resid[..., :1])

    def head_write(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        W_O = self.model.blocks[layer].attn.W_O            # [head, d_head, d_model]
        return torch.einsum("bphd,hdm->bphm", z.to(W_O.device), W_O)

    def weight(self, name: str) -> torch.Tensor:
        m = self.model
        table = {"W_E": lambda: m.W_E, "W_U": lambda: m.W_U, "W_pos": lambda: m.W_pos}
        if name not in table:
            raise KeyError(f"unknown weight '{name}'. Known: {sorted(table)}")
        return table[name]().detach()

    def generate(self, prompt: str, max_new_tokens: int = 40, edits: Sequence[Edit] = ()) -> str:
        tokens = self.model.to_tokens(prompt).to(self.device)
        n0 = tokens.shape[1]
        for _ in range(max_new_tokens):
            nxt = self.forward(tokens, edits=edits).logits.argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
        return self.model.tokenizer.decode(tokens[0, n0:].tolist())
