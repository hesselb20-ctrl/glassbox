"""Shared nnsight implementation behind ``local-4bit`` (quantized, local) and ``ndif`` (full precision, remote).

One forward pass = one nnsight trace. Sites are visited in forward order (nnsight requires it),
edits are applied in place on the real activations, and reads are saved. On NDIF the body of the
trace is shipped to the server and runs next to the model, so only the requested slices come back.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from .base import Backend, Edit, Run, UnsupportedPrimitive, parse_site

ORDER = {"embed": -1, "resid_pre": 0, "head_out": 1, "attn_out": 2, "resid_mid": 3, "mlp_out": 4, "resid_post": 5}


def _edit_inplace(x, op, val, pos, head):
    """Apply one edit in place on a real activation tensor. Pure torch, so it also runs server-side on NDIF."""
    import torch as _t

    if isinstance(pos, int):
        pos = [pos]
    idx = [slice(None), slice(None) if pos is None else list(pos)]
    if head is not None:
        idx.append(head)
    idx = tuple(idx)
    tgt = x[idx]
    if isinstance(val, _t.Tensor):
        val = val.to(device=x.device, dtype=x.dtype)
        if head is not None and val.dim() == 4:
            val = val[:, :, head]
        if pos is not None and val.dim() == tgt.dim() and val.shape[1] == x.shape[1] != tgt.shape[1]:
            val = val[:, list(pos)]
    if op == "replace":
        new = val.expand_as(tgt)
    elif op == "add":
        new = tgt + val
    elif op == "zero":
        new = _t.zeros_like(tgt)
    else:
        basis = val["basis"].to(device=x.device, dtype=x.dtype)
        mean = val.get("mean")
        mean = _t.zeros(basis.shape[1], device=x.device, dtype=x.dtype) if mean is None else mean.to(x.device, x.dtype)
        inside = ((tgt - mean) @ basis.T) @ basis
        new = tgt - inside if op == "project_out" else mean + inside
    x[idx] = new


# On NDIF the trace body runs on a server that does not have glassbox installed, so anything the
# body touches from this module (the edit helper above) must travel by value, not by import path.
try:
    import sys as _sys

    import cloudpickle as _cloudpickle

    _cloudpickle.register_pickle_by_value(_sys.modules[__name__])
except Exception:  # noqa: BLE001
    pass


class NNsightBackend(Backend):
    """Common machinery. Subclasses set ``name`` and ``remote`` and implement ``list_models`` / ``load``."""

    remote = False

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.cfg: dict[str, Any] = {}

    # -- architecture adapter ------------------------------------------------ #
    def _adapt(self) -> None:
        m = self.model
        c = m.config
        base = m
        style = None
        for path in (("transformer",), ("model", "language_model"), ("model",)):
            node = m
            try:
                for p in path:
                    node = getattr(node, p)
                if path == ("transformer",):
                    getattr(node, "h")
                    style = "gpt2"
                else:
                    getattr(node, "layers")
                    style = "llama"
                base = node
                break
            except AttributeError:
                continue
        if style is None:
            raise RuntimeError(f"glassbox does not know the module layout of {self.model_id}; "
                               "supported: GPT-2 style (transformer.h) and Llama/Qwen/Gemma style (model.layers)")
        self.style = style
        text_cfg = getattr(c, "text_config", None) or c
        get = lambda *names: next((getattr(text_cfg, n) for n in names if getattr(text_cfg, n, None) is not None), None)  # noqa: E731
        self.cfg = {
            "n_layers": get("num_hidden_layers", "n_layer"),
            "n_heads": get("num_attention_heads", "n_head"),
            "d_model": get("hidden_size", "n_embd"),
            "d_vocab": get("vocab_size"),
            "model_type": getattr(c, "model_type", None),
            "architecture": "MoE" if any(getattr(text_cfg, k, None) for k in ("num_experts", "num_local_experts", "n_routed_experts"))
                            or "moe" in str(getattr(c, "model_type", "")).lower() else "dense",
        }
        if style == "gpt2":
            self._layers = base.h
            n = self.cfg["n_layers"]
            self._attn_names, self._proj_names = ["attn"] * n, ["c_proj"] * n
            self._heads = [self.cfg["n_heads"]] * n
            self._norm, self._embed = base.ln_f, base.wte
        else:
            self._layers = base.layers
            n = self.cfg["n_layers"]
            # Hybrid models (Qwen3.5 / 3.8) alternate ordinary attention with "linear attention" layers that have a
            # different module name and a different number of heads. Record both per layer.
            types = list(getattr(text_cfg, "layer_types", None) or ["full_attention"] * n)
            linear_heads = getattr(text_cfg, "linear_num_value_heads", None)
            self._attn_names = ["linear_attn" if t == "linear_attention" else "self_attn" for t in types]
            self._proj_names = ["out_proj" if t == "linear_attention" else "o_proj" for t in types]
            self._heads = [linear_heads if t == "linear_attention" else self.cfg["n_heads"] for t in types]
            if "linear_attention" in types:
                self.cfg["layer_types"] = types
                self.cfg["full_attention_layers"] = [i for i, t in enumerate(types) if t == "full_attention"]
                self.cfg["heads_per_layer"] = {"full_attention": self.cfg["n_heads"], "linear_attention": linear_heads}
            self._norm, self._embed = base.norm, base.embed_tokens
        self._head = m.lm_head

    def attach(self, model: Any, model_id: str) -> dict[str, Any]:
        """Use an already constructed nnsight model (for tests and custom loading)."""
        self.model, self.model_id = model, model_id
        self._adapt()
        return self.info()

    def info(self) -> dict[str, Any]:
        return {**self.provenance(), **self.cfg, "attn_only": False, "has_tokenizer": True,
                "d_head": (self.cfg["d_model"] // self.cfg["n_heads"]) if self.cfg.get("n_heads") else None,
                "module_style": self.style, "remote": self.remote}

    # -- tokens ------------------------------------------------------------- #
    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        rows = [self._ids(p) for p in prompts]
        if len({len(r) for r in rows}) != 1:
            raise ValueError(f"prompts tokenise to different lengths {sorted({len(r) for r in rows})}; "
                             "patching needs position-aligned prompts")
        return torch.stack(rows)

    def _ids(self, prompt: str) -> torch.Tensor:
        """Tokenise one prompt. A BOS token is put in front if the tokenizer has one and did not add it,
        which matches the TransformerLens convention used by local-full."""
        tok = self.model.tokenizer
        ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        bos = tok.bos_token_id
        if bos is not None and (len(ids) == 0 or ids[0] != bos):
            ids = torch.cat([torch.tensor([bos]), ids])
        return ids

    def token_id(self, tok: str | int) -> int:
        if isinstance(tok, int):
            return tok
        ids = self.model.tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{tok!r} is {len(ids)} tokens, expected exactly one")
        return ids[0]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.model.tokenizer.decode([int(i)]) for i in ids]

    def bos_id(self) -> int | None:
        return self.model.tokenizer.bos_token_id

    # -- forward ------------------------------------------------------------ #
    def _canon(self, site: str) -> tuple[str, int | None]:
        kind, layer = parse_site(site)
        if kind == "resid_pre" and layer > 0:
            return "resid_post", layer - 1
        if kind not in ORDER:
            raise KeyError(f"site '{site}' is not available on backend '{self.name}'. Known kinds: {sorted(ORDER)}")
        return kind, layer

    def forward(self, tokens: torch.Tensor, read: Sequence[str] = (), edits: Sequence[Edit] = (),
                read_pos: int | None = None, logit_pos: int = -1,
                grad_metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
                grad_sites: Sequence[str] = ()) -> Run:
        if grad_metric is not None and self.remote:
            raise UnsupportedPrimitive("gradients are not offered on ndif in v1; run attribution patching on "
                                       "local-full or local-4bit, or use activation_patching here")
        heads = list(self._heads)
        canon = {s: self._canon(s) for s in list(read) + list(grad_sites) + [e.site for e in edits]}
        plan: dict[tuple, dict] = {}
        for s in list(read) + list(grad_sites):
            plan.setdefault(canon[s], {"edits": [], "names": []})["names"].append(s)
        for e in edits:
            plan.setdefault(canon[e.site], {"edits": [], "names": []})["edits"].append((e.op, e.value, e.pos, e.head))
        order = sorted(plan, key=lambda k: ((-1 if k[1] is None else k[1]), ORDER[k[0]]))
        grad_set = set(grad_sites)
        layers, embed, head = self._layers, self._embed, self._head
        attn_names, proj_names = list(self._attn_names), list(self._proj_names)   # plain data: the trace body must not capture glassbox objects
        want_grad = grad_metric is not None
        edit_fn = _edit_inplace
        import nnsight

        with self.model.trace(tokens, remote=self.remote):
            saved = {}
            grefs = {}
            if want_grad:
                e0 = embed.output
                e0.requires_grad_(True)
            for key in order:
                kind, L = key
                item = plan[key]
                if kind == "embed":
                    x = embed.output
                elif kind == "resid_pre":
                    x = layers[L].input
                elif kind == "head_out":
                    z = getattr(getattr(layers[L], attn_names[L]), proj_names[L]).input
                    x = z.view(z.shape[0], z.shape[1], heads[L], -1)
                elif kind == "attn_out":
                    ao = getattr(layers[L], attn_names[L]).output
                    x = ao[0] if isinstance(ao, tuple) else ao
                elif kind == "mlp_out":
                    x = layers[L].mlp.output
                elif kind == "resid_mid":
                    x = layers[L].mlp.input
                else:
                    out = layers[L].output
                    x = out[0] if isinstance(out, tuple) else out
                for op, val, pos, hd in item["edits"]:
                    edit_fn(x, op, val, pos, hd)
                for name in item["names"]:
                    if name in grad_set:
                        grefs[name] = z if kind == "head_out" else x
                    saved[name] = (x if (read_pos is None) else x[:, read_pos]).detach().float().cpu().save()
            lg = head.output[:, logit_pos]
            grads = {}
            if want_grad:
                loss = grad_metric(lg.float()).sum()
                with loss.backward():
                    for name in sorted(grefs, key=lambda n: order.index(canon[n]), reverse=True):
                        grads[name] = grefs[name].grad.detach().float().cpu().save()
            logits = lg.detach().float().cpu().save()
            nnsight.save(saved)
            nnsight.save(grads)

        acts = {s: saved[s] for s in read}
        out_grads = {}
        for s in grad_sites:
            g = grads[s]
            if canon[s][0] == "head_out":
                g = g.view(g.shape[0], g.shape[1], heads[canon[s][1]], -1)
            out_grads[s] = g if read_pos is None else g[:, read_pos]
        return Run(logits=logits, acts=acts, grads=out_grads)

    # -- optional features -------------------------------------------------- #
    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        shape = resid.shape
        x = resid.reshape(-1, 1, shape[-1])
        norm, head = self._norm, self._head
        dummy = torch.tensor([[self.model.tokenizer.eos_token_id or 0]])
        with self.model.trace(dummy, remote=self.remote):
            ref = norm.input                       # a real activation: tells us the device and dtype to use
            out = head(norm(x.to(device=ref.device, dtype=ref.dtype))).detach().float().cpu().save()
        return out.reshape(*shape[:-1], -1)

    def generate(self, prompt: str, max_new_tokens: int = 40, edits: Sequence[Edit] = ()) -> str:
        ids = self._ids(prompt)[None]
        for e in edits:
            if e.op != "add" or e.pos is not None:
                raise UnsupportedPrimitive("generation supports steering edits only (op 'add', all positions)")
        steer = [(self._canon(e.site)[1], e.value) for e in edits]
        layers = self._layers
        with self.model.generate(ids, max_new_tokens=int(max_new_tokens), do_sample=False, remote=self.remote) as tracer:
            if steer:
                for _step in tracer.iter[:int(max_new_tokens)]:
                    for L, v in sorted(steer, key=lambda t: t[0]):
                        out = layers[L].output
                        x = out[0] if isinstance(out, tuple) else out
                        x[:] = x + v.to(device=x.device, dtype=x.dtype)
            result = tracer.result.save()
        return self.model.tokenizer.decode(result[0][ids.shape[1]:], skip_special_tokens=True)

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
