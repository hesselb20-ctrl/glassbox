"""local-gguf: read activations from, and steer, a quantized GGUF model through llama.cpp.

What works: reading the residual stream, attention-block output and MLP output at every layer and
token (probes, logit lens, difference of means, activation statistics), steering with control
vectors, and before/after comparisons of behaviour.

What does not work in v1: overwriting activations. So no activation patching, path patching or
ablation on GGUF. Use ``local-4bit`` or ``ndif`` for those.

Every result is tagged ``screening (gguf quantized)``: the weights are quantized, so numbers are
indicative and nothing found here counts as verified.

Environment: ``GLASSBOX_LLAMA_DIR`` (folder of your llama.cpp build, default ``~/llama.cpp``) and
``GLASSBOX_GGUF_DIR`` (folder with .gguf files, default ``~/models``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .base import Backend, Edit, Run, UnsupportedPrimitive, backends_supporting, parse_site
from .llama_cpp import Worker, default_lib_dir
from .preflight import gpu_preflight, gpu_report

#: glassbox site kind -> candidate llama.cpp tensor names (architectures name them differently)
TENSOR_NAMES = {
    "resid_post": ["l_out"],
    "resid_mid": ["attn_residual", "ffn_inp"],
    "mlp_out": ["ffn_out"],
    "embed": ["model.input_embed", "inp_embd"],
}


def gguf_dir() -> Path:
    return Path(os.environ.get("GLASSBOX_GGUF_DIR", Path.home() / "models"))


class LocalGGUF(Backend):
    name = "local-gguf"

    def __init__(self) -> None:
        super().__init__()
        self.worker: Worker | None = None
        self.meta: dict[str, Any] = {}
        self.names: dict[str, str] = {}
        self.wrap = ("", "")
        self.path: Path | None = None
        self._unembed: tuple[torch.Tensor, torch.Tensor] | None = None

    # -- lifecycle ---------------------------------------------------------- #
    def list_models(self) -> list[dict[str, Any]]:
        d = gguf_dir()
        lib = default_lib_dir()
        if not d.exists():
            raise RuntimeError(f"no GGUF folder at {d}. Set GLASSBOX_GGUF_DIR.")
        out = []
        for f in sorted(d.glob("*.gguf")):
            gb = round(f.stat().st_size / 2**30, 1)
            out.append({"id": f.name, "size_gb": gb, "backend": self.name,
                        "note": ("abliterated community variant: not the official weights. " if "abliter" in f.name.lower() else "")
                                + ("very small file, probably a draft/speculative model. " if gb < 2 else "")})
        if not any(lib.glob("*llama*")):
            out.append({"id": None, "note": f"no llama.cpp build found in {lib}; set GLASSBOX_LLAMA_DIR"})
        return out

    def load(self, model_id: str, n_ctx: int = 1024, n_gpu_layers: int = -1,
             wrap: Sequence[str] | None = None, **_: Any) -> dict[str, Any]:
        """``wrap``: [prefix, suffix] put around every prompt, e.g. a chat template."""
        gpu_preflight()
        self.unload()
        path = Path(model_id)
        if not path.exists():
            path = gguf_dir() / model_id
        if not path.exists():
            raise FileNotFoundError(f"no GGUF file '{model_id}' (looked in {gguf_dir()})")
        self.path = path
        before = {g["index"]: g["free_gb"] for g in gpu_report()}
        self.worker = Worker()
        self.meta = self.worker.call("load", model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        # where did the weights actually go? measured, not assumed
        self.vram_gb = {g["name"]: round(before[g["index"]] - g["free_gb"], 1) for g in gpu_report()}
        self.on_gpu = sum(self.vram_gb.values()) > 0.5 * path.stat().st_size / 2**30
        self.model_id = path.name
        self.wrap = tuple(wrap) if wrap else ("", "")
        seen = self.worker.call("names", tokens=self.worker.call("tokenize", texts=["a b"])[0])
        self.names = {}
        for kind, cands in TENSOR_NAMES.items():
            for c in cands:
                if c in seen or f"{c}-0" in seen:
                    self.names[kind] = c
                    break
        if "resid_post" not in self.names:
            raise RuntimeError("this llama.cpp graph exposes no per-layer output tensor ('l_out'); cannot read the residual stream")
        return self.info()

    def unload(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self._unembed = None

    def provenance(self) -> dict[str, Any]:
        p = super().provenance()
        if self.model_id and "abliter" in self.model_id.lower():
            p["variant"] = "abliterated community variant (refusal direction removed); not the official weights"
            p["tag"] = "screening (gguf quantized, abliterated variant)"
        return p

    def info(self) -> dict[str, Any]:
        m = self.meta
        sites = sorted(self.names) + (["attn_out"] if {"resid_mid", "resid_post"} <= set(self.names) else [])
        return {
            **self.provenance(),
            "n_layers": m["n_layer"], "n_heads": m["n_head"], "d_model": m["n_embd"], "d_vocab": m["n_vocab"],
            "attn_only": False,
            "architecture": "MoE" if m.get("expert_count") not in (None, "", "0") else "dense",
            "llama_cpp_arch": m.get("arch"), "gguf_name": m.get("name"), "file": self.model_id,
            "vram_used_gb": getattr(self, "vram_gb", {}),
            "device": "GPU (llama.cpp offload)" if getattr(self, "on_gpu", False) else
                      "CPU (the llama.cpp GPU backend did not load; check that its CUDA DLLs are next to llama.dll)",
            "readable_site_kinds": sites,
            "steerable": "resid_post.L for L >= 1, all positions (llama.cpp control vectors)",
            "prompt_wrap": list(self.wrap),
            "has_tokenizer": True,
        }

    # -- tokens ------------------------------------------------------------- #
    def encode(self, prompts: Sequence[str]) -> torch.Tensor:
        rows = self.worker.call("tokenize", texts=[self.wrap[0] + p + self.wrap[1] for p in prompts])
        if len({len(r) for r in rows}) != 1:
            raise ValueError(f"prompts tokenise to different lengths {sorted({len(r) for r in rows})}")
        return torch.tensor(rows, dtype=torch.long)

    def token_id(self, tok: str | int) -> int:
        if isinstance(tok, int):
            return tok
        ids = self.worker.call("tokenize", texts=[tok], add_special=False)[0]
        if len(ids) != 1:
            raise ValueError(f"{tok!r} is {len(ids)} tokens, expected exactly one")
        return ids[0]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return self.worker.call("pieces", tokens=[int(i) for i in ids])

    # -- forward ------------------------------------------------------------ #
    def _control(self, edits: Sequence[Edit]) -> dict[int, np.ndarray] | None:
        control: dict[int, np.ndarray] = {}
        for e in edits:
            kind, layer = parse_site(e.site)
            if e.op != "add" or kind != "resid_post" or e.pos is not None or e.head is not None:
                ok = ", ".join(b for b in backends_supporting("activation_patching"))
                raise UnsupportedPrimitive(
                    f"local-gguf can only add a direction to resid_post.L at all positions (steering); "
                    f"it cannot apply '{e.op}' at '{e.site}'. Backends with full patching: {ok}.")
            v = e.value.detach().float().cpu().reshape(-1).numpy()
            if v.shape[0] != self.meta["n_embd"]:
                raise ValueError(f"steering vector has {v.shape[0]} dims, model has {self.meta['n_embd']}")
            control[layer] = control.get(layer, 0) + v
        return control or None

    def forward(self, tokens: torch.Tensor, read: Sequence[str] = (), edits: Sequence[Edit] = (),
                read_pos: int | None = None, logit_pos: int = -1,
                grad_metric: Callable | None = None, grad_sites: Sequence[str] = ()) -> Run:
        if grad_metric is not None or grad_sites:
            raise UnsupportedPrimitive("local-gguf has no gradients. Use local-full, local-4bit for attribution patching.")
        if logit_pos != -1:
            raise UnsupportedPrimitive("local-gguf returns logits at the last position only")
        control = self._control(edits)
        want: set[str] = set()
        plan: dict[str, tuple] = {}
        for s in read:
            kind, layer = parse_site(s)
            if kind == "attn_out" and {"resid_mid", "resid_post"} <= set(self.names):
                a = f"{self.names['resid_mid']}-{layer}"
                b = f"{self.names['resid_post']}-{layer - 1}" if layer > 0 else self.names.get("embed")
                if b is None:
                    raise KeyError("attn_out.0 needs the input embedding tensor, which this graph does not expose")
                plan[s] = ("diff", a, b)
                want |= {a, b}
            elif kind in self.names:
                n = self.names[kind] if layer is None else f"{self.names[kind]}-{layer}"
                plan[s] = ("get", n)
                want.add(n)
            else:
                raise KeyError(f"site '{s}' is not readable on local-gguf. Readable kinds: {self.info()['readable_site_kinds']}")
        logits, acts = [], {s: [] for s in read}
        for row in tokens.tolist():
            lg, got = self.worker.call("run", tokens=row, want=sorted(want), control=control)
            logits.append(torch.from_numpy(lg))
            for s, p in plan.items():
                a = got[p[1]] - got[p[2]] if p[0] == "diff" else got[p[1]]
                a = torch.from_numpy(a)
                acts[s].append(a[read_pos] if read_pos is not None else a)
        return Run(logits=torch.stack(logits), acts={s: torch.stack(v) for s, v in acts.items()}, grads={})

    def generate(self, prompt: str, max_new_tokens: int = 40, edits: Sequence[Edit] = ()) -> str:
        toks = self.worker.call("tokenize", texts=[self.wrap[0] + prompt + self.wrap[1]])[0]
        new = self.worker.call("generate", tokens=toks, max_new_tokens=int(max_new_tokens), control=self._control(edits))
        return "".join(self.worker.call("pieces", tokens=new))

    # -- unembedding (for the logit lens) ------------------------------------ #
    def _load_unembed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Final norm weight and unembedding matrix, dequantized from the GGUF file (cached on disk as fp16)."""
        if self._unembed is not None:
            return self._unembed
        cache = Path(os.environ.get("GLASSBOX_CACHE", Path.home() / ".cache" / "glassbox"))
        cache.mkdir(parents=True, exist_ok=True)
        f_w, f_n = cache / f"{self.path.stem}.unembed.f16.npy", cache / f"{self.path.stem}.outnorm.npy"
        if not (f_w.exists() and f_n.exists()):
            from gguf import GGUFReader
            from gguf.quants import dequantize

            reader = GGUFReader(str(self.path))
            tensors = {t.name: t for t in reader.tensors}
            t_w = tensors.get("output.weight") or tensors["token_embd.weight"]
            w = dequantize(t_w.data, t_w.tensor_type).reshape(self.meta["n_vocab"], self.meta["n_embd"])
            np.save(f_w, w.astype(np.float16))
            t_n = tensors["output_norm.weight"]
            np.save(f_n, dequantize(t_n.data, t_n.tensor_type).astype(np.float32).reshape(-1))
        self._unembed = (torch.from_numpy(np.load(f_n)), torch.from_numpy(np.load(f_w)))
        return self._unembed

    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        norm_w, W = self._load_unembed()
        eps = float(self.meta.get("rms_eps") or 1e-6)
        x = resid.float().cpu()
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        out = torch.cat([chunk @ W.float().T for chunk in x.split(16)]) if W.dtype != torch.float32 else x @ W.T
        return out.reshape(*shape[:-1], -1)
