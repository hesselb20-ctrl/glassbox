"""local-4bit: an official Hugging Face checkpoint loaded with nnsight + bitsandbytes 4-bit across your GPUs.

Full patching support, but the weights are 4-bit quantized, so every result is tagged
``screening (4-bit)`` and never counts as verified. Confirm on ``ndif`` at full precision.

The first load of a model downloads its safetensors (tens of GB for a 27B model). Glassbox
refuses to start that download unless ``allow_download=True`` is passed, so the scientist has
to ask the user first.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .nnsight_common import NNsightBackend
from .preflight import gpu_preflight

DEFAULT_MODELS = [m for m in os.environ.get("GLASSBOX_4BIT_MODELS", "Qwen/Qwen3.8-27B").split(",") if m]


def cached(repo_id: str) -> bool:
    """Is this repo's config already in the local Hugging Face cache?"""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


def repo_size_gb(repo_id: str) -> float | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        return round(sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 2**30, 1)
    except Exception:  # noqa: BLE001
        return None


class Local4Bit(NNsightBackend):
    name = "local-4bit"
    remote = False

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        for m in DEFAULT_MODELS:
            have = cached(m)
            out.append({"id": m, "backend": self.name, "downloaded": have,
                        "note": "ready to load" if have else
                        "not downloaded. Loading needs a large download: ask the user, then pass options {'allow_download': true}."})
        out.append({"id": "<any Hugging Face causal LM id>", "backend": self.name,
                    "note": "any official safetensors checkpoint can be loaded in 4-bit"})
        return out

    def load(self, model_id: str, allow_download: bool = False, max_memory: dict | None = None,
             quantize: bool = True, **kwargs: Any) -> dict[str, Any]:
        """``max_memory`` e.g. {"0": "10GiB", "1": "10GiB"}; default leaves ~1.5 GB headroom per GPU.

        ``quantize=False`` loads at full precision on one GPU (small models only; used by the tests).
        """
        if not cached(model_id) and not allow_download:
            size = repo_size_gb(model_id)
            raise RuntimeError(
                f"'{model_id}' is not downloaded yet"
                + (f" (about {size} GB of safetensors)" if size else "")
                + ". Ask the user before starting this download, then call load_model again with "
                  "options {'allow_download': true}.")
        gpu_preflight()
        from nnsight.modeling.transformers import TransformersModel

        self.unload()
        if quantize:
            from transformers import BitsAndBytesConfig

            n = torch.cuda.device_count()
            if max_memory is None:
                max_memory = {i: f"{max(1.0, torch.cuda.mem_get_info(i)[0] / 2**30 - 1.5):.1f}GiB" for i in range(n)}
            else:
                max_memory = {(int(k) if str(k).isdigit() else k): v for k, v in max_memory.items()}
            kwargs.update(
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True),
                device_map="auto", max_memory=max_memory)
            self.precision_note = "4-bit NF4 (bitsandbytes), bf16 compute"
        else:
            kwargs.setdefault("device_map", os.environ.get("GLASSBOX_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
            self.precision_note = "unquantized (test mode)"
        self.model = TransformersModel(model_id, task="text-generation", dispatch=True, **kwargs)
        self.model_id = model_id
        self._adapt()
        return self.info()

    def info(self) -> dict[str, Any]:
        out = super().info()
        out["quantization"] = getattr(self, "precision_note", None)
        try:
            out["device_map"] = sorted({str(v) for v in self.model._model.hf_device_map.values()})
        except Exception:  # noqa: BLE001
            pass
        return out
