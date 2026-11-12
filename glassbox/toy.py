"""Tiny transformers trained from scratch, for benchmarks with a known ground truth."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .tasks import modadd_split


def train_modadd(out_dir: str | Path, p: int = 113, train_frac: float = 0.3, split_seed: int = 0, seed: int = 0,
                 d_model: int = 128, n_heads: int = 4, d_mlp: int = 512, lr: float = 1e-3, weight_decay: float = 1.0,
                 max_epochs: int = 40000, stop_test_loss: float = 1e-5, device: str = "cuda:0", log_every: int = 100) -> dict:
    """Train a 1-layer transformer on (a + b) mod p with full-batch AdamW and heavy weight decay.

    This is the standard grokking setup: the model first memorises the training pairs, then
    generalises to the held-out pairs much later. Saves ``model.pt`` and ``training.json``.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(n_layers=1, n_heads=n_heads, d_model=d_model, d_head=d_model // n_heads, d_mlp=d_mlp, act_fn="relu",
               normalization_type=None, d_vocab=p + 1, d_vocab_out=p, n_ctx=3, init_weights=True, seed=seed)
    model = HookedTransformer(HookedTransformerConfig(**cfg)).to(device)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False          # biases off, as in the reference setup

    train, test = modadd_split(p, train_frac, split_seed)

    def tensors(pairs):
        x = torch.tensor([[a, b, p] for a, b in pairs], device=device)
        y = torch.tensor([(a + b) % p for a, b in pairs], device=device)
        return x, y

    (xtr, ytr), (xte, yte) = tensors(train), tensors(test)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / 10, 1))

    def loss_fn(x, y):
        logits = model(x)[:, -1].to(torch.float64)
        return -logits.log_softmax(-1).gather(1, y[:, None]).mean()

    hist = {"epoch": [], "train_loss": [], "test_loss": [], "test_acc": []}
    t0 = time.time()
    for epoch in range(max_epochs + 1):
        loss = loss_fn(xtr, ytr)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if epoch % log_every == 0:
            with torch.no_grad():
                te = float(loss_fn(xte, yte))
                acc = float((model(xte)[:, -1].argmax(-1) == yte).float().mean())
            hist["epoch"].append(epoch)
            hist["train_loss"].append(loss.item())
            hist["test_loss"].append(te)
            hist["test_acc"].append(acc)
            if te < stop_test_loss:
                break
    meta = {"task": "modadd", "p": p, "train_frac": train_frac, "split_seed": split_seed, "seed": seed,
            "epochs": epoch, "final_train_loss": hist["train_loss"][-1], "final_test_loss": hist["test_loss"][-1],
            "final_test_acc": hist["test_acc"][-1], "n_train": len(train), "n_test": len(test),
            "train_seconds": round(time.time() - t0, 1),
            "device": torch.cuda.get_device_name(0) if "cuda" in device else "cpu",
            "optimizer": f"AdamW lr={lr} wd={weight_decay} betas=(0.9,0.98), full batch"}
    torch.save({"cfg": cfg, "state_dict": model.state_dict(), "meta": meta}, out / "model.pt")
    (out / "training.json").write_text(json.dumps({"meta": meta, "history": hist}), encoding="utf-8")
    return meta
