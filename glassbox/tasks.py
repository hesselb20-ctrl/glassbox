"""Behaviour datasets: clean inputs, matched corrupted inputs and a metric.

A task defines *what the model does*, never how. Every generator is seeded, so a
held-out set is just a different seed.
"""
from __future__ import annotations

import random
from typing import Any

NAMES = [
    "Mary", "John", "Tom", "James", "Dan", "Sarah", "Anna", "Mark", "Paul", "Kate", "Lisa", "David",
    "Michael", "Robert", "Jennifer", "Laura", "Alex", "Emma", "Jack", "Ryan", "Amy", "George", "Peter",
    "Alice", "Kevin", "Brian", "Susan", "Rachel", "Steve", "Jason", "Maria", "Simon", "Jane", "Adam",
]
PLACES = ["store", "park", "school", "office", "garden", "house", "restaurant", "station", "hospital", "library"]
OBJECTS = ["drink", "book", "ring", "kiss", "bone", "basketball", "computer", "necklace", "letter", "gift"]
IOI_TEMPLATES = [
    "When{A} and{B} went to the{P},{C} gave a{O} to",
    "After{A} and{B} arrived at the{P},{C} handed a{O} to",
    "Then,{A} and{B} walked to the{P}.{C} offered a{O} to",
]


def _single_token(backend, words: list[str]) -> list[str]:
    keep = []
    for w in words:
        try:
            backend.token_id(" " + w)
            keep.append(w)
        except ValueError:
            pass
    return keep


def ioi(backend, n: int = 64, seed: int = 0, template: int = 0, corruption: str = "abc", **_: Any) -> dict:
    """Indirect object identification. ``"When Mary and John went to the store, John gave a drink to"`` -> `` Mary``.

    corruption ``abc``: three unrelated names (no repeated subject). ``flip``: swap which name is repeated.
    """
    rng = random.Random(seed)
    names, places, objects = (_single_token(backend, x) for x in (NAMES, PLACES, OBJECTS))
    tpl = IOI_TEMPLATES[template]
    clean, corrupt, io_tok, s_tok, order = [], [], [], [], []
    for i in range(n):
        io, s = rng.sample(names, 2)
        p, o = rng.choice(places), rng.choice(objects)
        first, second = (io, s) if i % 2 == 0 else (s, io)
        fill = dict(P=" " + p, O=" " + o)
        clean.append(tpl.format(A=" " + first, B=" " + second, C=" " + s, **fill))
        if corruption == "abc":
            a, b, c = rng.sample([x for x in names if x not in (io, s)], 3)
            corrupt.append(tpl.format(A=" " + a, B=" " + b, C=" " + c, **fill))
        elif corruption == "flip":
            corrupt.append(tpl.format(A=" " + first, B=" " + second, C=" " + io, **fill))
        else:
            raise ValueError("corruption must be 'abc' or 'flip'")
        io_tok.append(" " + io)
        s_tok.append(" " + s)
        order.append("IO-first" if i % 2 == 0 else "S-first")
    toks = backend.encode(clean)
    positions = []
    for row, io, s in zip(toks.tolist(), io_tok, s_tok):
        io_id, s_id = backend.token_id(io), backend.token_id(s)
        s_pos = [j for j, t in enumerate(row) if t == s_id]
        positions.append({"IO": row.index(io_id), "S1": s_pos[0], "S2": s_pos[1], "END": len(row) - 1})
    return {
        "task": "ioi",
        "description": "Complete the sentence with the name that was NOT repeated (the indirect object).",
        "clean": {"prompts": clean},
        "corrupt": {"prompts": corrupt},
        "metric": {"type": "logit_diff", "correct": io_tok, "incorrect": s_tok},
        "answers": io_tok,
        "meta": {"positions": positions, "name_order": order, "template": tpl, "corruption": corruption,
                 "token_strs": backend.decode(toks[0].tolist())},
    }


def induction(backend, n: int = 64, seed: int = 0, seq_len: int = 12, **_: Any) -> dict:
    """Repeated random tokens: ``[bos] r1..rN r1..r(N-1)`` -> ``rN``. Corrupted run breaks the repeat."""
    rng = random.Random(seed)
    info = backend.info()
    lo, hi = min(1000, info["d_vocab"] // 10), min(info["d_vocab"], 40000)
    bos = backend.bos_id()
    clean, corrupt, target = [], [], []
    for _ in range(n):
        seq = rng.sample(range(lo, hi), seq_len)
        other = rng.sample([t for t in range(lo, hi) if t not in seq][:5000], seq_len)
        head = [bos] if bos is not None else []
        clean.append(head + seq + seq[:-1])
        corrupt.append(head + other + seq[:-1])
        target.append(seq[-1])
    return {
        "task": "induction",
        "description": "A random token sequence is shown and then repeated; predict the next token of the repeat.",
        "clean": {"tokens": clean},
        "corrupt": {"tokens": corrupt},
        "metric": {"type": "logprob", "target": target},
        "answers": target,
        "meta": {"seq_len": seq_len, "has_bos": bos is not None, "n_positions": len(clean[0]),
                 "layout": "[bos?] + seq(seq_len) + seq[:-1]; corrupt replaces the first copy with other random tokens"},
    }


def modadd_split(p: int, train_frac: float, split_seed: int) -> tuple[list, list]:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    random.Random(split_seed).shuffle(pairs)
    k = int(train_frac * len(pairs))
    return pairs[:k], pairs[k:]


def modadd(backend=None, n: int | str = 256, seed: int = 0, p: int = 113, split: str = "all",
           train_frac: float = 0.3, split_seed: int = 0, **_: Any) -> dict:
    """Modular addition: tokens ``[a, b, =]`` -> ``(a + b) mod p``. ``n="all"`` gives the full p*p grid in order."""
    train, test = modadd_split(p, train_frac, split_seed)
    pool = {"all": [(a, b) for a in range(p) for b in range(p)], "train": train, "test": test}[split]
    rng = random.Random(seed)
    pairs = pool if n == "all" else rng.sample(pool, min(int(n), len(pool)))
    clean, corrupt, target, wrong = [], [], [], []
    for a, b in pairs:
        while True:
            a2, b2 = rng.randrange(p), rng.randrange(p)
            if (a2 + b2) % p != (a + b) % p:
                break
        clean.append([a, b, p])
        corrupt.append([a2, b2, p])
        target.append((a + b) % p)
        wrong.append((a2 + b2) % p)
    return {
        "task": "modadd",
        "description": f"Input tokens [a, b, '='] with a, b in 0..{p - 1}; the model outputs (a + b) mod {p} at the last position.",
        "clean": {"tokens": clean},
        "corrupt": {"tokens": corrupt},
        "metric": {"type": "logit_diff", "correct": target, "incorrect": wrong},
        "answers": target,
        "meta": {"p": p, "split": split, "n": len(pairs), "layout": "positions: 0=a, 1=b, 2='=' (token id p)"},
    }


_BENCH_MODELS: dict[str, Any] = {}


def bench(backend=None, path: str = "", n: int = 128, seed: int = 0, **_: Any) -> dict:
    """A glassbox-bench task: random valid inputs, with the model's own outputs as the answers.

    The answers come from the task's published checkpoint, loaded separately on the CPU, not from whatever model
    the backend is running. So a null version of the model is scored against the real behaviour, as it should be.
    Only the public task folder is read here; answer keys live elsewhere and this code never looks for them.
    """
    import json
    from pathlib import Path

    import torch

    d = Path(path)
    if ".glassbox-sealed" in d.as_posix():
        raise PermissionError("the sealed answer keys are off limits")
    task = json.loads((d / "task.json").read_text(encoding="utf-8"))
    if path not in _BENCH_MODELS:
        from .bench.generate import build_model

        ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        _BENCH_MODELS[path] = build_model(ckpt["cfg"], ckpt["state_dict"])
    model = _BENCH_MODELS[path]
    rng = random.Random(seed)
    ids, L = task["input_token_ids"], task["seq_len"]

    def draw(k: int) -> list[list[int]]:
        return [[task["bos_id"]] + [rng.choice(ids) for _ in range(L)] for _ in range(k)]

    def out(xs: list[list[int]]) -> list[int]:
        with torch.no_grad():
            return model(torch.tensor(xs))[:, -1].argmax(-1).tolist()

    clean = draw(int(n))
    ans = out(clean)
    corrupt, wrong = [], []
    for a in ans:                                   # a corrupted input is another input with a different answer
        for _ in range(200):
            c = draw(1)[0]
            w = out([c])[0]
            if w != a:
                break
        corrupt.append(c)
        wrong.append(w)
    vocab = task["vocab"]
    return {
        "task": "bench",
        "description": f"Input: BOS + {L} tokens; the model outputs one token at the last position. What it computes is for you to find out.",
        "clean": {"tokens": clean}, "corrupt": {"tokens": corrupt},
        "metric": {"type": "logit_diff", "correct": ans, "incorrect": wrong},
        "answers": ans,
        "meta": {"seq_len": L, "vocab": vocab, "input_token_ids": ids, "layout": task["layout"],
                 "examples_as_strings": [{"input": [vocab[i] for i in x], "output": vocab[a]} for x, a in zip(clean[:12], ans[:12])]},
    }


TASKS = {"ioi": ioi, "induction": induction, "modadd": modadd, "bench": bench}


def make(backend, spec: dict) -> dict:
    spec = dict(spec)
    task = spec.pop("task")
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}'. Known: {sorted(TASKS)}. Or pass your own clean/corrupt/metric.")
    return TASKS[task](backend, **spec)
