"""glassbox-bench: planted circuits with an exact, sealed answer key.

``generate`` makes a task set from a seed, ``run`` investigates it blind with a headless scientist, and
``grade`` (run by the user afterwards) scores the findings against the sealed keys.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def behavior(task: dict, rel: str) -> str:
    """What the scientist is told about a task. It says what the inputs look like, never what the function is."""
    k = len(task["input_token_ids"])
    text = (f"This small transformer was built for a benchmark. Its input is a BOS token followed by exactly {task['seq_len']} "
            f"tokens drawn from {k} symbols, and at the last position it outputs one symbol. The function it computes is "
            f"deterministic and is published nowhere: work out from the model what it is. Then find which attention heads and "
            f"MLP layers implement it. The model also contains components that play no part in the behaviour. "
            f"Use the dataset task 'bench' with path '{rel}' (plus n and seed); its answers are the model's own outputs and the "
            f"token strings are in its meta. Your predict(tokens, prompt) receives the full list of token ids including the BOS "
            f"and must return the id of the output token. In verify_hypothesis give the circuit as a list of heads 'L.H' and "
            f"MLP layers 'mlpL'.")
    if task["track"] == "trained":
        text += " This model was trained by gradient descent, so it may be imperfect and its mechanism may be messy."
    return text


def run(seed: int, only: list[str] | None = None, budget: int = 25, out_root: Path | str | None = None) -> list[dict]:
    """Investigate every task of a seed with a headless scientist (one after the other)."""
    from .. import headless

    out_root = Path(out_root) if out_root else ROOT / "benchmarks" / "glassbox-bench"
    base = out_root / f"seed-{seed}"
    done = []
    for t in json.loads((base / "tasks.json").read_text(encoding="utf-8"))["tasks"]:
        tid = t["id"]
        if only and tid not in only:
            continue
        if (base / "runs" / tid / "session.json").exists():
            continue
        task = json.loads((base / "tasks" / tid / "task.json").read_text(encoding="utf-8"))
        rel = (base / "tasks" / tid).resolve().relative_to(ROOT).as_posix()
        prompt, args = headless.build_prompt("local-full", f"checkpoint:{rel}/model.pt", behavior(task, rel), budget, tid)
        s = headless.run(prompt, base / "runs", tid, label=f"glassbox-bench seed {seed} {tid}", prompt_args=args)
        done.append({"task": tid, "seconds": s["wall_clock_seconds"], "turns": s["turns"], "error": s["is_error"]})
    return done
