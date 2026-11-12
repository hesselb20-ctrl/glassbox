"""Lab plumbing shared by every primitive: data, metrics, batching, artifacts, records."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from .. import tasks
from ..backends.base import Backend, Edit

# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #

class Metric:
    """A per-example scalar read off the logits at the answer position.

    spec ``{"type": "logit_diff", "correct": [...], "incorrect": [...]}`` or
    ``{"type": "logit" | "prob" | "logprob" | "accuracy", "target": [...]}``.
    Tokens may be ids or single-token strings; one value is broadcast to all examples.
    """

    def __init__(self, spec: dict, backend: Backend, n: int) -> None:
        self.spec = spec
        self.type = spec["type"]

        def ids(key: str) -> torch.Tensor:
            v = spec[key]
            v = v if isinstance(v, (list, tuple)) else [v]
            if len(v) == 1:
                v = list(v) * n
            if len(v) != n:
                raise ValueError(f"metric '{key}' has {len(v)} entries for {n} examples")
            return torch.tensor([backend.token_id(x) for x in v])

        if self.type == "logit_diff":
            self.correct, self.wrong = ids("correct"), ids("incorrect")
        elif self.type in ("logit", "prob", "logprob", "accuracy"):
            self.correct, self.wrong = ids("target"), None
        else:
            raise ValueError(f"unknown metric type '{self.type}'")

    def __call__(self, logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        c = self.correct[idx.cpu()].to(logits.device)[:, None]
        if self.type == "logit_diff":
            w = self.wrong[idx.cpu()].to(logits.device)[:, None]
            return (logits.gather(1, c) - logits.gather(1, w))[:, 0]
        if self.type == "logit":
            return logits.gather(1, c)[:, 0]
        if self.type == "prob":
            return logits.softmax(-1).gather(1, c)[:, 0]
        if self.type == "logprob":
            return logits.log_softmax(-1).gather(1, c)[:, 0]
        return (logits.argmax(-1) == c[:, 0]).float()


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Data:
    clean: Any
    corrupt: Any
    metric: Metric | None
    meta: dict
    n: int


@dataclasses.dataclass
class Out:
    metric: torch.Tensor | None
    acts: dict[str, torch.Tensor]
    grads: dict[str, torch.Tensor]
    logits: torch.Tensor | None


def parse_head(s: str | Sequence[int]) -> tuple[int, int]:
    if isinstance(s, str):
        a, b = s.split(".")
        return int(a), int(b)
    return int(s[0]), int(s[1])


def parse_component(c: str) -> tuple[str, int, int | None]:
    """``"9.6"`` -> head, ``"mlp3"`` -> MLP, ``"attn3"`` -> whole attention layer."""
    c = c.strip()
    if c.startswith("mlp"):
        return ("mlp", int(c[3:]), None)
    if c.startswith("attn"):
        return ("attn", int(c[4:]), None)
    layer, head = parse_head(c)
    return ("head", layer, head)


def component_site(kind: str, layer: int) -> str:
    return {"head": f"head_out.{layer}", "mlp": f"mlp_out.{layer}", "attn": f"attn_out.{layer}"}[kind]


def _jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().float().cpu().tolist())
    if isinstance(x, float):
        return round(x, 6) if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return _jsonable(x.item())
        except Exception:
            return str(x)
    return x


# --------------------------------------------------------------------------- #
# Lab
# --------------------------------------------------------------------------- #

class Lab:
    """Runs primitives on one backend and writes every result to disk."""

    def __init__(self, backend: Backend, out_dir: str | Path, batch_size: int = 64) -> None:
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.failed_variants: list[str] = []
        (self.out_dir / "experiments").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # -- ids ---------------------------------------------------------------- #
    def _next(self, folder: str, prefix: str) -> str:
        nums = [int(p.name[1:5]) for p in (self.out_dir / folder).glob(f"{prefix}[0-9][0-9][0-9][0-9]*")]
        return f"{prefix}{(max(nums) + 1 if nums else 1):04d}"

    # -- artifacts ---------------------------------------------------------- #
    def save_artifact(self, name: str, obj: Any) -> str:
        aid = f"{self._next('artifacts', 'A')}_{name}"
        torch.save(obj, self.out_dir / "artifacts" / f"{aid}.pt")
        return aid

    def load_artifact(self, aid: str) -> Any:
        path = self.out_dir / "artifacts" / f"{aid}.pt"
        if not path.exists():
            known = sorted(p.stem for p in (self.out_dir / "artifacts").glob("*.pt"))
            raise KeyError(f"no artifact '{aid}'. Known: {known}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def direction(self, spec: Any, site: str | None = None) -> torch.Tensor:
        """Resolve ``{"artifact": id, "key": site}`` or a raw list into a [d] vector."""
        if isinstance(spec, dict) and "artifact" in spec:
            art = self.load_artifact(spec["artifact"])
            dirs = art["directions"] if isinstance(art, dict) and "directions" in art else art
            if isinstance(dirs, dict):
                key = spec.get("key") or site
                if key not in dirs:
                    raise KeyError(f"artifact has directions for {sorted(dirs)}; asked for '{key}'")
                return dirs[key].float()
            return dirs.float()
        return torch.tensor(spec, dtype=torch.float32)

    def subspace_edit(self, spec: dict) -> Edit:
        """``{"artifact": id, "site": ..., "op": "project_out"|"keep_only", "pos": [...], "groups": [...]}`` -> Edit."""
        art = self.load_artifact(spec["artifact"])
        basis, mean = art["basis"].float(), art.get("mean")
        if spec.get("groups") is not None:
            rows = [i for g in spec["groups"] for i in art["groups"][str(g)]]
            basis = torch.linalg.qr(art["raw"][rows].float().T).Q.T
        return Edit(spec.get("site", art.get("site")), spec.get("op", "project_out"),
                    {"basis": basis, "mean": mean}, spec.get("pos"))

    # -- data --------------------------------------------------------------- #
    def tokens(self, spec: Any, allow_ragged: bool = False) -> Any:
        if isinstance(spec, torch.Tensor):
            return spec
        if isinstance(spec, dict):
            spec = spec.get("tokens", spec.get("prompts"))
        if spec is None or len(spec) == 0:
            raise ValueError("no inputs given: pass {'prompts': [...]} or {'tokens': [[...]]}")
        if isinstance(spec[0], str):
            rows = [self.backend.encode([p])[0] for p in spec]
        else:
            rows = [torch.tensor(r, dtype=torch.long) for r in spec]
        if len({len(r) for r in rows}) == 1:
            return torch.stack(rows)
        if not allow_ragged:
            raise ValueError(
                f"inputs have different token lengths {sorted({len(r) for r in rows})}; "
                "this primitive needs position-aligned inputs"
            )
        return rows

    def data(self, params: dict, need_corrupt: bool = True, need_metric: bool = True,
             allow_ragged: bool = False) -> Data:
        ds = tasks.make(self.backend, params["dataset"]) if "dataset" in params else params
        if "clean" not in ds:
            raise ValueError("pass either 'dataset': {'task': ...} or explicit 'clean' (and 'corrupt') inputs")
        clean = self.tokens(ds["clean"], allow_ragged)
        n = len(clean)
        corrupt = None
        if ds.get("corrupt") is not None:
            corrupt = self.tokens(ds["corrupt"], allow_ragged)
            if not allow_ragged and corrupt.shape != clean.shape:
                raise ValueError(f"clean {tuple(clean.shape)} and corrupt {tuple(corrupt.shape)} must align")
        elif need_corrupt:
            raise ValueError("this primitive needs 'corrupt' inputs as well as 'clean'")
        mspec = params.get("metric") or ds.get("metric")
        if mspec is None and need_metric:
            raise ValueError("this primitive needs a 'metric' (e.g. {'type': 'logit_diff', 'correct': [...], 'incorrect': [...]})")
        metric = Metric(mspec, self.backend, n) if mspec else None
        return Data(clean, corrupt, metric, ds.get("meta", {}), n)

    # -- batched forward ---------------------------------------------------- #
    def fwd(
        self,
        tokens: Any,
        read: Sequence[str] = (),
        edits: Sequence[Edit] = (),
        read_pos: int | None = None,
        metric: Metric | None = None,
        grad_sites: Sequence[str] = (),
        keep_logits: bool = False,
        batch_size: int | None = None,
    ) -> Out:
        """Run inputs in batches (grouped by length if ragged) and stitch the results back in order."""
        n = len(tokens)
        bs = batch_size or self.batch_size
        ragged = not isinstance(tokens, torch.Tensor)
        if ragged and read and read_pos is None:
            raise ValueError("ragged inputs can only be read at a fixed position (e.g. pos=-1)")
        if ragged:
            by_len: dict[int, list[int]] = {}
            for i, r in enumerate(tokens):
                by_len.setdefault(len(r), []).append(i)
            groups = [g[i:i + bs] for g in by_len.values() for i in range(0, len(g), bs)]
        else:
            groups = [list(range(i, min(n, i + bs))) for i in range(0, n, bs)]

        m_out = torch.zeros(n) if metric is not None else None
        acts: dict[str, list] = {s: [None] * len(groups) for s in read}
        grads: dict[str, list] = {s: [None] * len(groups) for s in grad_sites}
        logits_out: list = [None] * len(groups)
        for gi, g in enumerate(groups):
            idx = torch.tensor(g)
            batch = torch.stack([tokens[i] for i in g]) if ragged else tokens[idx]
            batch_edits = []
            for e in edits:
                v = e.value
                if isinstance(v, torch.Tensor) and n > 1 and v.shape[0] == n:
                    e = dataclasses.replace(e, value=v[idx.to(v.device)])
                batch_edits.append(e)
            gm = (lambda lg, idx=idx: metric(lg, idx)) if grad_sites else None
            run = self.backend.forward(batch, read=read, edits=batch_edits, read_pos=read_pos,
                                       grad_metric=gm, grad_sites=grad_sites)
            if metric is not None:
                m_out[idx] = metric(run.logits, idx).detach().float().cpu()
            for s in read:
                acts[s][gi] = run.acts[s]
            for s in grad_sites:
                grads[s][gi] = run.grads[s]
            if keep_logits:
                logits_out[gi] = run.logits.float().cpu()
        order = torch.tensor([i for g in groups for i in g]).argsort()

        def stitch(parts: list) -> torch.Tensor:
            cat = torch.cat(parts, dim=0)
            return cat[order.to(cat.device)]

        return Out(
            metric=m_out,
            acts={s: stitch(v) for s, v in acts.items()},
            grads={s: stitch(v) for s, v in grads.items()},
            logits=stitch(logits_out) if keep_logits else None,
        )

    def fwd_variants(self, tokens: torch.Tensor, variants: Sequence[Sequence[Edit]], metric: Metric,
                     source: torch.Tensor | None = None, max_rows: int | None = None) -> torch.Tensor:
        """Mean metric for each of several edit sets, packed into as few forward passes as possible.

        The batch is tiled once per variant and every edit is confined to its own block of rows. If ``source`` is
        given it rides along as one extra block, and edits with op ``copy`` take their value from it. So a whole
        patching sweep is a few forward passes with nothing uploaded, which is what makes it practical on NDIF.
        """
        n = len(tokens)
        max_rows = max_rows or getattr(self.backend, "max_rows", 512)
        per = max(1, max_rows // n - (1 if source is not None else 0))
        out = []
        for start in range(0, len(variants), per):
            chunk = variants[start:start + per]
            k = len(chunk)
            src_rows = list(range(k * n, (k + 1) * n))
            edits = []
            for j, es in enumerate(chunk):
                for e in es:
                    rows = list(range(j * n, (j + 1) * n))
                    edits.append(dataclasses.replace(e, rows=rows, value=src_rows if e.op == "copy" else e.value))
            batch = tokens.repeat(k, 1)
            if source is not None:
                batch = torch.cat([batch, source])
            try:
                run = self.backend.forward(batch, edits=edits)
            except Exception as err:  # noqa: BLE001
                if k == 1:                      # this one variant cannot be run on this backend/deployment: mark it, keep going
                    self.failed_variants.append(str(err).strip().splitlines()[-1][:200])
                    out.append(torch.full((1,), float("nan")))
                    continue
                out.append(torch.cat([self.fwd_variants(tokens, [v], metric, source, max_rows) for v in chunk]))
                continue
            m = metric(run.logits[: k * n], torch.arange(n).repeat(k)).detach().float().cpu()
            out.append(m.reshape(k, n).mean(1))
        return torch.cat(out)

    # -- ablation helper (shared with verify) -------------------------------- #
    def ablation_edits(self, comps: Sequence[tuple], mode: str, ref: dict[str, torch.Tensor] | None,
                       pos: Any = None, seed: int = 0) -> list[Edit]:
        edits = []
        gen = torch.Generator().manual_seed(seed)
        for kind, layer, head in comps:
            site = component_site(kind, layer)
            if mode == "zero":
                edits.append(Edit(site, "zero", None, pos, head))
                continue
            a = ref[site]
            if mode == "mean":
                v = a.mean(0, keepdim=True)
            elif mode == "resample":
                v = a[torch.randperm(a.shape[0], generator=gen).to(a.device)]
            else:
                raise ValueError("ablation mode must be 'mean', 'zero' or 'resample'")
            if head is not None:
                v = v[:, :, head]                 # only this head's slice travels (matters for remote backends)
            edits.append(Edit(site, "replace", v, pos, head))
        return edits

    def all_components(self, heads: bool = True, mlps: bool = True) -> list[tuple]:
        info = self.backend.info()
        out: list[tuple] = []
        for layer in range(info["n_layers"]):
            if heads:
                out += [("head", layer, h) for h in range(info["n_heads"])]
            if mlps and not info.get("attn_only"):
                out.append(("mlp", layer, None))
        return out

    # -- run a primitive ---------------------------------------------------- #
    def run(self, primitive: str, params: dict | None = None) -> dict:
        from . import REGISTRY  # late import: primitives import this module

        params = params or {}
        if primitive not in REGISTRY:
            raise KeyError(f"unknown primitive '{primitive}'. Known: {sorted(REGISTRY)}")
        self.backend.require(primitive)
        t0 = time.time()
        summary, data, fig = REGISTRY[primitive](self, params)
        eid = self._next("experiments", "E")
        plot = None
        if fig is not None:
            import matplotlib.pyplot as plt

            plot = f"experiments/{eid}_{primitive}.png"      # relative to the investigation folder
            fig.savefig(self.out_dir / plot)
            plt.close(fig)
        record = {
            "id": eid,
            "primitive": primitive,
            "provenance": self.backend.provenance(),
            "params": _jsonable(params),
            "summary": _jsonable(summary),
            "data": _jsonable(data),
            "plot": plot,
            "seconds": round(time.time() - t0, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.out_dir / "experiments" / f"{eid}_{primitive}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return record

    def load_experiment(self, eid: str) -> dict:
        hits = sorted((self.out_dir / "experiments").glob(f"{eid}_*.json"))
        if not hits:
            raise KeyError(f"no experiment '{eid}'")
        return json.loads(hits[0].read_text(encoding="utf-8"))

    def experiment_ids(self) -> list[str]:
        return sorted(p.name[:5] for p in (self.out_dir / "experiments").glob("E*.json"))
