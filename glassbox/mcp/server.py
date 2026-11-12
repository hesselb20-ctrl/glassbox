"""The glassbox MCP server: the lab bench an AI scientist works at.

Run with ``python -m glassbox.mcp`` (stdio). Runs are written under ``GLASSBOX_HOME``
(default ``./glassbox_runs``). The NDIF key, if you use NDIF, is read only from the
``NDIF_API_KEY`` environment variable of this process and is never logged or returned.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any

try:  # MCP Python SDK v2
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # SDK v1
    from mcp.server.fastmcp import FastMCP as _Server, Image

from ..session import Session

INSTRUCTIONS = """Glassbox is an interpretability lab. You are the scientist.
Loop: notebook_read -> hypothesis (notebook_append) -> run_experiment -> read the numbers and the plot ->
record the result and update the hypothesis -> repeat. Then verify_hypothesis on held-out inputs and write_report.
Rules: every claim must cite experiments that ran (ids like E0007). Never state a mechanism from memory.
Results from quantized backends are screening only; a finding is verified only at full precision."""

server = _Server("glassbox", instructions=INSTRUCTIONS)
session = Session()


def _safe(fn, *args: Any, **kwargs: Any) -> str:
    """Run a tool and always answer with JSON; errors come back as a message the scientist can act on."""
    try:
        return json.dumps(fn(*args, **kwargs), indent=1, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        return json.dumps({"error": type(e).__name__, "message": str(e)}, indent=1)


@server.tool()
def list_models(backend: str | None = None) -> str:
    """List the models each backend can load right now (local-full, local-gguf, local-4bit, ndif),
    whether NDIF is reachable, GPU memory, and whether another local model server is occupying the GPUs."""
    return _safe(session.list_models, backend)


@server.tool()
def load_model(backend: str, model: str, investigation: str | None = None, options: dict | None = None,
               budget: int | None = None) -> str:
    """Load a model and open an investigation (a folder with its own notebook). Re-using an investigation name resumes it.

    backend: local-full | local-gguf | local-4bit | ndif.   model: an id from list_models.
    budget: optional maximum number of experiments for this investigation."""
    return _safe(session.load_model, backend, model, investigation, options, budget)


@server.tool()
def backend_capabilities() -> str:
    """Which experiment primitives each backend supports, and whether its results count as verification
    (full precision) or only as screening (quantized)."""
    return _safe(session.backend_capabilities)


@server.tool()
def run_experiment(primitive: str, params: dict | None = None, note: str = "", detail: str = "auto") -> str:
    """Run one experiment. Returns structured JSON (summary + data) and saves a plot (see get_plot).
    The run is logged in the notebook automatically with its id (E0001, ...).

    Primitives:
      dataset                 {"task": "ioi"|"induction"|"modadd", "n": 64, "seed": 0}: inspect a behaviour dataset
      baseline                does the model do the behaviour at all? metric + accuracy
      attribution_patching    cheap first screen of every head / layer / position (gradient estimate)
      activation_patching     exact causal effect; component: resid_pre | resid_post | attn_out | mlp_out | head
      path_patching           senders -> receiver: "logits" or {"heads": ["9.9"], "input": "q"|"k"|"v"}
      ablation                components ["9.6","mlp0"] or sweep "heads"|"mlps"|"attn"; mode mean|zero|resample; joint;
                              or subspace {"artifact": id, "site": "embed", "pos": [0, 1], "op": "project_out" | "keep_only"}
      logit_lens              decode the residual stream after every layer
      direct_logit_attribution  each head's direct write along the answer direction
      linear_probe            data {"prompts"|"tokens": [...]} + labels [one per input] -> per-layer held-out accuracy (+ control)
      attention_patterns      query_pos / key_pos / offset / head "L.H"
      diff_of_means           positive vs negative prompts -> a direction per layer (saved as an artifact)
      steering                add a direction at sites with strengths alphas; metric and/or generated text before/after
      periodicity             Fourier analysis over an integer input: source "W_E"|"W_U"|"<site>", period
      weights_svd             singular value spectrum of W_E / W_U

    Data: pass "dataset": {"task": ..., "n": ..., "seed": ...} or explicit
    "clean": {"prompts": [...]}, "corrupt": {"prompts": [...]}, "metric": {"type": "logit_diff", "correct": [...], "incorrect": [...]}.
    If a primitive is not supported on the loaded backend the error names a backend that supports it."""
    return _safe(session.run_experiment, primitive, params, note, detail)


@server.tool()
def get_plot(experiment_id: str) -> Image:
    """Return the plot (PNG) of an experiment or verification, e.g. "E0007" or "V0001"."""
    return Image(path=session.get_plot(experiment_id))


@server.tool()
def notebook_append(kind: str, text: str, refs: list[str] | None = None, status: str | None = None,
                    target: str | None = None) -> str:
    """Write in the lab notebook. kind: hypothesis | result | decision | note.
    refs: the experiment ids (E0007) this entry rests on; required for a result.
    To update a hypothesis, pass target=<its id> and status = supported | rejected | revised, with the reason as text."""
    return _safe(session.notebook_append, kind, text, refs, status, target)


@server.tool()
def notebook_read(investigation: str | None = None, last_n: int | None = None, kinds: list[str] | None = None) -> str:
    """Read the notebook: hypotheses with their current status, experiments, results, decisions.
    Pass investigation=<name> to resume an earlier investigation; the model reloads on the next experiment."""
    return _safe(session.notebook_read, investigation, last_n, kinds)


@server.tool()
def verify_hypothesis(spec: dict) -> str:
    """Test the recovered algorithm on held-out inputs.

    spec = {
      "name": "...",
      "dataset": {"task": ..., "n": 200, "seed": <a seed you have NOT used>}   (or clean/corrupt/metric),
      "code": "def predict(tokens, prompt):\\n    ...  # return the predicted next token (id or string)",
      "circuit": {"components": ["9.9", "mlp0", ...], "mode": "mean", "reference": "corrupt"}
                 and/or {"subspaces": [{"artifact": "A0001_fourier_subspace", "site": "embed", "pos": [0, 1]}]},
      "thresholds": {...optional overrides...}
    }
    Checks (1) agreement between predict() and the real model, (2) causal: ablate the circuit -> behaviour breaks;
    ablate everything else -> it survives; plus a random control. Status is "verified" only at full precision;
    quantized backends can reach "screening-pass" at most."""
    return _safe(session.verify_hypothesis, spec)


@server.tool()
def write_report(report: dict) -> str:
    """Write report.md and report.html for the investigation.

    report = {"title", "behavior", "summary", "pseudocode",
              "claims": [{"text": "...", "evidence": ["E0003", "V0001"]}],     # every claim needs real evidence ids
              "circuit": {"nodes": [{"id", "label", "layer", "kind"}], "edges": [{"src", "dst", "label"}]},
              "key_plots": ["E0003", ...], "limitations": [...], "open_questions": [...], "label": "optional status label"}
    The call fails if a claim cites nothing or cites an experiment that does not exist."""
    return _safe(session.write_report, report)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
