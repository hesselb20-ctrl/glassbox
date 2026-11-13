"""Generate the README images from committed run records. No image APIs: matplotlib and hand-written SVG.

    python media/make_assets.py            # everything whose data exists
    python media/make_assets.py fourier    # one asset

Every number drawn here is read from benchmarks/, demos/ or ladder/. Assets whose runs do not exist are skipped.
"""
from __future__ import annotations

import html
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glassbox import style as S  # noqa: E402
from glassbox.backends.base import BACKENDS, PRIMITIVES, capability_table  # noqa: E402

S.apply_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

MEDIA = ROOT / "media"
MONO = "Consolas, 'Cascadia Mono', 'DejaVu Sans Mono', monospace"
SANS = "Inter, 'Segoe UI', system-ui, sans-serif"


def rec(folder: str, eid: str) -> dict:
    return json.loads(next((ROOT / folder / "experiments").glob(f"{eid}_*.json")).read_text(encoding="utf-8"))


def exists(*parts: str) -> bool:
    return (ROOT.joinpath(*parts)).exists()


# --------------------------------------------------------------------------- #
# SVG helpers
# --------------------------------------------------------------------------- #

class Svg:
    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">',
            f'<defs><filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="4" result="b"/>'
            f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
            f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{S.ACCENT}"/></marker>'
            f'<marker id="arrm" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{S.MUTED}"/></marker></defs>',
            f'<rect width="{w}" height="{h}" fill="{S.NAVY}" rx="14"/>',
        ]

    def box(self, x, y, w, h, stroke=S.GRID, fill=S.PANEL, glow=False, rx=10, sw=1.5):
        g = ' filter="url(#glow)"' if glow else ""
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{g}/>')

    def text(self, x, y, s, size=20, fill=S.TEXT, anchor="start", mono=False, weight=400):
        fam = MONO if mono else SANS
        self.out.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                        f'font-weight="{weight}">{html.escape(str(s))}</text>')

    def lines(self, x, y, items, size=18, fill=S.MUTED, gap=1.45, mono=False, anchor="start"):
        for i, s in enumerate(items):
            self.text(x, y + i * size * gap, s, size, fill, anchor, mono)

    def arrow(self, x1, y1, x2, y2, both=False, muted=False, label=None, dash=False):
        col, m = (S.MUTED, "arrm") if muted else (S.ACCENT, "arr")
        st = f' marker-start="url(#{m})"' if both else ""
        d = ' stroke-dasharray="6 6"' if dash else ""
        g = "" if muted else ' filter="url(#glow)"'
        self.out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.5" marker-end="url(#{m})"{st}{d}{g}/>')
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 10, label, 15, S.MUTED, "middle", mono=True)

    def path(self, d, muted=False):
        col, m = (S.MUTED, "arrm") if muted else (S.ACCENT, "arr")
        self.out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.5" marker-end="url(#{m})" filter="url(#glow)"/>')

    def save(self, name: str) -> None:
        (MEDIA / name).write_text("\n".join(self.out + ["</svg>"]), encoding="utf-8")
        print("wrote", name)


# --------------------------------------------------------------------------- #
# diagrams
# --------------------------------------------------------------------------- #

def how_it_works() -> None:
    s = Svg(1280, 760)
    s.text(48, 62, "How glassbox works", 34, weight=700)
    s.text(48, 96, "Claude Code is the scientist. Glassbox is the lab. Every claim has to come from an experiment that ran.", 19, S.MUTED)
    # top row
    y, h = 140, 250
    s.box(48, y, 270, h, S.ACCENT, glow=True)
    s.text(183, y + 44, "Claude Code", 26, anchor="middle", weight=700)
    s.text(183, y + 74, "the scientist", 18, S.ACCENT, "middle")
    s.lines(72, y + 118, ["forms hypotheses", "designs experiments", "reads results and plots", "rejects and revises"], 18)
    s.text(183, y + 232, "your subscription, no API key", 13.5, S.MUTED, "middle", mono=True)
    s.arrow(326, y + h / 2, 432, y + h / 2, both=True, label="MCP")
    s.box(440, y, 380, h)
    s.text(630, y + 44, "glassbox lab", 26, anchor="middle", weight=700)
    s.text(630, y + 74, "experiment primitives", 18, S.ACCENT, "middle")
    prims = ["attribution patching", "activation patching", "path patching", "ablation", "logit lens", "linear probes",
             "attention patterns", "steering", "difference of means", "Fourier analysis"]
    for i, p in enumerate(prims):
        s.text(460 + (i % 2) * 196, y + 112 + (i // 2) * 27, p, 14.5, S.MUTED, mono=True)
    s.arrow(828, y + h / 2, 904, y + h / 2, both=True)
    names = [("local-full", "TransformerLens, your GPU, full precision"), ("local-gguf", "llama.cpp, any GGUF, read + steer"),
             ("local-4bit", "official weights in 4-bit, 2 GPUs"), ("ndif", "NDIF remote, big models, full precision")]
    for i, (n, d) in enumerate(names):
        yy = y + i * 64
        full = BACKENDS[n]["verifies"]
        s.box(912, yy, 320, 54, S.ACCENT if full else S.GRID)
        s.text(928, yy + 23, n, 18, mono=True, weight=700)
        s.text(928, yy + 43, d, 13.5, S.MUTED)
    s.text(1072, y + 4 * 64 + 12, "accent border = results count as verification", 13.5, S.MUTED, "middle")
    # loop
    ly = 470
    s.text(48, ly, "The research loop", 24, weight=700)
    steps = ["hypothesize", "design experiment", "run", "read results", "update hypothesis"]
    bw, gap, x0 = 200, 38, 48
    for i, st in enumerate(steps):
        x = x0 + i * (bw + gap)
        s.box(x, ly + 26, bw, 58, S.ACCENT if i in (0, 4) else S.GRID)
        s.text(x + bw / 2, ly + 62, st, 18, anchor="middle", mono=True)
        if i < len(steps) - 1:
            s.arrow(x + bw + 4, ly + 55, x + bw + gap - 4, ly + 55)
    xe = x0 + 4 * (bw + gap) + bw / 2
    s.path(f"M{xe},{ly + 88} C{xe},{ly + 150} {x0 + bw / 2},{ly + 150} {x0 + bw / 2},{ly + 92}")
    s.text((xe + x0 + bw / 2) / 2, ly + 128, "repeat until the budget runs out or the story holds", 15, S.MUTED, "middle")
    # outputs
    oy = 650
    outs = [("notebook.jsonl", "every hypothesis, experiment, result and decision; sessions resume"),
            ("verify", "algorithm as code vs the real model on held-out inputs, plus causal ablations"),
            ("report", "pseudocode, circuit, plots, numbers, and which backend produced each one")]
    for i, (n, d) in enumerate(outs):
        x = 48 + i * 400
        s.box(x, oy, 384, 78)
        s.text(x + 18, oy + 30, n, 18, S.ACCENT, mono=True, weight=700)
        for j, line in enumerate(textwrap.wrap(d, 46)):
            s.text(x + 18, oy + 52 + j * 18, line, 13.5, S.MUTED)
    s.save("how-it-works.svg")


def free_stack() -> None:
    s = Svg(1280, 640)
    s.text(48, 62, "Runs for $0", 34, weight=700)
    s.text(48, 96, "No API key, no paid cloud GPUs anywhere in the default path.", 19, S.MUTED)
    cols = [
        ("Scientist", "Claude Code", ["on the subscription you already have", "drives the lab through MCP", "an API mode is optional, never required"], "$0 extra"),
        ("Small models and GGUF", "Your own GPUs", ["tested on RTX 4070 Super 12 GB + RTX 3060 12 GB", "full precision up to about 1-2B parameters",
                                                     "27B via llama.cpp GGUF or 4-bit"], "$0"),
        ("Large models, full precision", "NDIF", ["National Deep Inference Fabric", "remote execution through nnsight", "free for research; you sign up for a key"], "$0"),
    ]
    for i, (kicker, title, lines, price) in enumerate(cols):
        x = 48 + i * 400
        s.box(x, 130, 384, 250, S.ACCENT if i == 0 else S.GRID, glow=(i == 0))
        s.text(x + 22, 166, kicker.upper(), 14, S.MUTED, mono=True)
        s.text(x + 22, 206, title, 28, weight=700)
        s.lines(x + 22, 246, lines, 15.5)
        s.text(x + 22, 352, price, 30, S.ACCENT, mono=True, weight=700)
    s.text(48, 432, "Which backend for what", 24, weight=700)
    rows = [
        ("local-full", "find and verify circuits in small models (GPT-2, Pythia, toy models)", "verified"),
        ("local-gguf", "screen a big quantized model you already have: probes, logit lens, steering", "screening"),
        ("local-4bit", "screen the official big checkpoint with full patching, on two consumer GPUs", "screening"),
        ("ndif", "confirm at full precision on the big model", "verified"),
    ]
    for i, (n, d, st) in enumerate(rows):
        y = 462 + i * 40
        s.text(60, y + 20, n, 18, S.TEXT, mono=True, weight=700)
        s.text(220, y + 20, d, 17, S.MUTED)
        col = S.ACCENT if st == "verified" else S.MUTED
        s.box(1090, y, 142, 30, col, rx=15)
        s.text(1161, y + 21, st, 15, col, "middle", mono=True)
    s.save("free-stack.svg")


def backends() -> None:
    table = capability_table()["primitives"]
    prims = [p for p in PRIMITIVES if p not in ("dataset", "baseline")]
    names = list(BACKENDS)
    row_h, top, x0, cw = 38, 190, 330, 228
    h = top + (len(prims) + 2) * row_h + 40
    s = Svg(1280, h)
    s.text(48, 62, "Four backends, one interface", 34, weight=700)
    s.text(48, 96, "The same experiment code runs on all of them. An unsupported primitive raises an error that names a backend that works.", 17.5, S.MUTED)
    for j, n in enumerate(names):
        x = x0 + j * cw
        s.box(x, 124, cw - 14, 52, S.ACCENT if BACKENDS[n]["verifies"] else S.GRID)
        s.text(x + (cw - 14) / 2, 146, n, 18, anchor="middle", mono=True, weight=700)
        s.text(x + (cw - 14) / 2, 166, BACKENDS[n]["tag"], 12.5, S.MUTED, "middle")
    for i, p in enumerate(prims):
        y = top + i * row_h
        if i % 2 == 0:
            s.box(40, y, 1200, row_h, S.NAVY, "#0b112a", rx=6, sw=0)
        s.text(60, y + 25, p.replace("_", " "), 17, S.TEXT, mono=True)
        for j, n in enumerate(names):
            ok = table[p][n]
            cx = x0 + j * cw + (cw - 14) / 2
            if ok:
                s.out.append(f'<circle cx="{cx}" cy="{y + 19}" r="7" fill="{S.ACCENT}" filter="url(#glow)"/>')
            else:
                s.out.append(f'<line x1="{cx - 8}" y1="{y + 19}" x2="{cx + 8}" y2="{y + 19}" stroke="{S.DIM}" stroke-width="3"/>')
    y = top + len(prims) * row_h + 14
    s.out.append(f'<line x1="40" y1="{y - 6}" x2="1240" y2="{y - 6}" stroke="{S.GRID}" stroke-width="1.5"/>')
    s.text(60, y + 25, "counts as verification", 17, S.TEXT, mono=True, weight=700)
    for j, n in enumerate(names):
        cx = x0 + j * cw + (cw - 14) / 2
        v = BACKENDS[n]["verifies"]
        s.text(cx, y + 25, "yes" if v else "screening only", 16, S.ACCENT if v else S.MUTED, "middle", mono=True)
    s.text(60, y + row_h + 26, "local-gguf v1: reading and steering only, no patching. Gradients are not offered on ndif.", 15, S.MUTED)
    s.save("backends.svg")


# --------------------------------------------------------------------------- #
# plots from benchmark records
# --------------------------------------------------------------------------- #

def _title(fig, title, sub, wrap=150):
    fig.text(0.04, 0.965, title, fontsize=19, fontweight="bold", color=S.TEXT, ha="left", va="top", family="sans-serif")
    fig.text(0.04, 0.915, "\n".join(textwrap.wrap(sub, wrap)), fontsize=11.5, color=S.MUTED, ha="left", va="top",
             family="sans-serif", linespacing=1.4)


def fourier() -> None:
    e, u = rec("benchmarks/modadd", "E0003"), rec("benchmarks/modadd", "E0004")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True)
    fig.subplots_adjust(top=0.82, hspace=0.18, left=0.07, right=0.98, bottom=0.1)
    for ax, r, name in zip(axes, (e, u), ("embedding W_E", "unembedding W_U")):
        f, sh, key = r["data"]["frequencies"], np.array(r["data"]["share"]), set(r["summary"]["key_frequencies"])
        ax.bar(f, sh * 100, color=[S.ACCENT if k in key else S.DIM for k in f], width=0.8)
        for t in r["summary"]["top_frequencies"][:4]:
            ax.text(t["frequency"], t["share"] * 100 + 0.8, f"k={t['frequency']}", ha="center", fontsize=10, color=S.TEXT)
        ax.axhline(100 / len(f), color=S.MUTED, lw=1, ls=":")
        ax.set_ylabel("% of power")
        ax.set_title(f"{name}:  {r['summary']['share_of_power_in_key_frequencies'] * 100:.0f}% of power in "
                     f"{len(key)} of {len(f)} frequencies", fontsize=12)
        ax.grid(axis="x", visible=False)
    axes[1].set_xlabel("frequency k (waves per 113 tokens); dotted line = what a uniform spread would give")
    _title(fig, "The model learned to add with waves",
           f"Real data from the blind modular-addition run (E0003, E0004). 1-layer transformer, (a + b) mod 113. {e['provenance']['tag']}.")
    fig.savefig(MEDIA / "fourier.png", bbox_inches=None)
    plt.close(fig)
    print("wrote fourier.png")


def circuit_modadd() -> None:
    d = "benchmarks/modadd"
    e3, e4, e11, v = rec(d, "E0003"), rec(d, "E0004"), rec(d, "E0011"), rec(d, "V0001")["checks"]["causal_subspace"]
    e7, e8, e2 = rec(d, "E0007"), rec(d, "E0008"), rec(d, "E0002")
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_facecolor(S.NAVY)

    def node(x, y, w, h, title, lines, on=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=S.PANEL,
                                    ec=S.ACCENT if on else S.GRID, lw=1.8 if on else 1.2))
        ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top", fontsize=12.5, fontweight="bold", color=S.TEXT)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.74 - i * 0.31, line, ha="center", va="top", fontsize=9.2, color=S.MUTED)

    def arrow(x1, y1, x2, y2):
        for lw, a in ((7, 0.12), (2.2, 1.0)):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>" if lw < 3 else "-", mutation_scale=16,
                                         color=S.ACCENT, lw=lw, alpha=a))

    top = [t for t in e3["summary"]["top_frequencies"][:5]]
    counts = sorted(e11["summary"]["frequencies_with_units"].items(), key=lambda kv: -kv[1])
    cs = [f"k={k}: {n}" for k, n in counts[:4]]
    node(0.2, 3.75, 1.3, 1.15, "tokens", ["a   b   ="], on=False)
    node(1.95, 3.1, 2.45, 2.35, "1  embed as waves", ["cos / sin of 2\u03c0ka/113", "k = " + ", ".join(str(t["frequency"]) for t in top),
                                                      f"{e3['summary']['share_of_power_in_key_frequencies'] * 100:.0f}% of W_E power", "(E0003)"])
    node(4.85, 3.1, 2.45, 2.35, "2  attention mixes", ["4 heads at '='", "~50 / 50 on a and b",
                                                       f"all 4 ablated: {abs(e8['summary']['fraction_of_clean']) * 100:.0f}% left", "(E0008-E0010)"])
    node(7.75, 3.1, 2.45, 2.35, "3  MLP combines", ["each neuron tuned to one k", "  ".join(cs[:2]), "  ".join(cs[2:4]),
                                                    f"ablated: {e7['data']['metric'][0] / e7['summary']['metric_clean'] * 100:.1f}% left", "(E0007, E0011)"])
    node(10.65, 3.1, 2.15, 2.35, "4  read out", ["W_U uses the", "same frequencies", f"{e4['summary']['share_of_power_in_key_frequencies'] * 100:.0f}% of W_U power",
                                                 "(E0004, E0016)"])
    ax.text(11.72, 2.5, "(a + b) mod 113", ha="center", fontsize=13, color=S.ACCENT, fontweight="bold")
    ax.text(11.72, 2.15, f"accuracy {e2['summary']['clean_accuracy']:.2f} on all {e2['summary']['n']} pairs", ha="center", fontsize=9.3, color=S.MUTED)
    for x1, x2 in ((1.5, 1.95), (4.4, 4.85), (7.3, 7.75), (10.2, 10.65)):
        arrow(x1, 4.3, x2, 4.3)
    # causal strip
    ax.text(0.25, 1.55, "Causal check on held-out inputs (V0001)", fontsize=12.5, fontweight="bold", color=S.TEXT)
    items = [("only the key-frequency subspace kept", v["only_subspace_kept"]["accuracy"], S.ACCENT),
             ("key-frequency subspace removed", v["subspace_removed"]["accuracy"], S.WARN),
             ("random subspace of the same size removed", v["random_subspace_removed"]["accuracy"], S.MUTED)]
    for i, (lab, acc, col) in enumerate(items):
        y = 1.05 - i * 0.38
        ax.text(0.25, y, lab, fontsize=10.5, color=S.MUTED, va="center")
        ax.add_patch(FancyBboxPatch((5.0, y - 0.11), 6.0, 0.22, boxstyle="round,pad=0,rounding_size=0.05", fc=S.PANEL, ec=S.GRID, lw=1))
        ax.add_patch(FancyBboxPatch((5.0, y - 0.11), max(6.0 * acc, 0.03), 0.22, boxstyle="round,pad=0,rounding_size=0.05", fc=col, ec="none"))
        ax.text(11.2, y, f"{acc * 100:.1f}% accuracy", fontsize=10.5, color=col, va="center")
    ax.text(0.25, 6.05, "The recovered modular-addition circuit", fontsize=19, fontweight="bold", color=S.TEXT, family="sans-serif")
    ax.text(0.25, 5.7, f"Found by Claude Code driving glassbox, with no hints, on the {rec(d, 'E0001')['provenance']['backend']} backend. "
            f"Every number is from the run's own records.", fontsize=10.5, color=S.MUTED, family="sans-serif")
    ax.text(8.6, 1.55, f"subspace = {sum(v['dims'])} of {v['of_total_dims'][0]} embedding dims", fontsize=9.6, color=S.MUTED, ha="left")
    fig.savefig(MEDIA / "circuit-modadd.png")
    plt.close(fig)
    print("wrote circuit-modadd.png")


def verification() -> None:
    m, i = rec("benchmarks/modadd", "V0001"), rec("benchmarks/ioi", "V0002")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(top=0.78, wspace=0.22, left=0.06, right=0.98, bottom=0.2)
    ms = m["checks"]["causal_subspace"]
    labels = ["model\naccuracy", "algorithm\n= model", "only the\ncircuit", "circuit\nremoved", "random\nremoved"]
    vals = [m["model_accuracy"], m["checks"]["agreement"]["agreement_with_model"], ms["only_subspace_kept"]["accuracy"],
            ms["subspace_removed"]["accuracy"], ms["random_subspace_removed"]["accuracy"]]
    cols = [S.DIM, S.ACCENT, S.ACCENT, S.WARN, S.MUTED]
    ic = i["checks"]["causal_components"]
    vals2 = [i["model_accuracy"], i["checks"]["agreement"]["agreement_with_model"], ic["everything_else_ablated"]["fraction_of_clean"],
             ic["circuit_ablated"]["fraction_of_clean"], float(np.mean(ic["random_same_size_ablated_fraction"]))]
    for ax, v, title, yl in ((axes[0], vals, f"Modular addition: accuracy on {m['n_heldout']} held-out inputs", "accuracy"),
                             (axes[1], vals2, f"IOI in GPT-2 small: {i['n_heldout']} held-out prompts", "share (logit difference for the last three)")):
        ax.bar(range(5), v, color=cols, width=0.68)
        ax.axhline(0, color=S.GRID, lw=1)
        for k, x in enumerate(v):
            ax.text(k, max(x, 0) + 0.03, f"{x:.2f}", ha="center", fontsize=11, color=S.TEXT)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(min(-0.2, min(v) - 0.08), 1.22)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    axes[1].text(0.5, -0.3, f"The IOI circuit alone keeps {ic['everything_else_ablated']['fraction_of_clean']:.2f} of the logit difference, "
                 f"but its top-1 accuracy drops to {ic['everything_else_ablated']['accuracy']:.2f}.", transform=axes[1].transAxes,
                 ha="center", fontsize=9.8, color=S.MUTED)
    axes[0].text(0.5, -0.3, f"Circuit = {sum(ms['dims'])} of {ms['of_total_dims'][0]} embedding dimensions.", transform=axes[0].transAxes,
                 ha="center", fontsize=9.8, color=S.MUTED)
    _title(fig, "Verification: predict, then break it",
           "The algorithm is written as code and compared with the real model on unseen inputs. Then the claimed circuit is ablated, and everything but it. "
           f"{m['provenance']['tag']}.")
    fig.savefig(MEDIA / "verification.png", bbox_inches=None)
    plt.close(fig)
    print("wrote verification.png")


def table_png(name: str, title: str, sub: str, header: list[str], rows: list[list[str]], widths: list[float],
              colors: dict[tuple[int, int], str] | None = None, foot: str = "") -> None:
    n = len(rows)
    H = 2.2 + 0.62 * (n + 1) + (0.5 if foot else 0)
    fig, ax = plt.subplots(figsize=(13, H))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0.3, H - 0.45, title, fontsize=19, fontweight="bold", color=S.TEXT, family="sans-serif", va="top")
    for k, line in enumerate(textwrap.wrap(sub, 135)):
        ax.text(0.3, H - 0.95 - 0.27 * k, line, fontsize=10.5, color=S.MUTED, family="sans-serif", va="top")
    xs = np.concatenate([[0.3], 0.3 + np.cumsum(widths)])
    y0 = H - 2.2
    for j, hname in enumerate(header):
        ax.text(xs[j] + 0.1, y0 + 0.2, hname, fontsize=10.5, color=S.MUTED, va="center")
    ax.plot([0.3, 12.7], [y0 - 0.08, y0 - 0.08], color=S.GRID, lw=1.2)
    for i, row in enumerate(rows):
        y = y0 - 0.62 * (i + 1)
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((0.3, y - 0.02), 12.4, 0.58, boxstyle="round,pad=0,rounding_size=0.06", fc="#0b112a", ec="none"))
        for j, cell in enumerate(row):
            col = (colors or {}).get((i, j), S.TEXT if j == 0 else S.MUTED)
            ax.text(xs[j] + 0.1, y + 0.27, cell, fontsize=11 if j else 11.5, color=col, va="center",
                    fontweight="bold" if j == 0 else "normal")
    if foot:
        for k, line in enumerate(textwrap.wrap(foot, 165)):
            ax.text(0.3, 0.42 - 0.22 * k, line, fontsize=9.3, color=S.MUTED, family="sans-serif")
    fig.savefig(MEDIA / name)
    plt.close(fig)
    print("wrote", name)


def scorecard() -> None:
    card = json.loads((ROOT / "benchmarks" / "scorecard.json").read_text(encoding="utf-8"))["benchmarks"]
    short = {"modadd": "Modular addition", "induction": "Induction (2-layer)", "ioi": "IOI (GPT-2 small)"}
    rows, colors = [], {}
    for i, (name, b) in enumerate(card.items()):
        r, fv = b["rubric"], b["final_verification"]
        if name == "ioi":
            m = b["mechanical"]
            extra = f"{m['published_heads_found']} of {m['published_heads']} published heads"
        elif name == "modadd":
            extra = "key frequencies + causal subspace"
        else:
            extra = "prev-token + induction head"
        status = fv["status"] if fv["status"] == "verified" else "partial"
        rows.append([short[name], f"{r['found']} / {r['partial']} / {r['missed']}", extra, str(b["experiments"]),
                     f"{b['wall_clock_seconds'] / 60:.1f} min", status])
        colors[(i, 5)] = S.ACCENT if status == "verified" else S.WARN
        colors[(i, 1)] = S.TEXT
    table_png("scorecard.png", "Benchmark scorecard",
              "Three published circuits, each rediscovered by a fresh Claude Code session with MCP tools only and no hints in the prompt. "
              "Blind means: no hints given, every claim backed by an experiment in the notebook. It does not mean the scientist has never seen the papers.",
              ["benchmark", "found / partial / missed", "recovered", "experiments", "wall clock", "verification"],
              rows, [2.9, 2.6, 3.5, 1.3, 1.2, 0.9], colors,
              "Rubric = mechanism-level items from the paper. Full precision, local-full backend, RTX 4070 Super, one run each. "
              "Details, misses and caveats: benchmarks/SCORECARD.md.")


def notebook_png() -> None:
    nb = [json.loads(l) for l in (ROOT / "benchmarks" / "ioi" / "notebook.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {e["id"]: e for e in nb}
    rej = next(e for e in nb if e.get("status") == "rejected" and e.get("target"))
    hyp = by[rej["target"]]
    i = nb.index(rej)
    exps = [e for e in nb[nb.index(hyp):i] if e["kind"] == "experiment"]
    nxt = next(e for e in nb[i + 1:] if e["kind"] == "hypothesis" and not e.get("target"))
    cards = [(hyp["id"], "hypothesis", "rejected", hyp["text"], S.WARN)]
    cards += [(e["id"], "experiment", ", ".join(e["refs"]), e["text"], S.MUTED) for e in exps[:1]]
    cards += [(rej["id"], "result", "rejects " + hyp["id"], rej["text"], S.WARN), (nxt["id"], "hypothesis", "revised", nxt["text"], S.ACCENT)]
    wrapped = [(a, b, c, textwrap.wrap(t, 118)[:5], col) for a, b, c, t, col in cards]
    H = 1.5 + sum(0.62 + 0.27 * len(w[3]) for w in wrapped)
    fig, ax = plt.subplots(figsize=(13, H))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0.3, H - 0.4, "The lab notebook: a hypothesis rejected, then revised", fontsize=19, fontweight="bold", color=S.TEXT,
            family="sans-serif", va="top")
    ax.text(0.3, H - 0.88, "Verbatim entries from benchmarks/ioi/notebook.jsonl (blind IOI run on GPT-2 small). Nothing is edited in place; "
            "status changes are appended.", fontsize=10.5, color=S.MUTED, family="sans-serif", va="top")
    y = H - 1.35
    for nid, kind, status, lines, col in wrapped:
        h = 0.5 + 0.27 * len(lines)
        ax.add_patch(FancyBboxPatch((0.3, y - h), 12.4, h, boxstyle="round,pad=0,rounding_size=0.08", fc=S.PANEL, ec=col, lw=1.4))
        ax.text(0.5, y - 0.26, f"{nid}  {kind}", fontsize=10.5, color=S.TEXT, fontweight="bold", va="center")
        ax.text(12.5, y - 0.26, status, fontsize=10.5, color=col, va="center", ha="right")
        for k, line in enumerate(lines):
            ax.text(0.5, y - 0.56 - 0.27 * k, line, fontsize=9.6, color=S.MUTED, va="center")
        y -= h + 0.12
    fig.savefig(MEDIA / "notebook.png")
    plt.close(fig)
    print("wrote notebook.png")


def steering_27b() -> None:
    p = ROOT / "demos" / "qwen27b-language" / "summary.json"
    if not p.exists():
        print("skip steering-27b.png (no demo summary yet)")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], top=0.79, hspace=0.5, wspace=0.32, left=0.06, right=0.97, bottom=0.04)
    ax = fig.add_subplot(gs[0, 0])
    layers = list(range(len(d["separation_by_layer"])))
    ax.plot(layers, d["separation_by_layer"], color=S.ACCENT, marker="o", ms=3, label=d.get("separation_name", "held-out accuracy"))
    if d.get("control_by_layer"):
        ax.plot(layers, d["control_by_layer"], color=S.MUTED, ls="--", label="shuffled-label control")
    if d.get("lens_by_layer"):
        ax2 = ax.twinx()
        ax2.plot(layers, d["lens_by_layer"], color="#b69cff", lw=1.6)
        ax2.set_ylabel(d.get("lens_name", "logit lens"), color="#b69cff")
        ax2.tick_params(axis="y", colors="#b69cff")
        ax2.grid(False)
    for L in d.get("steer_layers", []):
        ax.axvline(L, color=S.DIM, lw=1, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel(d.get("separation_name", "held-out accuracy"))
    ax.set_title("Where the language is readable, by layer", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, frameon=False)
    ax = fig.add_subplot(gs[0, 1])
    st = d["steering"]
    ax.plot(st["alphas"], st["metric"], color=S.ACCENT, marker="o", label="language direction")
    if st.get("control_metric"):
        ax.plot(st["control_alphas"], st["control_metric"], color=S.MUTED, marker="o", ls="--", label="control direction (same size)")
        ax.legend(loc="upper right")
    ax.axhline(0, color=S.DIM, lw=1, ls=":")
    ax.set_xlabel(f"steering strength at {', '.join(d.get('steer_sites', []))} (fraction of the residual norm)")
    ax.set_ylabel(st.get("metric_short", "metric"))
    ax.text(0.0, -0.3, st.get("metric_name", ""), transform=ax.transAxes, fontsize=9.5, color=S.MUTED)
    ax.set_title("Steering along the direction flips the behaviour", fontsize=12)
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.95
    for g in d.get("generations", [])[:3]:
        ax.text(0.0, y, "prompt  ", fontsize=10, color=S.MUTED, va="top")
        ax.text(0.075, y, textwrap.shorten(g["prompt"], 120), fontsize=10, color=S.TEXT, va="top")
        ax.text(0.0, y - 0.1, "before  ", fontsize=10, color=S.MUTED, va="top")
        ax.text(0.075, y - 0.1, textwrap.shorten(g["before"].replace("\n", " "), 130), fontsize=10, color=S.MUTED, va="top")
        ax.text(0.0, y - 0.2, f"{g['alpha']:+.1f}".ljust(8), fontsize=10, color=S.ACCENT, va="top")
        ax.text(0.075, y - 0.2, textwrap.shorten(g["after"].replace("\n", " "), 130), fontsize=10, color=S.ACCENT, va="top")
        y -= 0.34
    _title(fig, d.get("title", "A direction found and steered in a 27B model on a home PC"),
           f"{d['model']} via llama.cpp on {' + '.join(d.get('gpus', []))}. Label: {d['label']}.", wrap=120)
    fig.savefig(MEDIA / "steering-27b.png", bbox_inches=None)
    plt.close(fig)
    print("wrote steering-27b.png")


def scale_ladder() -> None:
    p = ROOT / "ladder" / "summary.json"
    if not p.exists():
        print("skip scale-ladder.png (no ladder summary yet)")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    short = {"gpt2-local": "GPT-2 small, 124M", "gpt2-ndif": "GPT-2 small, 124M", "qwen27b-gguf": "27B abliterated GGUF",
             "qwen27b-ndif": "Qwen3.8-27B official", "qwen27b-4bit": "Qwen3.8-27B official"}
    prec = {"gpt2-local": "local-full, full", "gpt2-ndif": "ndif, full", "qwen27b-gguf": "local-gguf, screening",
            "qwen27b-ndif": "ndif, full", "qwen27b-4bit": "local-4bit, screening"}
    col = {"supported": S.ACCENT, "rejected": S.WARN}
    rows, colors = [], {}
    for i, r in enumerate(d["rungs"]):
        raw = json.loads((ROOT / "ladder" / r["rung"] / "rung.json").read_text(encoding="utf-8"))
        P, L = raw["predictions"], raw["n_layers"] - 1
        cells = [f"{P['P1']['accuracy']:.2f} acc, +{P['P1']['logit_diff']:.1f}",
                 f"layer {P['P2']['crossover_layer']} of {L}" if P["P2"].get("crossover_layer") is not None else "cannot patch",
                 ", ".join(h["head"] for h in P["P3"]["top_heads"][:2]) + f" ({P['P3']['top3_heads_effect']:.2f})" if P["P3"].get("top_heads") else "cannot patch",
                 f"{P['P4']['fraction_of_logit_diff_left']:.2f} left" if P["P4"].get("heads") else "cannot patch"]
        rows.append([short[r["rung"]], prec[r["rung"]]] + cells)
        for j, k in enumerate(("P1", "P2", "P3", "P4")):
            colors[(i, j + 2)] = col.get(P[k].get("verdict"), S.DIM)
    for n in d.get("pending", []):
        rows.append([short[n], prec[n], "not run yet", "", "", ""])
        for j in range(2, 6):
            colors[(len(rows) - 1, j)] = S.DIM
    table_png("scale-ladder.png", "Scale ladder: the IOI circuit from 124M to 27B",
              "Four predictions taken from the circuit found in GPT-2 small, tested with the same script on every rung. "
              "Green = supported, red = not met, grey = not testable or not run. "
              "All four hold in the official 27B, both in the local 4-bit screen and at full precision on NDIF.",
              ["model", "backend, precision", "P1 task", "P2 hand-off", "P3 late heads", "P4 backup"],
              rows, [3.0, 2.5, 2.0, 1.8, 2.2, 0.9], colors,
              "P1: top-1 accuracy and logit difference. P2: layer where patching END overtakes patching S2. P3: strongest heads and their joint patching effect. "
              "P4: logit difference left after ablating them. Details: ladder/LADDER.md.")


ASSETS = {"how-it-works": how_it_works, "free-stack": free_stack, "backends": backends, "fourier": fourier,
          "circuit-modadd": circuit_modadd, "verification": verification, "scorecard": scorecard, "notebook": notebook_png,
          "steering-27b": steering_27b, "scale-ladder": scale_ladder}


# --------------------------------------------------------------------------- #
# glassbox-bench, null controls, pre-registration, model diff
# --------------------------------------------------------------------------- #

BENCH = ROOT / "benchmarks" / "glassbox-bench"


def _bench_cards() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(BENCH.glob("seed-*/scorecard.json"))]


def bench_scorecard() -> None:
    cards = _bench_cards()
    if not cards:
        print("skip bench-scorecard.png (no graded seeds)")
        return
    fams: dict[str, dict] = {}
    for c in cards:
        for t in c["tasks"]:
            fams.setdefault(t["family"], {"track": t["track"], "cells": {}, "exp": []})
            f = fams[t["family"]]
            f["exp"].append(t["experiments"])
            if t["track"] == "compiled":
                ok = (t["mechanism_match"] or 0) >= 0.98 and (t["circuit"]["recall"] or 0) == 1.0
                f["cells"][c["seed"]] = (f"{t['circuit']['recall']:.2f} / {t['circuit']['precision']:.2f} / {t['mechanism_match'] or 0:.2f}", ok)
            else:
                f["cells"][c["seed"]] = (f"behaviour {t['behavior_prediction'] or 0:.2f}", (t["behavior_prediction"] or 0) >= 0.98)
    seeds = [c["seed"] for c in cards]
    order = sorted(fams, key=lambda k: (fams[k]["track"] != "compiled", k))
    rows, colors = [], {}
    for i, k in enumerate(order):
        f = fams[k]
        rows.append([k, f["track"]] + [f["cells"].get(s, ("-", False))[0] for s in seeds] + [f"{np.mean(f['exp']):.0f}"])
        for j, s in enumerate(seeds):
            colors[(i, 2 + j)] = S.ACCENT if f["cells"].get(s, ("", False))[1] else S.WARN
    solved = sum(c["summary"]["compiled_solved"] for c in cards)
    n_comp = sum(sum(t["track"] == "compiled" for t in c["tasks"]) for c in cards)
    conf = sum(c["summary"]["prereg_confirmed"] for c in cards)
    ref = sum(c["summary"]["prereg_refuted"] for c in cards)
    table_png("bench-scorecard.png", "glassbox-bench: planted circuits, sealed answer keys",
              f"Algorithms written in RASP, compiled to weights with Tracr, then rotated and padded with distractor heads and neurons. "
              f"A fresh task set per seed. {solved} of {n_comp} compiled tasks solved; predictions: {conf} confirmed, {ref} refuted. "
              "Every task at this difficulty was solved, so the bench does not yet separate a strong scientist from a weak one.",
              ["family", "track"] + [f"seed {s}" for s in seeds] + ["experiments"],
              rows, [2.3, 1.5] + [2.5] * len(seeds) + [1.1], colors,
              "Compiled cells: circuit recall / circuit precision / mechanism match on held-out inputs. Trained track: behaviour prediction only "
              "(mechanism unknown, never scored on circuit match). Keys were sealed outside the repo during the runs: an honour-system seal on one PC.")


def null_controls() -> None:
    files = [ROOT / "benchmarks" / b / "null_calibration.json" for b in ("modadd", "induction", "ioi")]
    if not all(f.exists() for f in files):
        print("skip null-controls.png (run benchmarks/reverify_nulls.py)")
        return
    names = {"modadd": "Modular addition", "induction": "Induction", "ioi": "IOI in GPT-2 small"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9))
    fig.subplots_adjust(top=0.72, wspace=0.24, left=0.06, right=0.98, bottom=0.17)
    for ax, f in zip(axes, files):
        d = json.loads(f.read_text(encoding="utf-8"))
        nc = next(n for n in d["null_controls"] if n["finding"].startswith("necessity"))
        null = np.array(d["null_samples"][nc["finding"]])
        lo, hi = min(null.min(), 0) - 0.05, max(nc["effect"], null.max()) + 0.1
        ax.hist(null, bins=np.linspace(lo, hi, 36), color=S.DIM, edgecolor=S.PANEL)
        ax.axvline(nc["null_p95"], color=S.MUTED, ls="--", lw=1.5)
        ax.axvline(nc["effect"], color=S.ACCENT, lw=2.5)
        ymax = ax.get_ylim()[1]
        ax.text(nc["null_p95"], ymax * 0.97, f" null p95 = {nc['null_p95']:.2f}", color=S.MUTED, fontsize=9.5, va="top", ha="left")
        ax.text(nc["effect"], ymax * 0.8, f"claimed circuit \n{nc['effect']:.2f} ", color=S.ACCENT, fontsize=10, va="top", ha="right")
        ax.set_title(f"{names[d['benchmark']]}  ({nc['n_null']} null samples)", fontsize=11.5)
        ax.set_xlabel("standardised drop in the metric when ablated")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("null samples")
    _title(fig, "Null controls: is the effect bigger than what a random choice gives?",
           "Grey: the same ablation applied to random component sets of the same size outside the claimed circuit (random subspaces for modular addition), and to "
           "the same circuit in null models with random or shuffled weights. A finding passes only above the 95th percentile of its nulls. "
           "full precision (local).", wrap=165)
    fig.savefig(MEDIA / "null-controls.png", bbox_inches=None)
    plt.close(fig)
    print("wrote null-controls.png")


def prereg_record() -> None:
    from glassbox.notebook import Notebook

    cards = _bench_cards()
    books = [b for b in sorted(BENCH.glob("seed-*/runs/t*/notebook.jsonl")) if "__null-" not in b.parent.name]
    if not books:
        print("skip prereg-record.png (no bench notebooks)")
        return
    tot = {"confirmed": 0, "refuted": 0, "inconclusive": 0}
    refuted = []
    for p in books:
        r = Notebook(p).prediction_record()
        for k in tot:
            tot[k] += r[k]
        for x in r["refuted_predictions"]:
            refuted.append({**x, "where": f"{p.parent.parent.parent.name} {p.parent.name}"})
    audit = ROOT / "audits" / "abliteration" / "summary.json"
    a = json.loads(audit.read_text(encoding="utf-8"))["predictions"] if audit.exists() else None
    decided = tot["confirmed"] + tot["refuted"]
    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], top=0.76, left=0.06, right=0.98, bottom=0.08, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    vals = [tot["confirmed"], tot["refuted"], tot["inconclusive"]]
    ax.bar(range(3), vals, color=[S.ACCENT, S.WARN, S.DIM], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=12, color=S.TEXT)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["confirmed", "refuted", "inconclusive"])
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(f"hit rate {tot['confirmed'] / decided * 100:.0f}%  ({decided} decided)", fontsize=12)
    ax.grid(axis="x", visible=False)
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 0.99, "Refuted predictions stay in the report. Three examples:", fontsize=11.5, color=S.TEXT, va="top", fontweight="bold")
    y = 0.88
    for x in sorted(refuted, key=lambda r: len(r["hypothesis"]))[:3]:
        hyp = textwrap.wrap(x["hypothesis"], 96)[:3]
        h = 0.075 * len(hyp) + 0.105
        ax.add_patch(FancyBboxPatch((0, y - h), 1, h, boxstyle="round,pad=0,rounding_size=0.015", fc=S.PANEL, ec=S.WARN, lw=1.2))
        for k, line in enumerate(hyp):
            ax.text(0.015, y - 0.03 - 0.075 * k, line, fontsize=9.3, color=S.MUTED, va="top")
        pr = x["prediction"]
        obs = f"{x['observed']:.3f}" if isinstance(x["observed"], float) else str(x["observed"])
        ax.text(0.015, y - 0.045 - 0.075 * len(hyp), f"predicted {pr['metric'].split('.', 1)[1][:44]} {pr['op']} {pr['threshold']}   observed {obs}   "
                f"[{x['where']}, {x['prereg']}]", fontsize=9.3, color=S.WARN, va="top")
        y -= h + 0.03
    extra = f" The model-diff audit (scored separately): {a['confirmed']} confirmed, {a['refuted']} refuted." if a else ""
    _title(fig, "Prediction record: every experiment is pre-registered",
           f"Before an experiment runs, the scientist commits to a quantitative prediction in a hash-chained notebook; the lab scores it. "
           f"glassbox-bench, {len(cards)} seeds, {len(books)} investigations.{extra}", wrap=160)
    fig.savefig(MEDIA / "prereg-record.png", bbox_inches=None)
    plt.close(fig)
    print("wrote prereg-record.png")


def model_diff() -> None:
    p = ROOT / "audits" / "abliteration" / "summary.json"
    if not p.exists():
        print("skip model-diff.png (no audit summary)")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    if "exploratory" not in d:
        print("skip model-diff.png (run audits/abliteration/run_audit.py explore)")
        return
    A, X, pr = d["activations"], d["exploratory"], d["predictions"]
    layers = [x["layer"] for x in A["per_layer"]]
    lo, hi = X["edited_layer_range"]
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], top=0.77, hspace=0.46, wspace=0.2, left=0.07, right=0.97, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    ax.axvspan(lo - 0.5, hi + 0.5, color=S.ACCENT, alpha=0.08, lw=0)
    ax.plot(layers, [abs(x["gap_ratio"]) for x in A["per_layer"]], color=S.ACCENT, lw=2.2)
    ax.axhline(1, color=S.DIM, ls=":", lw=1.2)
    ax.set_ylim(0, 1.15)
    ax.text((lo + hi) / 2, 1.07, f"edited layers {lo} to {hi}", color=S.ACCENT, fontsize=9.5, ha="center")
    ax.set_title("Activations: how much of the refusal signal is left", fontsize=11.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("gap left (edited / official)")

    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(lo - 0.5, hi + 0.5, color=S.ACCENT, alpha=0.08, lw=0)
    cos = X["cosine_edit_direction_vs_refusal_direction_by_layer"]
    ax.plot(layers, cos, color=S.ACCENT, lw=2.2, label="edit direction (from the weights)")
    N = X["nulls"]
    ax.axhline(N["unrelated_diffmeans_at_best_layer_p95"], color=S.MUTED, ls="--", lw=1.2, label="null p95: unrelated mean differences")
    ax.axhline(N["reader_matrix_directions_best_layer_each_p95"], color=S.MUTED, ls=":", lw=1.2, label="null p95: directions of unedited matrices")
    ax.legend(loc="upper left", fontsize=8.5)
    b = X["best_layer"]
    ax.annotate(f"{X['best_cosine']:.2f} at layer {b}", (b, X["best_cosine"]), (b + 5, X["best_cosine"] - 0.02), color=S.TEXT, fontsize=10,
                arrowprops={"arrowstyle": "-", "color": S.MUTED})
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Edit direction vs refusal direction (cosine)", fontsize=11.5)
    ax.set_xlabel("layer of the refusal direction (official model)")

    ax = fig.add_subplot(gs[1, :])
    roles = ["attn_out", "mlp_out", "attn_in", "mlp_in"]
    lab = ["attention out  (writes)", "MLP out  (writes)", "attention in  (reads)", "MLP in  (reads)"]
    M = np.full((4, len(layers)), np.nan)
    for m in X["by_matrix"].values():
        if m["layer"] < len(layers):
            M[roles.index(m["role"]), m["layer"]] = m["energy_over_floor"]
    im = ax.imshow(np.log10(np.clip(M, 0.5, 100)), cmap="glass", aspect="auto", vmin=np.log10(0.5), vmax=2)
    ax.set_yticks(range(4))
    ax.set_yticklabels(lab)
    ax.set_xlabel("layer")
    ax.set_title("Weights: structure of the difference in each matrix, as a multiple of its own quantization noise floor", fontsize=11.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["1x", "10x", "100x"])
    cb.outline.set_edgecolor(S.GRID)
    _title(fig, "Auditing a model edit: where does the abliterated 27B differ from the official one?",
           f"Two GGUF files, compared matrix by matrix and through activations on a public harmful/harmless prompt set. {pr['refuted']} of "
           f"{pr['confirmed'] + pr['refuted']} pre-registered predictions were refuted: the edit is not in every writing matrix. Exploratory follow-up: "
           f"{X['edited_matrices']} matrices in layers {lo} to {hi} share one direction (cosine above 0.99), and it matches the refusal direction. "
           f"Label: {d['label']}. No completions stored.", wrap=160)
    fig.savefig(MEDIA / "model-diff.png", bbox_inches=None)
    plt.close(fig)
    print("wrote model-diff.png")


ASSETS.update({"bench-scorecard": bench_scorecard, "null-controls": null_controls, "prereg-record": prereg_record, "model-diff": model_diff})

if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    for name in (sys.argv[1:] or ASSETS):
        ASSETS[name]()
