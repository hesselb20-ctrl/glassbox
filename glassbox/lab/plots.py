"""Plot helpers. Every figure carries the backend tag that produced its numbers."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import style

style.apply_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


def _stamp(fig, lab) -> None:
    p = lab.backend.provenance()
    fig.text(0.995, 0.005, f"{p['model']} · {p['backend']} · {p['tag']}", ha="right", va="bottom",
             fontsize=7.5, color=style.MUTED)


def heatmap(lab, matrix, title: str, xlabel: str, ylabel: str, xticks: Sequence | None = None,
            yticks: Sequence | None = None, diverging: bool = True, cbar: str = ""):
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    h, w = m.shape
    fig, ax = plt.subplots(figsize=(min(12, 2.6 + 0.48 * w), min(9, 1.8 + 0.36 * h)))
    if diverging:
        lim = float(np.nanmax(np.abs(m))) or 1.0
        im = ax.imshow(m, cmap="glass_div", vmin=-lim, vmax=lim, aspect="auto")
    else:
        im = ax.imshow(m, cmap="glass", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if xticks is not None and len(xticks) <= 40:
        ax.set_xticks(range(w))
        ax.set_xticklabels([str(t) for t in xticks], rotation=60 if max(len(str(t)) for t in xticks) > 3 else 0,
                           ha="right" if max(len(str(t)) for t in xticks) > 3 else "center")
    if yticks is not None and len(yticks) <= 40:
        ax.set_yticks(range(h))
        ax.set_yticklabels([str(t) for t in yticks])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar, color=style.MUTED)
    cb.outline.set_edgecolor(style.GRID)
    _stamp(fig, lab)
    return fig


def lines(lab, x, series: dict, title: str, xlabel: str, ylabel: str, hline: float | None = None):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = [style.ACCENT, style.MUTED, style.WARN, "#b69cff"]
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(x, ys, marker="o", ms=4, color=colors[i % len(colors)], label=name,
                ls="-" if i == 0 else "--")
    if hline is not None:
        ax.axhline(hline, color=style.DIM, lw=1, ls=":")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend()
    _stamp(fig, lab)
    return fig


def bars(lab, labels, values, title: str, xlabel: str, ylabel: str, highlight: Sequence[int] = ()):
    v = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(max(6.5, min(13, 0.22 * len(v) + 3)), 4.2))
    colors = [style.ACCENT if i in set(highlight) else style.DIM for i in range(len(v))]
    ax.bar(range(len(v)), v, color=colors, width=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    step = max(1, len(v) // 28)
    ax.set_xticks(range(0, len(v), step))
    ax.set_xticklabels([str(labels[i]) for i in range(0, len(v), step)],
                       rotation=60 if max(len(str(x)) for x in labels) > 3 else 0)
    ax.grid(axis="x", visible=False)
    _stamp(fig, lab)
    return fig
