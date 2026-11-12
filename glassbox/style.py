"""One palette and type scale for every plot, diagram and video frame."""
from __future__ import annotations

NAVY = "#070b1a"        # page background
PANEL = "#0e1530"       # card / axes background
GRID = "#1c2750"        # hairlines
TEXT = "#e6ebff"        # primary text
MUTED = "#8793c2"       # secondary text
ACCENT = "#3df5d0"      # the one glowing colour: active circuit paths, key results
DIM = "#3a4780"         # inactive components
WARN = "#ff6b8a"        # negative effects / failures, used sparingly
MONO = ["JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "monospace"]

PALETTE = dict(navy=NAVY, panel=PANEL, grid=GRID, text=TEXT, muted=MUTED, accent=ACCENT, dim=DIM, warn=WARN)


def apply_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": NAVY,
        "savefig.facecolor": NAVY,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": TEXT,
        "font.family": "monospace",
        "font.monospace": MONO,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.fontsize": 9.5,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
    })
    for name, colors in {
        "glass": [PANEL, "#14506a", ACCENT],
        "glass_div": [WARN, "#5a2a55", PANEL, "#14506a", ACCENT],
    }.items():
        if name not in matplotlib.colormaps:
            matplotlib.colormaps.register(LinearSegmentedColormap.from_list(name, colors))
