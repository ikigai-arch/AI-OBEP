"""Shared matplotlib style for paper figures (thin marks, recessive grid, fixed hue order)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots 1-3 (validated all-pairs for CVD); assigned to entities, never re-ordered.
ALGO_COLORS = {"PSO": "#2a78d6", "GA": "#eb6834", "HHO": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
NEUTRAL = "#8a8984"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False,
})
