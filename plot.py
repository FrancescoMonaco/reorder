"""
Elegant reproduction of Figure 2 (epsilon-flexibility) for the poster.

Left panel : synthetic 2D point set with its MST drawn on top. The heaviest
             edges ("the tail") are the ones that can be swapped for
             arbitrarily bad edges while adding at most an eps fraction of
             extra cost -- these are highlighted.
Right panel: the same MST edges, sorted by weight (rank on x, distance on
             y). The dashed line marks the dataset diameter -- the worst
             case replacement weight for any tail edge.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ---------- palette (matches the poster) ----------
INK    = "#2C2C2A"
GRAY   = "#95C596"
CORAL  = "#993C1D"
CORAL_L= "#D85A30"
BG     = "none"   # transparent, so it sits on the panel's tinted background

# Use seaborn for cleaner defaults
sns.set_theme(
    style="white",
    palette=[INK, GRAY, CORAL, CORAL_L],
    font="sans-serif",
    font_scale=1.0,
    rc={
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.size": 18,
        "axes.linewidth": 1.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

rng = np.random.default_rng(7)

# ---------- synthetic data: a dense cluster + a handful of outliers ----------
# gives the same "branchy tree with a few long bridges" look as the paper figure
main = rng.normal(scale=1.0, size=(45, 2))
main[:, 0] *= 1.7  # slightly elongate for visual texture

n_out = 10
angles = rng.uniform(0, 2 * np.pi, n_out)
radii = rng.uniform(3.2, 6.5, n_out)
outliers = np.c_[radii * np.cos(angles*3), radii * np.sin(angles*0.5)]

points = np.vstack([main, outliers])
n = len(points)

# ---------- MST ----------
D = squareform(pdist(points))
mst = minimum_spanning_tree(D).tocoo()
edges = list(zip(mst.row, mst.col, mst.data))
edges.sort(key=lambda e: e[2])          # increasing weight: e_(1), e_(2), ...
weights = np.array([w for _, _, w in edges])

diameter = D.max()                       # Phi(S)

# ---------- epsilon-flexibility ----------
eps = 2.0
best_psi = 0
for psi in range(0, n - 1):
    kept = n - 1 - psi
    lhs = psi * diameter
    rhs = eps * weights[:kept].sum()
    if lhs <= rhs:
        best_psi = psi                  # keep the largest psi that still satisfies it
psi = best_psi
kept_edges = edges[: n - 1 - psi]
tail_edges = edges[n - 1 - psi:]

# ================= figure =================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG)

# ---- left: MST with terminal nodes ----
axL.set_facecolor(BG)

# Draw kept edges (gray, thinner)
for i, j, w in kept_edges:
    axL.plot(*zip(points[i], points[j]),
             color=GRAY, lw=2.4, zorder=1,
             solid_capstyle='round', solid_joinstyle='round')

# Draw tail edges (coral, thicker, highlighted)
for i, j, w in tail_edges:
    axL.plot(*zip(points[i], points[j]),
             color=CORAL, lw=4.4, zorder=2,
             solid_capstyle='round', solid_joinstyle='round')

# Draw terminal nodes at edge endpoints
# For kept edges: small gray terminals
for i, j, w in kept_edges:
    axL.scatter(points[i, 0], points[i, 1],
                s=40, color=INK, zorder=3, linewidths=0.8,
                edgecolors=None, marker='o')
    axL.scatter(points[j, 0], points[j, 1],
                s=40, color=INK, zorder=3, linewidths=0.8,
                edgecolors=None, marker='o')

# For tail edges: larger coral terminals
for i, j, w in tail_edges:
    axL.scatter(points[i, 0], points[i, 1],
                s=60, color=CORAL, zorder=4, linewidths=1.2,
                edgecolors=None, marker='o')
    axL.scatter(points[j, 0], points[j, 1],
                s=60, color=CORAL, zorder=4, linewidths=1.2,
                edgecolors=None, marker='o')

axL.set_aspect("equal")
axL.axis("off")
axL.set_title("The minimum spanning tree", loc="left", fontsize=17, color=INK, pad=12)

# ---- right: edge weight segments (Tufte-style) ----
axR.set_facecolor(BG)

# Each edge is a vertical segment from y=0 to y=weight, at its rank on x
for r, (i, j, w) in enumerate(edges, start=1):
    is_tail = r > n - 1 - psi
    clr = CORAL if is_tail else GRAY
    lw  = 3.8 if is_tail else 2.4
    sz  = 50 if is_tail else 30
    # segment from (rank, 0) to (rank, weight)
    axR.plot([r, r], [0, w], color=clr, lw=lw, zorder=1,
             solid_capstyle='round')
    # start terminal at (rank, 0)
    axR.scatter(r, 0, s=sz, color=clr, zorder=2,
                edgecolors='white', linewidths=0.6)
    # end terminal at (rank, weight)
    axR.scatter(r, w, s=sz, color=clr, zorder=2,
                edgecolors='white', linewidths=0.6)

# Diameter line (horizontal, at y=diameter)
axR.axhline(diameter, color=CORAL_L, lw=1.2, ls=(0, (5, 3)), zorder=1)
axR.annotate(
    "diameter $\\Phi(S)$\n(worst-case replacement)",
    xy=(n, diameter),
    xytext=(n + 2, diameter + 0.5),
    fontsize=12, color=CORAL_L, va="bottom", ha="left",
    arrowprops=dict(arrowstyle="->", color=CORAL_L, lw=0.8),
)

# Fill region for tail edges (vertical dashed lines from weight to diameter)
if len(tail_edges):
    tail_ranks = np.arange(n - len(tail_edges), n + 1)
    tail_ws    = np.array([w for _, _, w in tail_edges])
    for r, w in zip(tail_ranks, tail_ws):
        axR.plot([r, r], [w, diameter], color=CORAL_L, lw=3.75, alpha=0.25, zorder=0)

# Legend using proxy artists
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=GRAY, lw=2.4, marker='o', markersize=5,
           markerfacecolor=GRAY, markeredgecolor=None, label='verified'),
    Line2D([0], [0], color=CORAL, lw=3.8, marker='o', markersize=7,
           markerfacecolor=CORAL, markeredgecolor=None, label='flexible tail'),
]
axR.legend(handles=legend_elements, frameon=False, fontsize=13)

axR.set_xlabel("edges, sorted by weight", fontsize=16, color=INK)
axR.set_ylabel("weight", fontsize=16, color=INK)

# Tufte-style separated spines: y-axis on the right, bottom spine away from zero
axR.spines["top"].set_visible(False)
axR.spines["left"].set_visible(False)
axR.spines["right"].set_position(("outward", 8))
axR.spines["bottom"].set_position(("outward", 8))
axR.spines["right"].set_color(INK)
axR.spines["bottom"].set_color(INK)

# Ticks outward, y-axis ticks on the right
axR.tick_params(axis="both", direction="out", colors=INK, pad=6)
axR.yaxis.set_label_position("right")
axR.yaxis.tick_right()

axR.set_title(f"The last {psi} edges are flexible ($\\varepsilon={eps:g}$)",
              loc="left", fontsize=17, color=INK, pad=12)

fig.tight_layout()
fig.savefig("fig_flexibility.pdf", transparent=True, bbox_inches="tight")
fig.savefig("fig_flexibility.png", transparent=True, dpi=400, bbox_inches="tight")
print(f"n={n}  psi={psi}  diameter={diameter:.2f}  tail edges={len(tail_edges)}")