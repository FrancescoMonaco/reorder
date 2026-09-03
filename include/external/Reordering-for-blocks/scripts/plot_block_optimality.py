#!/usr/bin/env python3
"""
Block-packing optimality plots.

Reads the flattened analysis CSV produced by scripts/parse_results.py
(results_analysis.csv, columns nonzero_blocks_{bs} etc.) and, for every
(matrix, permutation, block size) pair, computes:

    optimal_blocks = ceil(nnz / bs^2)          # perfect packing lower bound
    gap_ratio      = nonzero_blocks / optimal_blocks   # >= 1.0

No reordering can do better than `optimal_blocks` (all nonzeros stored
tightly in bs x bs blocks), so the gap ratio measures how far each
algorithm is from the theoretical optimum.

Plots (designed to stay readable with many matrices x many algorithms):
  1. gap_distribution.{png,pdf}  -- one subplot per block size; boxplot of
     the gap ratio across all matrices, one box per algorithm (sorted by
     median), log-scale y, reference line at 1.0.
  2. gap_profile_bs{bs}.{png,pdf} -- per-matrix view for one block size:
     matrices on x (sorted by nnz), one thin line per algorithm.

Usage (with the reorder env):
    ~/.micromamba/envs/reorder/bin/python plot_block_optimality.py
    ~/.micromamba/envs/reorder/bin/python plot_block_optimality.py \
        --csv results/results_analysis.csv --perm-type SYMMETRIC \
        --block-sizes 16 32 64
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / shared settings
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
REORDER_DIR = SCRIPTS_DIR.parents[0]          # .../include/external/Reordering-for-blocks
REPO_ROOT = SCRIPTS_DIR.parents[3]            # repository root
DEFAULT_CSV = REPO_ROOT / "results" / "results_analysis.csv"
DEFAULT_OUT = REORDER_DIR / "plots" / "block_optimality"

# Reuse the canonical perm -> (display, color) mapping when available
try:
    from settings import PERMS as _PERMS  # noqa: E402

    PERM_STYLE = {k: (v["display"], v["color"]) for k, v in _PERMS.items()}
except Exception:  # pragma: no cover - fallback if settings.py is unavailable
    PERM_STYLE = {}


def perm_display(perm: str) -> str:
    return PERM_STYLE.get(perm, (perm, None))[0]


def perm_color(perm: str) -> str:
    color = PERM_STYLE.get(perm, (perm, None))[1]
    return color if color else "#708090"


# Colors keyed by *display* name (plots group by display name)
ALGO_COLORS = {disp: color for _, (disp, color) in PERM_STYLE.items()}


def algo_color(algorithm: str) -> str:
    color = ALGO_COLORS.get(algorithm)
    return color if color else "#708090"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: Path, perm_type: str, block_sizes, perms, matrices):
    df = pd.read_csv(csv_path)

    block_cols = {
        int(c.rsplit("_", 1)[1]): c
        for c in df.columns
        if c.startswith("nonzero_blocks_")
    }
    if not block_cols:
        sys.exit(f"Error: no 'nonzero_blocks_*' columns found in {csv_path}")

    needed = ["matrix", "perm", "perm_type", "nnz"] + list(block_cols.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"Error: CSV is missing required columns: {missing}")

    if perm_type.upper() != "ALL":
        df = df[df["perm_type"].astype(str).str.upper() == perm_type.upper()]
    if perms:
        df = df[df["perm"].isin(perms)]
    if matrices:
        df = df[df["matrix"].isin(matrices)]

    # Keep only the latest run per (matrix, perm, perm_type), like parse_results.py
    if "job_id" in df.columns:
        df = df.sort_values("job_id")
    df = df.drop_duplicates(subset=["matrix", "perm", "perm_type"], keep="last")

    df = df.dropna(subset=["matrix", "perm", "nnz"])
    for bs in block_sizes:
        if bs not in block_cols:
            print(f"Warning: block size {bs} not present in CSV, skipping", file=sys.stderr)
            continue
        df = df.dropna(subset=[block_cols[bs]])
        df = df[df["nnz"] > 0]

    if df.empty:
        sys.exit("Error: no rows left after filtering -- check --perm-type/--perms/--matrices")

    # Long form: one row per (matrix, perm, block_size)
    rows = []
    for bs in block_sizes:
        if bs not in block_cols:
            continue
        col = block_cols[bs]
        sub = df[["matrix", "perm", "nnz", col]].copy()
        sub = sub[(sub[col] > 0) & (sub["nnz"] > 0)]
        sub = sub.rename(columns={col: "nonzero_blocks"})
        sub["block_size"] = bs
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long["optimal_blocks"] = np.ceil(
        long["nnz"] / long["block_size"].astype(float) ** 2
    ).astype(np.int64)
    long["gap_ratio"] = long["nonzero_blocks"] / long["optimal_blocks"]
    long["algorithm"] = long["perm"].map(perm_display)
    return long, sorted(block_cols.keys() & set(block_sizes))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(long: pd.DataFrame, out_csv: Path = None):
    agg = (
        long.groupby(["block_size", "perm", "algorithm"])["gap_ratio"]
        .agg(median="median", mean="mean", best="min", worst="max", n_matrices="count")
        .reset_index()
        .sort_values(["block_size", "median"])
    )
    for bs, group in agg.groupby("block_size"):
        print(f"\n=== Gap ratio (nonzero_blocks / optimal_blocks), block size {bs} ===")
        print(f"{'algorithm':<14}{'median':>10}{'mean':>10}{'best':>10}{'worst':>12}{'n':>8}")
        for _, r in group.iterrows():
            print(
                f"{r['algorithm']:<14}{r['median']:>10.2f}{r['mean']:>10.2f}"
                f"{r['best']:>10.2f}{r['worst']:>12.2f}{int(r['n_matrices']):>8}"
            )
    if out_csv:
        agg.to_csv(out_csv, index=False)
        print(f"\nSummary written to {out_csv}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Plot 1: distribution of gap ratio per algorithm, one panel per block size
# ---------------------------------------------------------------------------
def plot_distribution(long: pd.DataFrame, block_sizes, out_dir: Path, formats):
    n = len(block_sizes)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False,
        sharey=True,
    )

    for ax, bs in zip(axes.ravel(), block_sizes):
        sub = long[long["block_size"] == bs]
        order = (
            sub.groupby("algorithm")["gap_ratio"].median().sort_values().index.tolist()
        )
        data = [sub.loc[sub["algorithm"] == a, "gap_ratio"].values for a in order]
        boxes = ax.boxplot(
            data,
            tick_labels=order,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, algo in zip(boxes["boxes"], order):
            patch.set_facecolor(algo_color(algo))
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)
        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, zorder=0)
        ax.set_yscale("log")
        ax.set_title(f"block size {bs}")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("gap ratio  (blocks / optimal)")
        ax.tick_params(axis="x", rotation=60)

    # hide unused axes
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Distance from perfect packing (dashed red = theoretical optimum)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for fmt in formats:
        path = out_dir / f"gap_distribution.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}", file=sys.stderr)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: per-matrix gap profile for one block size
# ---------------------------------------------------------------------------
def plot_profile(long: pd.DataFrame, block_size: int, out_dir: Path, formats,
                 label_every: int):
    sub = long[long["block_size"] == block_size]
    if sub.empty:
        print(f"No data for block size {block_size}, skipping profile plot", file=sys.stderr)
        return

    # x order: matrices sorted by nnz
    nnz = sub.groupby("matrix")["nnz"].first().sort_values()
    matrices = nnz.index.tolist()
    x_of = {m: i for i, m in enumerate(matrices)}

    fig, ax = plt.subplots(figsize=(14, 6))
    for algo, group in sub.groupby("algorithm"):
        g = group.groupby("matrix")["gap_ratio"].mean()
        xs = np.array([x_of[m] for m in g.index])
        order = np.argsort(xs)
        ax.plot(
            xs[order], g.values[order],
            label=algo, color=algo_color(algo), linewidth=1.1, alpha=0.9,
        )

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel(f"matrices sorted by nnz ({len(matrices)} matrices)")
    ax.set_ylabel("gap ratio  (blocks / optimal)")
    ax.set_title(f"Nonzero-block count vs theoretical optimum, block size {block_size}")
    step = max(1, label_every)
    ax.set_xticks(range(0, len(matrices), step))
    ax.set_xticklabels(
        [matrices[i] for i in range(0, len(matrices), step)],
        rotation=90, fontsize=6,
    )
    ax.legend(ncol=2, fontsize=9, loc="best")
    ax.grid(True, which="major", axis="y", alpha=0.3)
    fig.tight_layout()

    for fmt in formats:
        path = out_dir / f"gap_profile_bs{block_size}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}", file=sys.stderr)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot nonzero-block counts against the perfect-packing optimum."
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Analysis CSV (default: {DEFAULT_CSV})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--perm-type", default="SYMMETRIC",
                   choices=["ROW", "SYMMETRIC", "ALL"],
                   help="Perm pipeline to use (default: SYMMETRIC)")
    p.add_argument("--block-sizes", type=int, nargs="+", default=None,
                   help="Block sizes to plot (default: all found in the CSV)")
    p.add_argument("--perms", nargs="+", default=None,
                   help="Restrict to these permutation ids (e.g. SB_rcm GROOT_reorder)")
    p.add_argument("--matrices", nargs="+", default=None,
                   help="Restrict to these matrix names")
    p.add_argument("--profile-block-size", type=int, default=32,
                   help="Block size used for the per-matrix profile plot (default: 32)")
    p.add_argument("--label-every", type=int, default=25,
                   help="Label every N-th matrix on the profile x axis (default: 25)")
    p.add_argument("--summary-csv", type=Path, default=None,
                   help="Also write the per-algorithm summary to this CSV")
    p.add_argument("--format", dest="formats", nargs="+", default=["png"],
                   choices=["png", "pdf"], help="Output format(s) (default: png)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.csv.exists():
        sys.exit(f"Error: CSV not found: {args.csv}")

    # Discover available block sizes first (needed before filtering/melting)
    probe = pd.read_csv(args.csv, nrows=0)
    all_bs = sorted(
        int(c.rsplit("_", 1)[1]) for c in probe.columns if c.startswith("nonzero_blocks_")
    )
    block_sizes = args.block_sizes if args.block_sizes else all_bs

    long, block_sizes = load_data(
        args.csv, args.perm_type, block_sizes, args.perms, args.matrices
    )
    print(
        f"Loaded {long['matrix'].nunique()} matrices x {long['algorithm'].nunique()} "
        f"algorithms, perm_type={args.perm_type}, block sizes {block_sizes}",
        file=sys.stderr,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print_summary(long, args.summary_csv)
    plot_distribution(long, block_sizes, args.out_dir, args.formats)
    plot_profile(long, args.profile_block_size, args.out_dir, args.formats,
                 args.label_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())


