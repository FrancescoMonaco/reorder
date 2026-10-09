"""Cluster matrices by SuiteSparse environment, assign each matrix the
algorithm maximizing block density, plot whether similar environments
benefit from the same algorithm.
Inputs : results/results_analysis.csv (SYMMETRIC, block_density_16)
Outputs: plots/env_algorithm_affinity/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path("/home/francescopio.monaco/reorder")
try:
    import sys
    sys.path.insert(0, str(REPO / "include/external/Reordering-for-blocks/scripts"))
    from settings import PALETTE
except Exception:
    PALETTE = ["#30123B", "#4361EE", "#28BBEC", "#1AE4B6", "#62FC6B",
               "#AAF334", "#E3CA24", "#FE912E", "#F24716", "#C71804",
               "#910502", "#000000"]

ANALYSIS_CSV = REPO / "results" / "results_analysis.csv"
OUTDIR = REPO / "plots" / "env_algorithm_affinity"
OUTDIR.mkdir(parents=True, exist_ok=True)

DENSITY_COL = "block_density_16"
ALGOS = ["CLUB_reorder", "SB_amd", "SB_gray", "SB_metis", "SB_patoh",
         "SB_rabbit", "SB_rcm", "SB_slashburn"]

ENV = {
    "1138_bus.mtx": "HB power network", "494_bus.mtx": "HB power network",
    "662_bus.mtx": "HB power network", "685_bus.mtx": "HB power network",
    "bcspwr04.mtx": "HB power network", "bcspwr05.mtx": "HB power network",
    "bcspwr06.mtx": "HB power network", "bcspwr07.mtx": "HB power network",
    "bcspwr08.mtx": "HB power network", "bcspwr09.mtx": "HB power network",
    "bcspwr10.mtx": "HB power network",
    "bcsstk02.mtx": "HB structural", "bcsstk04.mtx": "HB structural",
    "bcsstk05.mtx": "HB structural", "bcsstk06.mtx": "HB structural",
    "bcsstk08.mtx": "HB structural", "bcsstk09.mtx": "HB structural",
    "bcsstk10.mtx": "HB structural", "bcsstk11.mtx": "HB structural",
    "bcsstk19.mtx": "HB structural", "bcsstk20.mtx": "HB structural",
    "bcsstm07.mtx": "HB structural", "bcsstm08.mtx": "HB structural",
    "bcsstm09.mtx": "HB structural", "bcsstm11.mtx": "HB structural",
    "bcsstm21.mtx": "HB structural", "bcsstm23.mtx": "HB structural",
    "bcsstm24.mtx": "HB structural", "bcsstm26.mtx": "HB structural",
    "bcsstk07.mtx": "HB structural-dup", "bcsstk12.mtx": "HB structural-dup",
    "bcsstk13.mtx": "HB CFD", "PR02R.mtx": "Fluorem CFD",
    "ash292.mtx": "HB least squares",
    "arc130.mtx": "HB materials",
    "thermal2.mtx": "Schmid thermal",
    "3elt.mtx": "AG-Monien mesh", "3elt_dual.mtx": "AG-Monien mesh",
    "airfoil1.mtx": "AG-Monien mesh", "airfoil1_dual.mtx": "AG-Monien mesh",
    "grid2.mtx": "AG-Monien mesh", "grid2_dual.mtx": "AG-Monien mesh",
    "netz4504.mtx": "AG-Monien mesh", "ukerbe1.mtx": "AG-Monien mesh",
    "ukerbe1_dual.mtx": "AG-Monien mesh", "diag.mtx": "AG-Monien mesh",
}

SHORT = {"CLUB_reorder": "CLUB", "SB_amd": "AMD", "SB_gray": "Gray",
         "SB_metis": "METIS", "SB_patoh": "PaToH", "SB_rabbit": "Rabbit",
         "SB_rcm": "RCM", "SB_slashburn": "SlashBurn"}
TOL = 1e-12


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    sym = df[(df["perm_type"] == "SYMMETRIC") & (df["perm"].isin(ALGOS))].copy()
    piv = sym.pivot_table(index="matrix", columns="perm",
                          values=DENSITY_COL, aggfunc="max")
    piv = piv.reindex(columns=ALGOS)
    piv = piv[piv.notna().sum(axis=1) >= 7]  # PR02R lacks METIS only
    print(f"matrices kept: {len(piv)} / {sym['matrix'].nunique()}")
    best_val = piv.max(axis=1, skipna=True)
    winner = piv.idxmax(axis=1)
    full_tie = (piv.max(axis=1, skipna=True)
                - piv.min(axis=1, skipna=True)).abs().le(TOL)
    is_best = piv.sub(best_val, axis=0).abs().le(TOL) & piv.notna()
    n_tied = is_best.sum(axis=1)
    base = df[df["perm"].isna()].set_index("matrix")[DENSITY_COL]
    bv = pd.Series(best_val.values, index=piv.index)
    bs = piv.index.map(base)
    gain = (bv - bs) / bs * 100
    res = pd.DataFrame({
        "matrix": piv.index,
        "environment": piv.index.map(ENV),
        "best_algorithm": winner.values,
        "best_d16": bv.values,
        "baseline_d16": bs.values,
        "gain_pct": gain.values,
        "n_tied": n_tied.values,
        "all_tied": full_tie.values,
    }).sort_values(["environment", "matrix"]).reset_index(drop=True)
    res.to_csv(OUTDIR / "env_best_algorithm.csv", index=False)
    print(res["best_algorithm"].value_counts().to_string())
    print(res.groupby(["environment", "best_algorithm"]).size().to_string())
    print("full-tie:", sorted(piv.index[full_tie].tolist()))

    ct = pd.crosstab(res["environment"], res["best_algorithm"])
    ct = ct.reindex(columns=ALGOS, fill_value=0)
    env_order = res["environment"].value_counts().index.tolist()
    ct = ct.reindex(env_order)
    counts = res["environment"].value_counts()
    colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(ALGOS)}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bottom = np.zeros(len(ct))
    x = np.arange(len(ct))
    for a in ALGOS:
        v = ct[a].to_numpy()
        ax.bar(x, v, bottom=bottom, color=colors[a], label=SHORT[a],
               edgecolor="white", linewidth=0.8)
        for i, (val, b) in enumerate(zip(v, bottom)):
            if val > 0:
                ax.text(i, b + val / 2, str(int(val)), ha="center",
                        va="center", fontsize=8)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(n={counts[e]})" for e in ct.index], fontsize=9)
    ax.set_ylabel("matrices (best algorithm by block_density_16)")
    ax.set_title("Do matrices from the same environment prefer the same "
                 "algorithm? (best = max block_density_16, n=46)",
                 fontsize=11, pad=12)
    ax.set_ylim(0, bottom.max() + 1)
    ax.legend(title="winning algorithm", ncol=4, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTDIR / "env_best_algorithm_stacked.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUTDIR / "env_best_algorithm_stacked.pdf", bbox_inches="tight")
    plt.close(fig)

    cat_order = ["AG-Monien mesh", "HB power network", "HB structural",
                 "HB structural-dup", "Fluorem CFD", "HB CFD",
                 "HB least squares", "HB materials", "Schmid thermal"]
    env_blocks = [e for e in cat_order if e in set(res["environment"])]
    res2 = res.copy()
    res2["env_cat"] = pd.Categorical(res2["environment"],
                                     categories=env_blocks, ordered=True)
    res2 = res2.sort_values(["env_cat", "gain_pct"]).reset_index(drop=True)
    xx, ticks, ticklabels, bounds = [], [], [], []
    pos = 0
    for e in env_blocks:
        idx = res2.index[res2["environment"] == e].tolist()
        bounds.append(pos - 0.75)
        for j, i in enumerate(idx):
            xx.append(pos + j)
        ticks.append(pos + (len(idx) - 1) / 2)
        ticklabels.append(f"{e}\n(n={counts[e]})")
        pos += len(idx) + 1
    bounds.append(pos - 1 + 0.75)
    xx = np.array(xx)
    ymax = float(res2["gain_pct"].max())
    fig2, ax2 = plt.subplots(figsize=(16, 5.6))
    for b in bounds:
        ax2.axvline(b, color="0.8", linewidth=0.8, zorder=1)
    for i, r in res2.iterrows():
        ax2.scatter(xx[i], r["gain_pct"], s=90, color=colors[r["best_algorithm"]],
                    zorder=3, edgecolors="black" if r["all_tied"] else "white",
                    linewidths=1.4 if r["all_tied"] else 0.9)
    ax2.set_xticks(xx)
    ax2.set_xticklabels([m.replace(".mtx", "") for m in res2["matrix"]],
                        rotation=90, fontsize=7, va="top", ha="center")
    short_env = {"HB structural-dup": "HB struct-dup", "Fluorem CFD": "Fluorem CFD",
                 "HB CFD": "HB CFD", "HB least squares": "HB LSQ",
                 "HB materials": "HB mat.", "Schmid thermal": "Schmid th.",
                 "HB structural": "HB structural", "HB power network": "HB power network",
                 "AG-Monien mesh": "AG-Monien mesh"}
    # stagger header heights so narrow neighbours do not collide
    for k, (t, lab) in enumerate(zip(ticks, ticklabels)):
        env_full = env_blocks[k]
        disp = short_env.get(env_full, env_full)
        n = counts[env_full]
        y_pos = ymax * 1.08 if (k % 2 == 0) else ymax * 1.22
        ax2.text(t, y_pos, f"{disp}\n(n={n})", ha="center", va="bottom",
                 fontsize=8, style="italic",
                 bbox=dict(facecolor="0.95", edgecolor="none", pad=2))
    ax2.set_ylabel("gain of best algorithm over unordered baseline (%)")
    ax2.set_title("Per-matrix winner of block_density_16 "
                  "(black ring = all algorithms tied)", fontsize=11, pad=64)
    ax2.set_xlim(bounds[0], bounds[-1])
    ax2.set_ylim(-12, ymax * 1.52)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    handles = [Patch(facecolor=colors[a], edgecolor="white", label=SHORT[a])
               for a in ALGOS]
    ax2.legend(handles=handles, title="winning algorithm", ncol=8, fontsize=8,
               title_fontsize=8, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "per_matrix_best_algorithm.png", dpi=200,
                 bbox_inches="tight")
    fig2.savefig(OUTDIR / "per_matrix_best_algorithm.pdf", bbox_inches="tight")
    plt.close(fig2)
    summ = ct.copy()
    summ["n_matrices"] = counts.reindex(summ.index)
    summ.to_csv(OUTDIR / "env_summary.csv")
    print(f"wrote {OUTDIR}")


if __name__ == "__main__":
    main()

