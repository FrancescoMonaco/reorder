#!/usr/bin/env python3
"""Initial clustering of matrix fingerprints to recover SuiteSparse families.
Merges /data/matrices/features/*.csv, parses GROUP + kind from each .mtx header,
log-scales heavy tails, imputes, scales, KMeans sweep (k=2..15) + Agglomerative
baseline, UMAP viz (viz only, never for clustering), 3 concise figures.
Usage (reorder_env):
    micromamba run -n reorder_env python scripts/cluster_matrices.py
Outputs: /data/matrices/matrix_features.csv, plots/matrix_clusters/*, matrix_clusters.csv
"""
from __future__ import annotations
import glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import umap

FEAT_DIR = Path("/data/matrices/features")
MTX_LIST = Path("/data/matrices/datasets/matrices_extended.txt")
MERGED = Path("/data/matrices/matrix_features.csv")
OUTDIR = Path("/home/francescopio.monaco/reorder/plots/matrix_clusters")
LOG_COLS = ["nnz", "density"]
DROP_COLS = ["matrix", "status", "load_s", "feat_s", "approx_betweenness_mean", "approx_betweenness_max"]
TOP_N_GROUPS = 8
# Trimmed set: one feature per idea (see analysis: drop col-twins, maxes, size dupes).
# Permutation-INVARIANT clustering set. Permutation-dependent baselines
# (bandwidth_*, *_spread, profile, vert_adj_ratio, block16_*) are still
# computed by matrix_features.py but EXCLUDED here -- they serve only as
# regression targets ("initial vs reordered gain").
TRIMMED = ["log_n_rows", "log_nnz", "log_density",
           "row_mean", "row_cv", "row_gini", "row_frac_deg1", "row_frac_gt50", "row_col_deg_corr",
           "hub_prominence",
           "edge_jaccard_mean", "edge_jaccard_p90",
           "largest_comp_frac", "pseudo_diameter_norm", "hub1pct_coverage",
           "triangle_closure_rate", "assortativity"]
TRIMMED_OUT = Path("/home/francescopio.monaco/reorder/plots/matrix_clusters_invariant")

# Vision-model pruners used in SPRINkLE config names (longest first so "erk"
# is not shadowed by "er"). Used to peel "<arch>-<pruner>-<sparsity>-...".
_VISION_PRUNERS = ("erk", "er", "grasp", "prospr", "snip", "synflow", "uniform")
_ARCH_RE = re.compile(
    r"^(?P<arch>.+?)-(?:" + "|".join(_VISION_PRUNERS) +
    r")-[0-9.]+-CIFAR-10-seed[0-9]+$"
)

def parse_sprinkle_stem(stem: str) -> tuple[str, str, str, str]:
    """Parse a SPRINkLE matrix stem, e.g.
    SPRINkLE__CONV-6-er-0.8-CIFAR-10-seed0_matrices__matrices__er__convs1_0
    -> (arch, config, pruner, layer).
    arch: model architecture family (CONV-6, Resnet-20, Wide-Resnet-28-2,
          meta-llama-Llama-3.1-8B-Instruct, google-gemma-2-9b-it).
    config: model+pruner+sparsity (+dataset/seed for vision models).
    pruner: pruning method (er, wanda, 2ssp, ...).
    layer:  leaf layer name (convs1_0, model_layers_1_mlp_up_proj, ...)."""
    parts = stem.split("__")
    config = parts[1] if len(parts) > 1 else stem
    if config.endswith("_matrices"):
        config = config[: -len("_matrices")]
    pruner = parts[3] if len(parts) > 3 else "Unknown"
    layer = "__".join(parts[4:]) if len(parts) > 4 else ""
    m = _ARCH_RE.match(config)
    arch = m.group("arch") if m else config  # LLM configs have no CIFAR suffix
    return arch, config, pruner, layer

def parse_header(mtx: Path) -> tuple[str, str]:
    grp, kind = "Unknown", "Unknown"
    try:
        with open(mtx, errors="ignore") as f:
            for i, line in enumerate(f):
                if line.startswith("% name:"):
                    m = re.match(r"%\s*name:\s*([^/\s]+)/", line)
                    if m:
                        grp = m.group(1).strip()
                elif line.startswith("% kind:"):
                    kind = line.split("kind:",1)[1].strip()
                if not line.startswith("%") and i > 5:
                    break
    except Exception:
        pass
    # SPRINkLE matrices (written by scipy.io.mmwrite) carry no SuiteSparse
    # header, so recover the family + pruner from the stem instead.
    if grp == "Unknown" and mtx.stem.startswith("SPRINkLE__"):
        arch, config, pruner, layer = parse_sprinkle_stem(mtx.stem)
        return arch, pruner
    return grp, kind

def load(feat_dir: Path | None = None, mtx_list: Path | None = None,
         merged: Path | None = None) -> pd.DataFrame:
    feat_dir = Path(feat_dir) if feat_dir else FEAT_DIR
    mtx_list = Path(mtx_list) if mtx_list else MTX_LIST
    merged = Path(merged) if merged else MERGED
    files = sorted(glob.glob(str(feat_dir/"*.csv")))
    assert files, f"no csvs in {feat_dir}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(merged, index=False); print(f"merged {len(files)} -> {merged} shape={df.shape}")
    paths = [l.strip() for l in open(mtx_list) if l.strip()]
    by_stem = {}
    for p in paths:
        by_stem.setdefault(Path(p).stem, p)
    groups, kinds = [], []
    for name in df["matrix"]:
        stem = Path(str(name)).stem
        p = by_stem.get(stem)
        if p and Path(p).exists():
            g, k = parse_header(Path(p))
        elif stem.startswith("SPRINkLE__"):
            g, k = parse_sprinkle_stem(stem)[0], parse_sprinkle_stem(stem)[2]
        else:
            g, k = "Unknown", "Unknown"
        groups.append(g); kinds.append(k)
    df["group"] = groups; df["kind"] = kinds
    print("top groups:"); print(df["group"].value_counts().head(15).to_string())
    return df


def preprocess(df: pd.DataFrame, feats: list[str] | None = None):
    if feats is None:
        feats = [c for c in df.columns if c not in DROP_COLS+["group","kind"]]
    else:
        feats = [c for c in feats if c in df.columns]
    X = df[feats].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in LOG_COLS:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
    med = X.median()
    X = X.fillna(med).fillna(0.0)
    Xs = StandardScaler().fit_transform(X.values)
    return feats, Xs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["full", "trimmed"], default="trimmed")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--feature-dir", default=None, help="dir of per-matrix CSV fingerprints (default: FEAT_DIR)")
    ap.add_argument("--mtx-list", default=None, help="list of .mtx paths for group labels (default: MTX_LIST)")
    ap.add_argument("--merged", default=None, help="output merged csv (default: MERGED)")
    ap.add_argument("--outdir", default=None, help="output dir for plots/csvs (overrides feature-set default)")
    ap.add_argument("--title", default="matrix fingerprints", help="dataset name for plot titles")
    ap.add_argument("--group-label", default="SuiteSparse group", help="x/legend label for the group axis")
    a = ap.parse_args()
    outdir = Path(a.outdir) if a.outdir else (TRIMMED_OUT if a.feature_set == "trimmed" else OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(a.feature_dir, a.mtx_list, a.merged)
    feats, Xs = preprocess(df, TRIMMED if a.feature_set == "trimmed" else None)
    print(f"features ({a.feature_set}): {len(feats)}: {feats}")
    # --- KMeans sweep kmin..kmax + Agglomerative baseline ---
    ks, sils, aris, nmis, inertias = [], [], [], [], []
    labels_by_k = {}
    # encode groups for ARI/NMI (including Other tail)
    gcode = pd.Categorical(df["group"]).codes
    for k in range(a.kmin, a.kmax + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
        lab = km.labels_
        labels_by_k[k] = lab
        ks.append(k); inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(Xs, lab)))
        aris.append(float(adjusted_rand_score(gcode, lab)))
        nmis.append(float(normalized_mutual_info_score(gcode, lab)))
    sweep = pd.DataFrame({"k": ks, "silhouette": sils, "ARI_vs_group": aris,
                          "NMI_vs_group": nmis, "inertia": inertias})
    sweep.to_csv(outdir/"k_sweep.csv", index=False)
    print(sweep.to_string(index=False))
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    print(f"best k by silhouette: {best_k}")
    agg = AgglomerativeClustering(n_clusters=best_k).fit(Xs)
    print(f"Agglomerative(k={best_k}) ARI={adjusted_rand_score(gcode, agg.labels_):.3f} "
          f"NMI={normalized_mutual_info_score(gcode, agg.labels_):.3f}")
    df["cluster"] = labels_by_k[best_k]
    df["cluster_agg"] = agg.labels_
    # --- UMAP (viz only) ---
    emb = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=0).fit_transform(Xs)
    df["umap1"] = emb[:, 0]; df["umap2"] = emb[:, 1]
    df.to_csv(outdir/"matrix_clusters.csv", index=False)
    # lump small groups for clean legend
    top = df["group"].value_counts().head(TOP_N_GROUPS).index.tolist()
    df["group_top"] = df["group"].where(df["group"].isin(top), "Other")
    # --- Fig1: UMAP by cluster ---
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for c in sorted(df["cluster"].unique()):
        s = df[df["cluster"] == c]
        ax.scatter(s["umap1"], s["umap2"], s=14, alpha=0.8, label=f"C{c} (n={len(s)})")
        # annotate top-3 groups in cluster
        tg = s["group"].value_counts().head(3)
        ax.text(s["umap1"].median(), s["umap2"].median(), "\n".join(tg.index.tolist()),
                fontsize=6, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    ax.set_title(f"UMAP of {len(df)} {a.title} (KMeans k={best_k})")
    ax.set_xlabel("umap1"); ax.set_ylabel("umap2")
    ax.legend(fontsize=7, markerscale=1.5, title="cluster")
    fig.tight_layout(); fig.savefig(outdir/"umap_by_cluster.png", dpi=200); plt.close(fig)
    # --- Fig2: UMAP by true group ---
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    order = list(top)+(["Other"] if (df["group_top"] == "Other").any() else [])
    for g in order:
        s = df[df["group_top"] == g]
        ax.scatter(s["umap1"], s["umap2"], s=14, alpha=0.8, label=f"{g} (n={len(s)})")
    ax.set_title(f"Same UMAP colored by {a.group_label} (top {TOP_N_GROUPS} + Other)")
    ax.set_xlabel("umap1"); ax.set_ylabel("umap2")
    ax.legend(fontsize=7, markerscale=1.5, title="group")
    fig.tight_layout(); fig.savefig(outdir/"umap_by_group.png", dpi=200); plt.close(fig)
    # --- Fig3: contingency heatmap + silhouette inset ---
    ct = pd.crosstab(df["cluster"], df["group_top"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [3, 1.2]})
    im = ax1.imshow(ct.values, aspect="auto", cmap="Blues")
    ax1.set_xticks(range(len(ct.columns))); ax1.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(ct.index))); ax1.set_yticklabels([f"C{c}" for c in ct.index], fontsize=8)
    ax1.set_title(f"Cluster x {a.group_label} counts")
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            v = ct.values[i, j]
            if v > 0:
                ax1.text(j, i, str(v), ha="center", va="center", fontsize=7,
                         color="white" if v > ct.values.max()/2 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax2.plot(sweep["k"], sweep["silhouette"], "o-", label="silhouette")
    ax2.plot(sweep["k"], sweep["ARI_vs_group"], "s--", label="ARI vs group")
    ax2.axvline(best_k, color="red", ls=":", label=f"best k={best_k}")
    ax2.set_xlabel("k"); ax2.set_title("KMeans sweep"); ax2.legend(fontsize=8); ax2.grid(ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(outdir/"contingency_and_sweep.png", dpi=200); plt.close(fig)
    # --- per-cluster fingerprint means for reordering hypotheses ---
    probe = ["edge_jaccard_mean","hub_prominence","pseudo_diameter_norm","row_gini","hub1pct_coverage",
             "n_rows","nnz","triangle_closure_rate","assortativity"]
    prof = df.groupby("cluster")[[c for c in probe if c in df.columns]].mean().round(4)
    prof["n"] = df.groupby("cluster").size()
    prof.to_csv(outdir/"cluster_profile.csv")
    print(prof.to_string())
    print(f"wrote {outdir}")
if __name__ == "__main__":
    main()
