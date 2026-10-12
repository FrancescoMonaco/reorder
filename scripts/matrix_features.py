#!/usr/bin/env python3
"""Per-matrix structural fingerprint for clustering reorderability.
One input matrix -> one output CSV row (SbatchMan-friendly, like perms.yaml).
Usage (reorder_env):
    micromamba run -n reorder_env python scripts/matrix_features.py <matrix.mtx> --out /data/matrices/features/<name>.csv
    micromamba run -n reorder_env python scripts/matrix_features.py --merge /data/matrices/features --out /data/matrices/matrix_features.csv
Design: O(nnz) or capped sampling only. No exact betweenness/closeness/eigen.
Only numpy+scipy. Rectangular/pattern/symmetric handled.
"""
from __future__ import annotations
import argparse, csv, math, sys, time
from pathlib import Path
import numpy as np
import scipy.io, scipy.sparse
from scipy.sparse.csgraph import connected_components, dijkstra

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    xs = np.sort(x); n = xs.size; cum = np.cumsum(xs)
    return float((2.0*np.sum((np.arange(n)+1)*xs))/(n*cum[-1])-(n+1)/n)

def skew_manual(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    m = x.mean(); c = x-m; m2 = np.mean(c**2)
    if m2 == 0:
        return 0.0
    return float(np.mean(c**3)/(m2**1.5))

def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else 0.0

def powerlaw_alpha(deg: np.ndarray) -> float:
    d = np.asarray(deg, dtype=np.float64); d = d[d > 0]
    if d.size < 50:
        return float("nan")
    d = np.sort(d)[::-1]
    k = max(50, d.size//5); tail = d[:k]
    if tail[-1] <= 0 or tail[0] == tail[-1]:
        return float("nan")
    r = np.arange(1, k+1, dtype=np.float64)
    lx, ly = np.log(r), np.log(tail); lx -= lx.mean()
    slope = float(np.dot(lx, ly-ly.mean())/np.dot(lx, lx))
    return float(1.0-1.0/slope) if slope < 0 else float("nan")

def pagerank_csr(adj, alpha=0.85, iters=10) -> np.ndarray:
    n = adj.shape[0]
    deg = np.diff(adj.indptr).astype(np.float64)
    dangling = deg == 0
    inv = np.zeros(n); nz = deg > 0; inv[nz] = 1.0/deg[nz]
    P = adj.copy(); P.data = np.ones_like(P.data)
    P = scipy.sparse.diags(inv) @ P
    PT = P.T.tocsr()
    r = np.full(n, 1.0/n); teleport = (1.0-alpha)/n
    for _ in range(iters):
        dang = float(r[dangling].sum()) if dangling.any() else 0.0
        r = alpha*(PT @ r + dang/n)+teleport
        s = r.sum()
        if s != 0:
            r /= s
    return r
def extract(mtx_path: Path, seed=0, n_jaccard_pairs=2000, n_bfs_sources=8,
            n_assort_edges=5000, n_tri_samples=2000, block=16,
            max_nodes_exact=50000, with_betweenness=False) -> dict:
    t_all = time.perf_counter()
    rng = np.random.default_rng(seed)
    rec: dict = {"matrix": mtx_path.name, "status": "ok"}
    t0 = time.perf_counter()
    M = scipy.io.mmread(str(mtx_path))
    A = M.tocoo() if scipy.sparse.issparse(M) else scipy.sparse.coo_matrix(np.asarray(M))
    rec["load_s"] = round(time.perf_counter()-t0, 3)
    n_rows, n_cols = A.shape; nnz = int(A.nnz)
    rec.update(n_rows=n_rows, n_cols=n_cols, nnz=nnz,
               density=float(nnz/(n_rows*n_cols)) if n_rows*n_cols else 0.0,
               is_square=int(n_rows == n_cols))
    rec.update(log_n_rows=float(math.log1p(n_rows)), log_nnz=float(math.log1p(nnz)),
               log_density=float(math.log1p(rec["density"])))
    csr = A.tocsr(); csc = A.tocsc()
    row_deg = np.diff(csr.indptr).astype(np.int64)
    col_deg = np.diff(csc.indptr).astype(np.int64)
    def deg_stats(d, pfx):
        d = d.astype(np.float64)
        rec[pfx+"_mean"] = float(d.mean()) if d.size else 0.0
        rec[pfx+"_std"] = float(d.std()) if d.size else 0.0
        rec[pfx+"_cv"] = float(d.std()/d.mean()) if d.size and d.mean() else 0.0
        rec[pfx+"_max"] = int(d.max()) if d.size else 0
        rec[pfx+"_p50"] = pct(d, 50); rec[pfx+"_p90"] = pct(d, 90); rec[pfx+"_p99"] = pct(d, 99)
        rec[pfx+"_gini"] = gini(d); rec[pfx+"_skew"] = skew_manual(d)
        rec[pfx+"_frac_empty"] = float((d == 0).mean()) if d.size else 0.0
        rec[pfx+"_frac_deg1"] = float((d == 1).mean()) if d.size else 0.0
        rec[pfx+"_frac_gt50"] = float((d > 50).mean()) if d.size else 0.0
    deg_stats(row_deg, "row"); deg_stats(col_deg, "col")
    if n_rows == n_cols and n_rows > 1:
        try:
            rec["row_col_deg_corr"] = float(np.corrcoef(row_deg.astype(float), col_deg.astype(float))[0, 1])
        except Exception:
            rec["row_col_deg_corr"] = float("nan")
    else:
        rec["row_col_deg_corr"] = float("nan")
    if n_rows == n_cols:
        try:
            B = csr.copy(); B.data[:] = 1  # structural symmetry: ignore numeric values
            if nnz <= 10_000_000:
                rec["pattern_symmetric"] = int((B != B.T).nnz == 0)
            else:
                S = (B+B.T).tocoo(); rec["pattern_symmetric"] = int(S.nnz == nnz)
        except Exception:
            rec["pattern_symmetric"] = float("nan")
    else:
        rec["pattern_symmetric"] = float("nan")
    rows = A.row.astype(np.int64); cols = A.col.astype(np.int64)
    if nnz:
        diff = np.abs(rows-cols)
        rec["bandwidth_max"] = int(diff.max()); rec["bandwidth_avg"] = float(diff.mean())
        rec["bandwidth_norm"] = float(diff.max()/max(n_rows, n_cols))
        rec["frac_nnz_band32"] = float((diff <= 32).mean())
        rec["frac_nnz_band128"] = float((diff <= 128).mean())
        rec["diag_frac"] = float((rows == cols).mean()) if n_rows == n_cols else 0.0
    else:
        rec.update(bandwidth_max=0, bandwidth_avg=0.0, bandwidth_norm=0.0,
                   frac_nnz_band32=0.0, frac_nnz_band128=0.0, diag_frac=0.0)
    indptr, indices = csr.indptr, csr.indices
    nonempty = row_deg > 0
    if nonempty.any():
        st = indptr[:-1][nonempty]; en = indptr[1:][nonempty]-1
        spreads = (indices[en]-indices[st]).astype(np.float64)
        rec["avg_row_spread"] = float(spreads.mean()); rec["max_row_spread"] = int(spreads.max())
        rec["profile"] = int(spreads.sum())
    else:
        rec.update(avg_row_spread=0.0, max_row_spread=0, profile=0)
    try:
        ci, cx = csc.indptr, csc.indices
        vd = np.diff(cx); b = ci[1:-1]-1
        v = (b >= 0) & (b < len(vd)); vd[b[v]] = -1
        consec = int(np.sum(vd == 1)); tot = int(nnz-int((col_deg > 0).sum()))
        rec["vert_adj_ratio"] = float(consec/tot) if tot > 0 else 0.0
    except Exception:
        rec["vert_adj_ratio"] = float("nan")
    try:
        cn = col_deg > 0
        if cn.any():
            cs = csc.indptr[:-1][cn]; ce = csc.indptr[1:][cn]-1
            csp = (csc.indices[ce]-csc.indices[cs]).astype(float)
            rec["avg_col_spread"] = float(csp.mean()); rec["max_col_spread"] = int(csp.max())
        else:
            rec.update(avg_col_spread=0.0, max_col_spread=0)
    except Exception:
        rec.update(avg_col_spread=float("nan"), max_col_spread=float("nan"))
    try:
        if nnz:
            br = (rows//block).astype(np.int64); bc = (cols//block).astype(np.int64)
            key = br*(int(math.ceil(n_cols/block))+1)+bc
            n_occ = int(np.unique(key).size)
            d16 = float(nnz/(n_occ*block*block))
        else:
            n_occ, d16 = 0, 0.0
        rec["block16_occupied"] = n_occ; rec["block16_density"] = d16
        rec["block_fill_gain"] = float(d16/rec["density"]) if rec["density"] else 0.0
    except Exception:
        rec.update(block16_occupied=float("nan"), block16_density=float("nan"), block_fill_gain=float("nan"))
    # Edge-sampled Jaccard: sample nonzero EDGES (u,v), score N(u) cap N(v).
    # Nonzero-biased (hubs appear proportional to degree) and identical under
    # any row/col permutation, unlike random node-pair sampling.
    try:
        if nnz:
            E = min(nnz, 2000)
            ei = rng.choice(nnz, size=E, replace=False)
            eu, ev = rows[ei], cols[ei]
            js = []
            # CSR rows only valid when square; general path uses boolean matvec.
            for u, v in zip(eu.tolist(), ev.tolist()):
                if u >= n_rows or v >= n_cols:
                    continue
                # neighbor sets: row u of CSR; column v of CSC (= row v of CSR when square)
                Nu = indices[indptr[u]:indptr[u + 1]]
                if n_rows == n_cols:
                    Nv = indices[indptr[v]:indptr[v + 1]]
                else:
                    Nv = csc.indices[csc.indptr[v]:csc.indptr[v + 1]]
                if Nu.size == 0 or Nv.size == 0:
                    continue
                if Nu.size > 5000 or Nv.size > 5000:
                    continue  # skip mega-hub pairs (cost), still unbiased at 2k samples
                inter = np.intersect1d(Nu, Nv, assume_unique=False).size
                union = Nu.size + Nv.size - inter
                js.append(inter / union if union else 0.0)
            if js:
                ja = np.array(js)
                rec["edge_jaccard_mean"] = float(ja.mean())
                rec["edge_jaccard_p90"] = float(np.percentile(ja, 90))
                rec["edge_jaccard_frac_gt05"] = float((ja > 0.5).mean())
            else:
                rec.update(edge_jaccard_mean=0.0, edge_jaccard_p90=0.0, edge_jaccard_frac_gt05=0.0)
        else:
            rec.update(edge_jaccard_mean=0.0, edge_jaccard_p90=0.0, edge_jaccard_frac_gt05=0.0)
        # legacy node-pair columns kept for continuity (deprecated, permutation-sensitive)
        rec["jaccard_mean"] = rec["edge_jaccard_mean"]
        rec["jaccard_max"] = rec["edge_jaccard_p90"]
        rec["jaccard_frac_gt05"] = rec["edge_jaccard_frac_gt05"]
    except Exception:
        rec.update(edge_jaccard_mean=float("nan"), edge_jaccard_p90=float("nan"),
                   edge_jaccard_frac_gt05=float("nan"),
                   jaccard_mean=float("nan"), jaccard_max=float("nan"),
                   jaccard_frac_gt05=float("nan"))
    square = (n_rows == n_cols)
    try:
        if square:
            S = (csr+csr.T); S.data[:] = 1; S = S.tocsr()
            n_comp, labels = connected_components(S, directed=False)
            cnt = np.bincount(labels)
            rec["n_components"] = int(n_comp); rec["largest_comp_frac"] = float(cnt.max()/n_rows)
            G = S; degG = np.diff(G.indptr).astype(float); nG = n_rows
        else:
            B = scipy.sparse.bmat([[None, csr], [csr.T, None]], format="csr"); B.data[:] = 1
            n_comp, labels = connected_components(B, directed=False)
            cnt = np.bincount(labels)
            rec["n_components"] = int(n_comp); rec["largest_comp_frac"] = float(cnt.max()/(n_rows+n_cols))
            G = B; degG = np.diff(G.indptr).astype(float); nG = n_rows+n_cols
    except Exception:
        G = None; nG = n_rows if square else n_rows+n_cols; degG = row_deg.astype(float)
        rec["n_components"] = float("nan"); rec["largest_comp_frac"] = float("nan")
    # 2-sweep pseudo-diameter: pick random start -> farthest a -> farthest b.
    # Deterministic given seed, permutation-invariant (graph distances only),
    # 2 unweighted BFS instead of 8 random-source Dijkstras.
    try:
        if G is not None and nG > 1:
            cand = np.flatnonzero(degG > 0)
            if cand.size == 0:
                raise ValueError("empty graph")
            s0 = int(rng.choice(cand, size=1)[0])
            d1 = dijkstra(G, directed=False, indices=s0, unweighted=True)
            f1 = d1[np.isfinite(d1)]
            a = int(np.argmax(d1)) if f1.size else s0
            d2 = dijkstra(G, directed=False, indices=a, unweighted=True)
            f2 = d2[np.isfinite(d2)]; f2 = f2[f2 > 0]
            pdiam = float(d2[np.isfinite(d2)].max()) if np.isfinite(d2).any() else 0.0
            rec["pseudo_diameter"] = pdiam
            rec["pseudo_diameter_norm"] = float(pdiam / math.log(nG)) if nG > 1 else 0.0
            rec["bfs_depth_mean"] = float(f2.mean()) if f2.size else 0.0
            rec["bfs_depth_max"] = float(f2.max()) if f2.size else 0.0
        else:
            rec.update(pseudo_diameter=0.0, pseudo_diameter_norm=0.0,
                       bfs_depth_mean=float("nan"), bfs_depth_max=float("nan"))
    except Exception:
        rec.update(pseudo_diameter=float("nan"), pseudo_diameter_norm=float("nan"),
                   bfs_depth_mean=float("nan"), bfs_depth_max=float("nan"))
    try:
        if G is not None and nG <= 2_000_000:
            pr = pagerank_csr(G.tocsr(), iters=10)
            rec["pagerank_max"] = float(pr.max()); rec["pagerank_std"] = float(pr.std())
        else:
            rec.update(pagerank_max=float("nan"), pagerank_std=float("nan"))
    except Exception:
        rec.update(pagerank_max=float("nan"), pagerank_std=float("nan"))
    try:
        k = max(1, int(0.01*degG.size)); top = np.partition(degG, -k)[-k:].sum()
        rec["hub1pct_coverage"] = float(top/degG.sum()) if degG.sum() else 0.0
        mu = float(degG.mean()) if degG.size else 0.0
        rec["hub_prominence"] = float(degG.max()/mu) if mu else 0.0
    except Exception:
        rec.update(hub1pct_coverage=float("nan"), hub_prominence=float("nan"))
    try:
        if square and nnz:
            cand = np.flatnonzero(row_deg >= 2)
            if cand.size:
                T = min(n_tri_samples, cand.size); us = rng.choice(cand, size=T, replace=False)
                closed = 0; tested = 0
                for u in us:
                    s, e = indptr[u], indptr[u+1]; nb = indices[s:e]
                    if nb.size < 2 or nb.size > 2000:
                        continue
                    v, w = rng.choice(nb, size=2, replace=False)
                    vs, ve = indptr[v], indptr[v+1]; seg = indices[vs:ve]
                    j = int(np.searchsorted(seg, w))
                    if j < (ve-vs) and seg[j] == w:
                        closed += 1
                    tested += 1
                rec["triangle_closure_rate"] = float(closed/tested) if tested else 0.0
            else:
                rec["triangle_closure_rate"] = 0.0
        else:
            rec["triangle_closure_rate"] = 0.0
    except Exception:
        rec["triangle_closure_rate"] = float("nan")
    try:
        rec["powerlaw_alpha"] = powerlaw_alpha(row_deg if square else np.concatenate([row_deg, col_deg]))
    except Exception:
        rec["powerlaw_alpha"] = float("nan")
    try:
        if nnz and square:
            E = min(n_assort_edges, nnz); ei = rng.choice(nnz, size=E, replace=False)
            du = row_deg[rows[ei]].astype(float); dv = row_deg[cols[ei]].astype(float)
            rec["assortativity"] = float(np.corrcoef(du, dv)[0, 1]) if du.std() and dv.std() else 0.0
        else:
            rec["assortativity"] = float("nan")
    except Exception:
        rec["assortativity"] = float("nan")
    if with_betweenness and square and n_rows <= max_nodes_exact:
        try:
            import networkx as nx
            k = int(min(n_rows, 256)); g = nx.from_scipy_sparse_array(csr, create_using=nx.Graph)
            bw = nx.betweenness_centrality(g, k=k, seed=seed); v = np.array(list(bw.values()))
            rec["approx_betweenness_mean"] = float(v.mean()); rec["approx_betweenness_max"] = float(v.max())
        except Exception:
            rec.update(approx_betweenness_mean=float("nan"), approx_betweenness_max=float("nan"))
    else:
        rec["approx_betweenness_mean"] = float("nan"); rec["approx_betweenness_max"] = float("nan")
    rec["feat_s"] = round(time.perf_counter()-t_all, 3)
    return rec
COLS = ["matrix","status","n_rows","n_cols","nnz","density","is_square","pattern_symmetric",
"log_n_rows","log_nnz","log_density",
"row_mean","row_std","row_cv","row_max","row_p50","row_p90","row_p99","row_gini","row_skew",
"row_frac_empty","row_frac_deg1","row_frac_gt50",
"col_mean","col_std","col_cv","col_max","col_p50","col_p90","col_p99","col_gini","col_skew",
"col_frac_empty","col_frac_deg1","col_frac_gt50","row_col_deg_corr",
"bandwidth_max","bandwidth_avg","bandwidth_norm","frac_nnz_band32","frac_nnz_band128","diag_frac",
"avg_row_spread","max_row_spread","profile","avg_col_spread","max_col_spread","vert_adj_ratio",
"block16_occupied","block16_density","block_fill_gain",
"jaccard_mean","jaccard_max","jaccard_frac_gt05",
"edge_jaccard_mean","edge_jaccard_p90","edge_jaccard_frac_gt05",
"n_components","largest_comp_frac","bfs_depth_mean","bfs_depth_max",
"pseudo_diameter","pseudo_diameter_norm",
"pagerank_max","pagerank_std","hub1pct_coverage","hub_prominence",
"triangle_closure_rate","powerlaw_alpha","assortativity",
"approx_betweenness_mean","approx_betweenness_max","load_s","feat_s"]
def write_row(rec, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerow({c: rec.get(c, "") for c in COLS})
def merge_dir(indir, out):
    files = sorted(indir.glob("*.csv"))
    if not files:
        sys.exit(f"no *.csv in {indir}")
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for fp in files:
            with open(fp, newline="") as fi:
                for row in csv.DictReader(fi):
                    w.writerow({c: row.get(c, "") for c in COLS})
    print(f"merged {len(files)} files -> {out}")
def main() -> int:
    ap = argparse.ArgumentParser(description="Matrix structural fingerprint (one row per matrix).")
    ap.add_argument("matrix", type=Path, nargs="?", help="Path to input .mtx (omit with --merge).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--merge", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--with-betweenness", action="store_true")
    ap.add_argument("--max-nodes-exact", type=int, default=50000)
    a = ap.parse_args()
    if a.merge is not None:
        if a.out is None:
            sys.exit("--merge requires --out <merged.csv>")
        merge_dir(a.merge, a.out); return 0
    if a.matrix is None:
        sys.exit("give a matrix path, or use --merge DIR --out MERGED.csv")
    if not a.matrix.exists():
        sys.exit(f"not found: {a.matrix}")
    try:
        rec = extract(a.matrix, seed=a.seed, block=a.block,
                      max_nodes_exact=a.max_nodes_exact, with_betweenness=a.with_betweenness)
    except Exception as e:
        rec = {"matrix": a.matrix.name, "status": f"error: {type(e).__name__}: {e}"}
        for c in COLS:
            rec.setdefault(c, "")
    if a.out is None:
        w = csv.DictWriter(sys.stdout, fieldnames=COLS, extrasaction="ignore")
        w.writeheader(); w.writerow({c: rec.get(c, "") for c in COLS})
    else:
        write_row(rec, a.out); print(f"wrote {a.out} ({rec.get('status')}, {rec.get('feat_s','?')}s)")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

