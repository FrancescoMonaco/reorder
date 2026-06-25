"""
reorder_test.py — Python equivalent of test/reorder_test.cpp
Uses pyNNDescent to build a k-NN graph on the masked (binary sketch)
matrix, then applies Reverse Cuthill-McKee on the k-NN adjacency
to produce a bandwidth-reducing row permutation.
"""

import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from pynndescent import NNDescent


# ---------------------------------------------------------------------------
# mask(A, W) → Ahat
# Exact replica of the C++ club::mask().
# For each row, column indices are divided by W to get window indices,
# then deduplicated.  The result is a binary CSR matrix (values = 1).
# ---------------------------------------------------------------------------
def mask(A: sp.csr_matrix, W: int) -> sp.csr_matrix:
    n_rows = A.shape[0]
    n_cols_hat = int(np.ceil(A.shape[1] / W))

    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0

    col_lists: list[np.ndarray] = []
    for i in range(n_rows):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        if row_start == row_end:
            col_lists.append(np.array([], dtype=np.int64))
            indptr[i + 1] = indptr[i]
            continue
        windows = np.unique(A.indices[row_start:row_end] // W)
        col_lists.append(windows)
        indptr[i + 1] = indptr[i] + len(windows)

    indices = np.concatenate(col_lists) if col_lists else np.array([], dtype=np.int64)
    data = np.ones(len(indices), dtype=np.uint8)

    Ahat = sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols_hat))
    return Ahat


# ---------------------------------------------------------------------------
# count_nonzero_blocks(A, bw, bh)
# Matches the C++ club::count_nonzero_blocks for CSR.
# Counts distinct (block_row, block_col) pairs occupied by at least one nz.
# Vectorised — no Python-level loop over individual nonzeros.
# ---------------------------------------------------------------------------
def count_nonzero_blocks(A: sp.csr_matrix, bw: int, bh: int) -> int:
    coo = A.tocoo()
    block_rows = coo.row // bh
    block_cols = coo.col // bw
    # Pack into a single int64 for fast deduplication
    keys = block_rows.astype(np.int64) * (int(np.ceil(A.shape[1] / bw)) + 1) + block_cols
    return int(np.unique(keys).shape[0])


# ---------------------------------------------------------------------------
# cluster_nndescent(Ahat, n_neighbors) → permutation P
#
# 1. Build the k-NN graph of Ahat rows using pyNNDescent (Jaccard metric).
# 2. Convert the k-NN index lists into a symmetric sparse adjacency matrix.
# 3. Apply Reverse Cuthill-McKee (RCM) on that adjacency — a BFS variant
#    purpose-built for bandwidth/profile reduction.  Starts from a
#    peripheral node and visits neighbours in ascending-degree order.
#    O(n + edges), no separate community-detection algorithm.
# ---------------------------------------------------------------------------
def cluster_nndescent(Ahat: sp.csr_matrix, n_neighbors: int = 15) -> np.ndarray:
    n = Ahat.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.intp)

    # Clamp n_neighbors to n-1 (pyNNDescent requirement)
    k = min(n_neighbors, n - 1)
    if k < 2:
        return np.arange(n, dtype=np.intp)

    # Build the NN graph — sparse input, Jaccard metric
    nnd = NNDescent(Ahat, metric="hamming", n_neighbors=k, verbose=False)
    nn_indices, _ = nnd.neighbor_graph

    # ------------------------------------------------------------------
    # Build symmetric sparse adjacency from the k-NN index lists.
    # Edge (i, j) exists if i is among j's k-NN OR j is among i's k-NN.
    # ------------------------------------------------------------------
    src = np.repeat(np.arange(n, dtype=np.int32), k)
    dst = nn_indices.ravel().astype(np.int32)

    # Drop self-loops and any invalid sentinel indices
    valid = (dst >= 0) & (dst < n) & (dst != src)
    src, dst = src[valid], dst[valid]

    adj = sp.csr_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)),
        shape=(n, n),
    )
    # Symmetrise: union of directed k-NN edges
    adj = adj + adj.T
    adj.data = np.minimum(adj.data, 1.0)  # binary adjacency

    # ------------------------------------------------------------------
    # Reverse Cuthill-McKee — bandwidth-reducing permutation.
    # This is a BFS from a pseudo-peripheral node, visiting neighbours
    # in ascending-degree order.  O(n + nnz(adj)).
    # ------------------------------------------------------------------
    perm = reverse_cuthill_mckee(adj, symmetric_mode=True)
    return perm


# ---------------------------------------------------------------------------
# permute(A, P) → A with rows reordered by permutation P
# Matches the C++ club::permute for CSR.
# ---------------------------------------------------------------------------
def permute(A: sp.csr_matrix, P: np.ndarray) -> sp.csr_matrix:
    return A[P]


# ---------------------------------------------------------------------------
# Main — mirrors reorder_test.cpp
# ---------------------------------------------------------------------------
def reorder(A: sp.csr_matrix, W: int, n_neighbors: int = 15) -> sp.csr_matrix:
    W_mes = 32

    blocks_before = count_nonzero_blocks(A, W_mes, W_mes)
    print(f"  Starting reordering | nonzero blocks = {blocks_before}")

    Ahat = mask(A, W)
    print(f"  Masking completed   | sketch shape = {Ahat.shape}")

    P = cluster_nndescent(Ahat, n_neighbors)
    print(f"  Clustering done     | permutation length = {len(P)}")

    A_perm = permute(A, P)

    blocks_after = count_nonzero_blocks(A_perm, W_mes, W_mes)
    print(f"  Permutation applied | nonzero blocks = {blocks_after}")

    return A_perm


def main():
    test_matrices = [
        "matrices/1138_bus/1138_bus.mtx",
        "matrices/ash292/ash292.mtx",
        "matrices/lp_stocfor3/lp_stocfor3.mtx",
        "matrices/thermal2/thermal2.mtx",
        "matrices/webbase-2001/webbase-2001.mtx",
        "matrices/indochina-2004/indochina-2004.mtx",
        "matrices/kmer_V2a/kmer_V2a.mtx",
        "matrices/PR02R/PR02R.mtx",
    ]

    for path_str in test_matrices:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: Could not open file {path_str}")
            continue

        print(f"Processing {path_str}")
        A = sio.mmread(str(path))
        A = sp.csr_matrix(A)
        print(f"  rows={A.shape[0]}, cols={A.shape[1]}, nnz={A.nnz}")

        start = time.perf_counter()
        reorder(A, W=32, n_neighbors=5)
        elapsed = time.perf_counter() - start
        print(f"  Reordering completed | Time (s) = {elapsed:.4f}")
        print()


if __name__ == "__main__":
    main()
