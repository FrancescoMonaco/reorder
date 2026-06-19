"""
reorder_test.py — Python equivalent of test/reorder_test.cpp
Uses pyNNDescent to build a k-NN graph on the masked (binary sketch)
matrix and derives row permutations via greedy nearest-neighbor chain
traversal on that graph.
"""

import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
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
# 2. Derive the row permutation via greedy nearest-neighbor chain:
#    from the current row, always step to the closest unvisited neighbor.
#    This keeps rows with similar sparsity patterns adjacent without
#    running any separate community-detection algorithm.
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
    # nn_indices[i] and nn_distances[i] are sorted by distance (nearest first)
    nnd = NNDescent(Ahat, metric="jaccard", n_neighbors=k, verbose=False)
    nn_indices, nn_distances = nnd.neighbor_graph

    # ------------------------------------------------------------------
    # Greedy nearest-neighbor chain traversal
    #
    # From the current node, always jump to the nearest unvisited
    # neighbour in its k-NN list.  When all k neighbours have been
    # visited, advance a linear fallback pointer to find the next
    # unvisited row and start a new chain segment.
    #
    # Amortised O(n·k) because the fallback pointer only moves forward.
    # ------------------------------------------------------------------
    visited = np.zeros(n, dtype=np.bool_)
    perm = np.empty(n, dtype=np.intp)
    pos = 0
    fallback = 0  # linear scan pointer — only advances

    current = 0
    visited[current] = True
    perm[pos] = current
    pos += 1

    while pos < n:
        # Try to step to the nearest unvisited neighbour
        found = False
        for j in range(k):
            nb = nn_indices[current, j]
            if nb >= 0 and not visited[nb]:
                current = nb
                visited[current] = True
                perm[pos] = current
                pos += 1
                found = True
                break

        if not found:
            # All k neighbours already visited — advance fallback pointer
            while fallback < n and visited[fallback]:
                fallback += 1
            if fallback < n:
                current = fallback
                visited[current] = True
                perm[pos] = current
                pos += 1

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
        reorder(A, W=1, n_neighbors=15)
        elapsed = time.perf_counter() - start
        print(f"  Reordering completed | Time (s) = {elapsed:.4f}")
        print()


if __name__ == "__main__":
    main()
