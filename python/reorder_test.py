"""
reorder_test.py — Python equivalent of test/reorder_test.cpp
Uses pyNNDescent to build a k-NN graph on the masked (binary sketch)
matrix and derives row permutations via BFS traversal on that graph.
"""

import time
from collections import deque
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
# ---------------------------------------------------------------------------
def count_nonzero_blocks(A: sp.csr_matrix, bw: int, bh: int) -> int:
    occupied: set[tuple[int, int]] = set()
    n_rows = A.shape[0]
    for i in range(n_rows):
        block_row = i // bh
        for k in range(A.indptr[i], A.indptr[i + 1]):
            block_col = int(A.indices[k]) // bw
            occupied.add((block_row, block_col))
    return len(occupied)


# ---------------------------------------------------------------------------
# cluster_nndescent(Ahat, n_neighbors) → permutation P
#
# 1. Build the k-NN graph of Ahat rows using pyNNDescent (Jaccard metric).
# 2. Derive the row permutation via BFS on the k-NN adjacency.
#    No separate community-detection algorithm — just a graph walk
#    that naturally groups similar rows together.
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
    nnd = NNDescent(Ahat, metric="jaccard", n_neighbors=k, verbose=False)
    nn_indices, _ = nnd.neighbor_graph

    # BFS traversal on the k-NN adjacency → permutation
    visited = np.zeros(n, dtype=np.bool_)
    perm = np.empty(n, dtype=np.intp)
    pos = 0
    queue: deque[int] = deque()

    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        queue.append(seed)
        while queue:
            node = queue.popleft()
            perm[pos] = node
            pos += 1
            for nb in nn_indices[node]:
                if nb >= 0 and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

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
