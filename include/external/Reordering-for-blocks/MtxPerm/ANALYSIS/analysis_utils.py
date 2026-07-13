#!/usr/bin/env python3
"""
Utility functions for matrix analysis.
"""

import numpy as np
from scipy.sparse import coo_matrix


def analyze_block_structure(A_scipy, block_sizes=None):
    """
    Analyze block structure at multiple block sizes.
    
    Args:
        A_scipy: Scipy sparse matrix
        block_sizes: List of block sizes to analyze (default: powers of 2 from 4 to 128)
        
    Returns:
        List of dictionaries with block analysis results
    """
    if block_sizes is None:
        block_sizes = [4, 8, 16, 32, 64, 128]
    
    m, n = A_scipy.shape
    
    # Extract indices once for all block sizes
    if not isinstance(A_scipy, coo_matrix):
        A_coo = A_scipy.tocoo()
    else:
        A_coo = A_scipy
        
    rows, cols = A_coo.row, A_coo.col
    
    block_analysis = []
    
    for bs in block_sizes:
        # Calculate number of block rows and columns
        block_rows = (m + bs - 1) // bs
        block_cols = (n + bs - 1) // bs
        total_blocks = block_rows * block_cols
        
        # Compute block indices using numpy vectorization
        block_row_indices = rows // bs
        block_col_indices = cols // bs
        
        # Count unique (block_row, block_col) pairs efficiently
        # Use pairing function to create unique integers, then count unique values
        # This is faster than np.unique on 2D array for large matrices
        # Note: block_ids can be large, so ensure we use int64 if needed, 
        # but python handles large ints automatically.
        block_ids = block_row_indices.astype(np.int64) * block_cols + block_col_indices
        unique_block_ids = np.unique(block_ids)
        nonzero_blocks = len(unique_block_ids)

        # Calculate density within nonzero blocks (efficiency of BSR)
        # total nonzero / total nonzero block area
        total_bsr_elements = nonzero_blocks * bs * bs
        block_density = A_coo.nnz / total_bsr_elements if total_bsr_elements > 0 else 0

        # Calculate blocks per block row from the already-unique block_ids
        unique_block_row_ids = unique_block_ids // block_cols
        _, blocks_per_row = np.unique(unique_block_row_ids, return_counts=True)
        
        max_blocks_per_row = int(np.max(blocks_per_row)) if len(blocks_per_row) > 0 else 0
        avg_blocks_per_row = float(np.mean(blocks_per_row)) if len(blocks_per_row) > 0 else 0.0
        
        block_info = {
            "block_size": bs,
            "block_rows": block_rows,
            "block_cols": block_cols,
            "total_blocks": total_blocks,
            "nonzero_blocks": nonzero_blocks,
            "block_density": block_density,
            "max_blocks_per_row": max_blocks_per_row,
            "avg_blocks_per_row": avg_blocks_per_row
        }
        
        block_analysis.append(block_info)
    
    return block_analysis

def analyze_bandwidth(A_scipy):
    """
    Calculate bandwidth metrics for a sparse matrix.
    
    Args:
        A_scipy: Scipy sparse matrix
        
    Returns:
        Dictionary with bandwidth metrics
    """
    if not isinstance(A_scipy, coo_matrix):
        A_coo = A_scipy.tocoo()
    else:
        A_coo = A_scipy
        
    rows, cols = A_coo.row, A_coo.col
    
    if len(rows) == 0:
        return {
            "bandwidth_lower": 0,
            "bandwidth_upper": 0,
            "bandwidth_total": 0,
            "bandwidth_avg": 0.0
        }
    
    # Calculate distance from diagonal
    diffs = cols - rows
    
    # Bandwidth definitions
    # Lower bandwidth: max(i - j) for i > j  => max(-diffs) where diffs < 0
    # Upper bandwidth: max(j - i) for j > i  => max(diffs) where diffs > 0
    
    min_diff = np.min(diffs)
    max_diff = np.max(diffs)
    
    bw_lower = abs(min_diff) if min_diff < 0 else 0
    bw_upper = max_diff if max_diff > 0 else 0
    # Alternative definition: max(|i-j|)
    max_dist = np.max(np.abs(diffs))
    
    # Average distance from diagonal
    avg_dist = np.mean(np.abs(diffs))
    
    return {
        "bandwidth_lower": int(bw_lower),
        "bandwidth_upper": int(bw_upper),
        "bandwidth_max": int(max_dist), # Often called just "bandwidth"
        "bandwidth_avg": float(avg_dist)
    }

import functools

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _make_reuse_distance_numba():
    """Build and return numba-jitted reuse distance kernel."""
    @njit(cache=True)
    def _compute_reuse(indices, n_cols):
        nnz = len(indices)
        tree_size = nnz + 2  # 1-indexed Fenwick tree

        tree = np.zeros(tree_size, dtype=np.int64)
        last_seen = -np.ones(n_cols, dtype=np.int64)  # -1 = not seen
        reuse_dists = np.empty(nnz, dtype=np.int64)
        rd_count = 0
        cold_misses = 0

        for pos_0 in range(nnz):
            col = indices[pos_0]
            pos = pos_0 + 1  # 1-indexed

            prev = last_seen[col]
            if prev != -1:
                # prefix(pos - 1) - prefix(prev)
                rd = np.int64(0)
                i = pos - 1
                while i > 0:
                    rd += tree[i]
                    i -= i & (-i)
                i = prev
                while i > 0:
                    rd -= tree[i]
                    i -= i & (-i)
                reuse_dists[rd_count] = rd
                rd_count += 1
                # remove old marker
                i = prev
                while i < tree_size:
                    tree[i] -= 1
                    i += i & (-i)
            else:
                cold_misses += 1

            # place marker at current position
            i = pos
            while i < tree_size:
                tree[i] += 1
                i += i & (-i)
            last_seen[col] = pos

        return reuse_dists[:rd_count], cold_misses

    return _compute_reuse


def _reuse_distance_python(indices, n_cols):
    """Pure-python fallback for reuse distance (slow for large nnz)."""
    nnz = len(indices)
    tree_size = nnz + 2
    tree = np.zeros(tree_size, dtype=np.int64)

    def _update(i, delta):
        while i < tree_size:
            tree[i] += delta
            i += i & (-i)

    def _prefix(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    last_seen = {}
    reuse_distances = []
    cold_misses = 0

    for pos_0 in range(nnz):
        col = int(indices[pos_0])
        pos = pos_0 + 1

        if col in last_seen:
            prev = last_seen[col]
            rd = _prefix(pos - 1) - _prefix(prev)
            reuse_distances.append(rd)
            _update(prev, -1)
        else:
            cold_misses += 1

        _update(pos, 1)
        last_seen[col] = pos

    if reuse_distances:
        return np.array(reuse_distances, dtype=np.int64), cold_misses
    return np.empty(0, dtype=np.int64), cold_misses


@functools.lru_cache(maxsize=1)
def _get_reuse_kernel():
    """Return the best available reuse-distance kernel (cached after first call)."""
    if _HAS_NUMBA:
        return _make_reuse_distance_numba()
    return _reuse_distance_python


def _array_stats(arr):
    """Compute mean/median/max/std for a numeric array, or zeros if empty."""
    if len(arr) == 0:
        return 0.0, 0.0, 0, 0.0
    return float(np.mean(arr)), float(np.median(arr)), int(np.max(arr)), float(np.std(arr))


def analyze_access_distances(A_scipy):
    """
    Compute reuse distance and index distance statistics from the CSR access stream.

    Reuse distance: number of distinct column indices accessed between two
    consecutive accesses to the same column. Captures temporal locality
    (cache reuse of the x-vector in SpMV/SpMM).

    Index distance: absolute difference between consecutive column indices
    in the CSR row-by-row access stream. Captures spatial locality (cache
    line reuse).

    Args:
        A_scipy: Scipy sparse matrix

    Returns:
        Dictionary with reuse distance and index distance statistics.
    """
    A_csr = A_scipy.tocsr()
    if not A_csr.has_sorted_indices:
        A_csr.sort_indices()

    indices = A_csr.indices  # full column-index access stream in CSR order
    nnz = len(indices)

    if nnz == 0:
        return {
            "reuse_distance_mean": 0.0,
            "reuse_distance_median": 0.0,
            "reuse_distance_max": 0,
            "reuse_distance_std": 0.0,
            "reuse_distance_count": 0,
            "reuse_distance_cold_misses": 0,
            "index_distance_mean": 0.0,
            "index_distance_median": 0.0,
            "index_distance_max": 0,
            "index_distance_std": 0.0,
        }

    indices_i64 = indices.astype(np.int64)

    # Index distance: |col[i+1] - col[i]| across full CSR stream
    idx_dists = np.abs(np.diff(indices_i64))

    # Reuse distance via Fenwick (BIT) tree – O(nnz log n)
    n_cols = int(A_csr.shape[1])
    kernel = _get_reuse_kernel()
    rd_arr, cold_misses = kernel(indices_i64, n_cols)

    rd_mean, rd_median, rd_max, rd_std = _array_stats(rd_arr)
    id_mean, id_median, id_max, id_std = _array_stats(idx_dists)

    return {
        "reuse_distance_mean": rd_mean,
        "reuse_distance_median": rd_median,
        "reuse_distance_max": rd_max,
        "reuse_distance_std": rd_std,
        "reuse_distance_count": len(rd_arr),
        "reuse_distance_cold_misses": int(cold_misses),
        "index_distance_mean": id_mean,
        "index_distance_median": id_median,
        "index_distance_max": id_max,
        "index_distance_std": id_std,
    }


def analyze_locality(A_scipy):
    """
    Calculate locality metrics.
    
    Args:
        A_scipy: Scipy sparse matrix
        
    Returns:
        Dictionary with locality metrics
    """
    # Convert to CSR for row-based locality analysis
    A_csr = A_scipy.tocsr()
    m, n = A_csr.shape
    
    # Row bandwidths (spread of nonzeros in each row)
    # For each row i, bw_i = max(j) - min(j)
    
    # We can compute this efficiently
    row_min = np.zeros(m)
    row_max = np.zeros(m)
    
    # This is a bit slow in pure python loop, let's try to vectorize if possible
    # Or just iterate since m might be large but loop overhead is small per row
    
    # Actually, for CSR, indices are stored in A_csr.indices
    # A_csr.indptr points to start/end of each row
    
    indices = A_csr.indices
    indptr = A_csr.indptr
    
    # Calculate row spreads
    # We need to handle empty rows
    
    # Vectorized approach for min/max per row is tricky without reducing
    # Let's use a loop with numba if available, or just optimized numpy
    
    # Calculate min and max column index for each row
    # Since indices are sorted in CSR (usually), min is first, max is last
    
    # Check if sorted
    if not A_csr.has_sorted_indices:
        A_csr.sort_indices()
        
    indices = A_csr.indices
    indptr = A_csr.indptr
    
    # Rows with nonzeros
    row_nnz = np.diff(indptr)
    non_empty_rows = row_nnz > 0
    
    # Get start and end indices for non-empty rows
    starts = indptr[:-1][non_empty_rows]
    ends = indptr[1:][non_empty_rows] - 1
    
    # Get min and max column indices
    min_cols = indices[starts]
    max_cols = indices[ends]
    
    row_spreads = max_cols - min_cols
    
    # Profile: sum of row spreads (envelope size)
    profile = np.sum(row_spreads)
    
    # Average row spread
    avg_row_spread = np.mean(row_spreads) if len(row_spreads) > 0 else 0
    
    # Row NNZ statistics (number of nonzeros per row)
    max_nnz_per_row = int(np.max(row_nnz)) if len(row_nnz) > 0 else 0
    avg_nnz_per_row = float(np.mean(row_nnz)) if len(row_nnz) > 0 else 0.0
    
    # Max and average row spread (only for non-empty rows)
    max_row_spread = int(np.max(row_spreads)) if len(row_spreads) > 0 else 0
    
    # ========== Vertical Locality Metrics ==========
    # Convert to CSC to analyze column-wise (vertical) patterns
    A_csc = A_csr.tocsc()
    csc_indptr = A_csc.indptr
    csc_indices = A_csc.indices  # Row indices for each column
    
    # Column NNZ statistics
    col_nnz = np.diff(csc_indptr)
    non_empty_cols = col_nnz > 0
    
    # Calculate vertical adjacency: count consecutive row indices per column
    # Vectorized approach to avoid looping over columns, which can be slow.
    # We find where the difference between consecutive row indices is 1.
    # This must not happen across column boundaries.
    
    # Find the difference between each element and the next in csc_indices
    vertical_diffs = np.diff(csc_indices)
    
    # Identify column boundaries in the diff array. A new column starts
    # at every index pointed to by csc_indptr.
    col_boundary_indices = csc_indptr[1:-1] - 1
    # Filter to valid diff indices: skip empty leading columns (>= 0)
    # and trailing columns (< len(vertical_diffs))
    valid = (col_boundary_indices >= 0) & (col_boundary_indices < len(vertical_diffs))
    col_boundary_indices = col_boundary_indices[valid]
    vertical_diffs[col_boundary_indices] = -1 # Invalidate diffs at boundaries
    
    # Vertical adjacency ratio: fraction of adjacent pairs that are consecutive
    consecutive_vertical_pairs = np.sum(vertical_diffs == 1)
    total_vertical_pairs = A_csc.nnz - int(np.sum(non_empty_cols)) # Total pairs is nnz - num_non_empty_cols
    vertical_adjacency_ratio = (consecutive_vertical_pairs / total_vertical_pairs if total_vertical_pairs > 0 else 0.0)
    
    # Column spread statistics (similar to row spread but for columns)
    col_starts = csc_indptr[:-1][non_empty_cols]
    col_ends = csc_indptr[1:][non_empty_cols] - 1
    min_rows = csc_indices[col_starts]
    max_rows = csc_indices[col_ends]
    col_spreads = max_rows - min_rows
    
    avg_col_spread = float(np.mean(col_spreads)) if len(col_spreads) > 0 else 0.0
    max_col_spread = int(np.max(col_spreads)) if len(col_spreads) > 0 else 0
    
    return {
        "profile": int(profile),
        "avg_row_spread": float(avg_row_spread),
        "max_row_spread": max_row_spread,
        "max_nnz_per_row": max_nnz_per_row,
        "avg_nnz_per_row": avg_nnz_per_row,
        "num_empty_rows": int(m - np.sum(non_empty_rows)),
        # Vertical locality metrics
        "consecutive_vertical_pairs": int(consecutive_vertical_pairs),
        "total_vertical_pairs": int(total_vertical_pairs),
        "vertical_adjacency_ratio": float(vertical_adjacency_ratio),
        "avg_col_spread": avg_col_spread,
        "max_col_spread": max_col_spread,
        "num_empty_cols": int(A_csc.shape[1] - np.sum(non_empty_cols))
    }
