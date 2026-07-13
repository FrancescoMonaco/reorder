# Approximate Minimum Degree (AMD)

## Overview

Approximate Minimum Degree (AMD) is a fill-reducing ordering algorithm for sparse symmetric matrices. It produces permutations that minimize the fill-in (new nonzeros created) during Cholesky or LU factorization. AMD is widely used as preprocessing for direct sparse solvers and is considered one of the most effective fill-reduction heuristics.

## Algorithm

AMD is based on the **minimum degree** concept from elimination graphs:

1. **Elimination graph**: Maintain a graph where eliminating a vertex v means removing v and adding edges between all pairs of v's neighbors (clique formation).
2. **Minimum degree selection**: At each step, select the vertex with the smallest degree in the current elimination graph. This greedily minimizes the immediate fill-in.
3. **Approximate degree**: Instead of maintaining the exact elimination graph (which is expensive), AMD uses bounds and heuristics to **approximate** vertex degrees, dramatically reducing computation time.
4. **Aggressive absorption**: Optionally, vertices that would create identical fill patterns are merged ("absorbed"), further reducing the graph size.

Key approximations:
- **External degree**: Uses the number of distinct elements adjacent to a vertex rather than the true degree in the elimination graph
- **Mass elimination**: Eliminates multiple vertices of the same minimum degree simultaneously
- **Aggressive absorption**: Detects and merges indistinguishable vertices

### Complexity

- **Time**: O(n * nnz) in the worst case, but typically much faster due to aggressive pruning
- **Space**: O(n + nnz)

## Why It Works

During Cholesky/LU factorization, eliminating a vertex creates edges between all its uneliminated neighbors (fill-in). Choosing the vertex with the minimum degree at each step minimizes the number of new edges created, keeping the factor sparse. The "approximate" part trades a small loss in optimality for a large speedup.

## Effect on Matrix Structure

- **Minimizes fill-in**: Produces orderings where `L` and `U` factors have few additional nonzeros
- **Creates nested dissection-like structure**: Tends to separate clusters of tightly connected vertices
- **May not reduce bandwidth**: Optimizes for factorization fill, not for banded structure
- **Useful for**: Direct sparse solvers, preconditioning, and as a general-purpose reordering when factorization quality matters

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/amd_perm.cpp`)

Uses the SparseBase library's `AMDReorder` class, which wraps the SuiteSparse AMD library:

```cpp
reorder::AMDReorderParams params;
// Optional: params.dense = 10.0;       // dense row threshold
// Optional: params.aggressive = 1.0;   // aggressive absorption
int* perm = bases::ReorderBase::Reorder<reorder::AMDReorder>(
    params, csr, {&cpu_context}, true);
```

- Requires a **square matrix**
- SparseBase returns Old-to-New; inverted to New-to-Old
- Optional command-line parameters for `dense` and `aggressive` thresholds
- Build dependency: SuiteSparse AMD library (detected at CMake time via `HAVE_AMD_REORDER`)
- Output: 1-based permutation file

### Parameters

| Parameter | Default | CLI arg | Description |
|-----------|---------|---------|-------------|
| dense | AMD default (~10.0) | argv[3] | Threshold for treating rows as "dense" (bypassed during elimination) |
| aggressive | AMD default (~1.0) | argv[4] | Aggressiveness of absorption (higher = more aggressive merging) |

### Permutation Type

- **Symmetric** (`P*A*P^T`): AMD is defined for symmetric matrices and applies the same permutation to rows and columns.

## References

- Amestoy, P. R., Davis, T. A., and Duff, I. S. "An Approximate Minimum Degree Ordering Algorithm." *SIAM Journal on Matrix Analysis and Applications*, 17(4):886-905, 1996.
- Davis, T. A. *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
