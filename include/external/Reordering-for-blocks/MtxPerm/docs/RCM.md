# Reverse Cuthill-McKee (RCM)

## Overview

Reverse Cuthill-McKee (RCM) is a bandwidth-reduction algorithm for sparse matrices. It reorders the rows and columns of a matrix to move nonzero elements closer to the diagonal, reducing the **bandwidth** (the maximum distance of any nonzero from the diagonal). Originally proposed by Cuthill and McKee (1969) and later reversed by George (1971), RCM is one of the oldest and most widely-used reordering heuristics.

## Algorithm

The algorithm operates on the adjacency graph of the sparse matrix:

1. **Select a starting vertex**: Choose a peripheral vertex (typically one with low degree via a pseudo-peripheral node finder).
2. **BFS traversal**: Perform breadth-first search from the starting vertex. At each level, visit neighbors in order of ascending degree (fewest connections first).
3. **Reverse**: Reverse the resulting ordering. This reversal was shown by George to produce smaller envelopes/profiles than the original Cuthill-McKee ordering.

The result is a permutation where connected vertices are close together in the new ordering, concentrating nonzeros near the diagonal in a banded pattern.

### Complexity

- **Time**: O(n + nnz) for the BFS traversal, where n is the number of vertices and nnz is the number of nonzero entries
- **Space**: O(n)

## Why It Works

RCM exploits the fact that BFS naturally groups vertices by proximity in the graph. By ordering vertices in BFS layers and preferring low-degree neighbors, the algorithm minimizes the "spread" of connections across the ordering. The reversal step further tightens the profile because the last BFS levels tend to have fewer vertices, creating a better taper.

## Effect on Matrix Structure

- **Reduces bandwidth**: Nonzeros cluster near the diagonal
- **Improves cache locality**: Sequential access patterns during SpMV/SpMM
- **Banded structure**: Creates approximately banded matrices, beneficial for block-based formats (BSR) and fill-in reduction in direct solvers

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/rcm_perm.cpp`)

Uses the SparseBase library's `RCMReorder` class:

```cpp
reorder::RCMReorderParams params;
int* perm = bases::ReorderBase::Reorder<reorder::RCMReorder>(
    params, csr, {&cpu_context}, true);
```

- Loads the matrix via `reorder_utils::load_matrix()` (MTX -> CSR)
- Requires a **square matrix**
- SparseBase returns an **Old-to-New** permutation; the wrapper inverts it to **New-to-Old** for the pipeline
- Output: 1-based permutation file
- No tunable parameters

### SciPy Python (`RCM/rcm_perm.py`)

Uses `scipy.sparse.csgraph.reverse_cuthill_mckee`:

```python
perm = reverse_cuthill_mckee(A, symmetric_mode=True)
```

- Loads MTX via `scipy.io.mmread`, converts to CSC
- `symmetric_mode=True` treats the matrix as symmetric
- Returns a **New-to-Old** permutation (0-based), converted to 1-based for output
- No tunable parameters

### Permutation Type

- **Symmetric** (`P*A*P^T`): Same permutation applied to rows and columns. This is the natural application since RCM is defined on undirected graphs.

## References

- Cuthill, E. and McKee, J. "Reducing the bandwidth of sparse symmetric matrices." *Proceedings of the 1969 24th National Conference*, ACM, 1969.
- George, J. A. "Computer implementation of the finite element method." PhD thesis, Stanford University, 1971.
