# Degree-Based Ordering

## Overview

Degree-based ordering is one of the simplest reordering heuristics. It sorts vertices by their degree (number of nonzeros per row) in ascending order, placing low-degree rows first and high-degree rows last. This creates a characteristic triangular nonzero pattern.

## Algorithm

1. **Compute degrees**: For each row i, compute `degree[i] = nnz_in_row(i)` (the number of nonzeros in that row).
2. **Sort by degree**: Create a permutation by sorting vertex indices by their degree values in ascending order. Uses a **stable sort** to maintain the relative order of vertices with equal degree.

### Complexity

- **Time**: O(n log n) for the sort, O(n) to compute degrees
- **Space**: O(n)

## Why It Works

Degree ordering groups vertices with similar connectivity patterns together. Rows with similar numbers of nonzeros tend to have more regular structure, which can improve:

- **Load balancing**: Grouping rows by degree allows thread schedulers to assign similar workloads
- **Memory access regularity**: Rows of similar length lead to more predictable memory access patterns
- **SIMD utilization**: More uniform row lengths within thread blocks reduce warp divergence on GPUs

## Effect on Matrix Structure

- **Triangular profile**: Low-degree rows (few nonzeros) at top, high-degree rows (many nonzeros) at bottom
- **Does not reduce bandwidth**: Column indices remain scattered; this is purely a row-based reordering
- **Improves regularity**: Creates more uniform row lengths within local neighborhoods of the permutation

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/degree_perm.cpp`)

Uses the SparseBase library's `DegreeReorder` class:

```cpp
reorder::DegreeReorderParams params(true);  // ascending=true
int* perm = bases::ReorderBase::Reorder<reorder::DegreeReorder>(
    params, csr, {&cpu_context}, true);
```

- `ascending=true`: Low-degree vertices first, high-degree last
- Works on **both square and rectangular** matrices (no square check)
- SparseBase returns Old-to-New; inverted to New-to-Old for the pipeline
- Output: 1-based permutation file

### Python GraphBLAS (`RANDOM/GB_degree_permutation.py`)

Uses SciPy to compute row degrees directly from CSR structure:

```python
degrees = np.diff(A.indptr)
perm = np.argsort(degrees, kind='stable') + 1  # 1-based
```

- Loads MTX via `scipy.io.mmread`, converts to CSR
- `np.diff(A.indptr)` computes the number of nonzeros per row from the CSR row pointer array
- Stable sort preserves original order among rows with equal degree

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| ascending | true | If true, low-degree first; if false, high-degree first |

### Permutation Type

- **Symmetric** (`P*A*P^T`) for square matrices: Reorders both rows and columns together
- **Row** (`P*A`) for rectangular matrices: Only reorders rows
