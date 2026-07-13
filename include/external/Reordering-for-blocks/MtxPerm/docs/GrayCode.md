# Gray Code Reordering

## Overview

Gray code reordering permutes the rows of a sparse matrix so that consecutive rows in the new ordering have similar sparsity patterns, minimizing the Hamming distance between adjacent rows. Named after Frank Gray's reflected binary code (1953), this technique was adapted for sparse matrix reordering to improve data locality during SpMV operations.

## Algorithm

The algorithm works by representing each row as a binary vector (nonzero = 1, zero = 0) and then ordering these vectors according to a Gray code-like sequence:

1. **Bitmap construction**: For each row, create a bitmap of the nonzero column positions. The bitmap resolution is configurable (16, 32, or 64 bits).
2. **Separate sparse/dense rows**: Rows with nnz <= `nnz_threshold` are considered "sparse" and those above are "dense". They are reordered separately.
3. **Gray code sorting**: Within each group, rows are sorted so that consecutive rows differ in as few bit positions as possible (minimizing Hamming distance).
4. **Group-based refinement**: For sparse rows, an additional grouping step with configurable `sparse_density_group_size` provides finer-grained ordering.

### Complexity

- **Time**: O(n log n) for sorting rows by their bitmap values
- **Space**: O(n) for bitmaps

## Why It Works

In SpMV/SpMM, consecutive rows that access similar column indices lead to better cache reuse of the input vector x. If row i accesses columns {2, 5, 8} and row i+1 accesses columns {2, 5, 9}, the data for columns 2 and 5 is already in cache. Gray code ordering maximizes this overlap by grouping structurally similar rows together.

## Effect on Matrix Structure

- **Improves vertical adjacency**: Nonzeros in consecutive rows tend to appear in the same columns
- **Enhances block density**: For block formats (BSR, BCSR), similar rows create denser blocks
- **Does not reduce bandwidth per se**: Focuses on column pattern similarity rather than diagonal proximity
- **Particularly effective for** matrices with regular or repeating substructures (e.g., FEM meshes)

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/gray_perm.cpp`)

Uses the SparseBase library's `GrayReorder` class:

```cpp
int avg_nnz_per_row = (n > 0) ? (nnz / n) : 1;
int group_size = std::max(1, avg_nnz_per_row / 16);
reorder::GrayReorderParams params(
    reorder::BitMapSize::BitSize32,  // 32-bit resolution
    32,                               // nnz threshold for sparse vs dense
    group_size                        // group size based on matrix density
);
int* perm = bases::ReorderBase::Reorder<reorder::GrayReorder>(
    params, csr, {&cpu_context}, true);
```

- Requires a **square matrix**
- Bitmap resolution set to 32 bits by default
- Sparse/dense threshold: 32 nnz per row
- Group size: automatically computed as `avg_nnz_per_row / 16`
- SparseBase returns Old-to-New; inverted to New-to-Old
- Includes validation of permutation values (bounds checking)
- Output: 1-based permutation file

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| BitMapSize | BitSize32 | Resolution of bitmap encoding (16, 32, or 64 bits) |
| nnz_threshold | 32 | Rows with nnz <= this are "sparse" |
| sparse_density_group_size | avg_nnz/16 | Group size for refinement within sparse section |

### Permutation Type

- **Symmetric** (`P*A*P^T`): Applied to both rows and columns of square matrices

## References

- Gray, F. "Pulse code communication." U.S. Patent 2,632,058, 1953.
- Pinar, A. and Heath, M. T. "Improving performance of sparse matrix-vector multiplication." *Proceedings of Supercomputing*, 1999.
