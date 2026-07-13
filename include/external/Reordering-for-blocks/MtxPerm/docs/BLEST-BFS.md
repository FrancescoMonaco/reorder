# BLEST-BFS: Blazingly Efficient BFS Reordering

## Overview

BLEST (Blazingly Efficient BFS using Tensor Cores) is a GPU-accelerated BFS framework by Elbek and Kaya (2025). In the context of this project, BLEST's internal reordering is used as a matrix permutation technique. BLEST reorders the graph using a combination of **Gorder** (a cache-aware ordering) and **Jaccard-based windowed refinement**, then represents the graph in its custom **BVSS** (Bit-Vector Sparse Storage) format optimized for Tensor Core BFS.

## Algorithm

BLEST's reordering combines two techniques:

### 1. Gorder (Base Ordering)

Gorder (Wei et al., SIGMOD 2016) is a cache-aware ordering that maximizes the number of "cache hits" during graph traversal:

1. **Sliding window**: Maintain a window of size W over the vertex ordering
2. **Score function**: For each unordered vertex, compute a score based on how many of its neighbors are within the current window
3. **Greedy selection**: At each step, select the vertex with the highest score (most neighbors in the window)
4. **Priority queue**: Use a unit heap data structure for efficient score maintenance

The Gorder implementation is in `Gorder/Graph.cpp`.

### 2. Jaccard-Based Windowed Refinement

When `--jaccard 1` (default), BLEST applies an additional refinement:

1. **Jaccard similarity**: For each pair of vertices within a window, compute Jaccard similarity of their neighbor sets
2. **Local swaps**: Reorder vertices within windows to maximize similarity between adjacent vertices
3. **Window size**: Configurable (default: 65536)

### BVSS Construction

After reordering, BLEST constructs a Bit-Vector Sparse Storage (BVSS) format:
- The adjacency matrix is divided into 8-bit slices
- Each slice is stored as packed bit vectors
- This format enables Tensor Core operations for parallel BFS frontier expansion

### Complexity

- **Time**: O(n * W) for Gorder where W is the window size, O(n * k * W) for Jaccard refinement
- **Space**: O(n + nnz)

## Why It Works

BLEST's reordering targets BFS performance specifically:
- **Gorder** ensures that vertices likely to be explored together are stored nearby, improving cache locality for BFS frontier operations
- **Jaccard refinement** further groups structurally similar vertices (similar neighbor sets), which means BFS expansion from one vertex benefits from cache lines loaded for previous vertices
- The combined ordering is particularly effective for the BVSS bit-vector format, where similar neighbor sets lead to more efficient Tensor Core operations

## Effect on Matrix Structure

- **Cache-optimized locality**: Adjacent vertices in the ordering share many neighbors
- **Good for traversal operations**: Optimized for BFS but beneficial for general graph operations
- **Particularly effective for social networks**: The `isSocialNetwork()` heuristic adjusts padding strategy based on degree distribution

## Implementation in MtxPerm

### Custom Driver (`BLEST/blest/blest_driver.cu`)

A custom CUDA driver replaces BLEST's default `main()` to support the reordering pipeline:

```cpp
CSC* csc = new CSC(inputFile, true, false);  // Load MTX as undirected

unsigned* inversePermutation = nullptr;
if (noReorder) {
    // Identity permutation (for benchmarking pre-reordered matrices)
    for (unsigned i = 0; i < N; ++i)
        inversePermutation[i] = i;
} else {
    inversePermutation = csc->reorder(sliceSize);
}

BVSS* bvss = new BVSS(sliceSize, noMasks, devnull);
bvss->constructFromCSCMatrix(csc);
```

Key globals:
- `JACKARD_ON`: Enable/disable Jaccard refinement
- `WINDOW_SIZE`: Window size for Jaccard
- `FULL_PADDING`: Padding strategy (auto-detected based on network type)

### Build

```bash
bash MtxPerm/BLEST/install.sh
```

Requires CUDA >= 13.0, g++ >= 12.3, cmake >= 3.18. GPU compute capability >= 80 (Ampere+).

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| jaccard | 1 | `--jaccard 0\|1` | Enable Jaccard-based refinement |
| window | 65536 | `--window W` | Window size for Jaccard refinement |
| no-reorder | false | `--no-reorder` | Use identity permutation (skip reordering) |
| n-sources | 64 | `--n-sources N` | Number of random BFS sources for benchmarking |
| seed | 42 | `--seed S` | Random seed for source selection |

### Usage Note

BLEST is primarily a **BFS benchmark** framework that includes its own reordering. In this project, we use BLEST both as:
1. An **operation** (BFS benchmark) with external pre-reordering
2. A **reorderer** via its internal Gorder+Jaccard pipeline

### Permutation Type

- **Symmetric** (`P*A*P^T`): The reordering treats the graph as undirected.

## References

- Elbek, D. and Kaya, K. "BLEST: Blazingly Efficient BFS using Tensor Cores." *arXiv:2512.21967*, 2025.
- Wei, H., Yu, J. X., Lu, C., and Lin, X. "Speedup Graph Processing by Graph Ordering." *ACM SIGMOD*, 2016. (Gorder)
