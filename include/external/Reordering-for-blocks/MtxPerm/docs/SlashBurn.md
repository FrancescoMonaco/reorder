# SlashBurn

## Overview

SlashBurn is a reordering algorithm designed for graph compression and community detection. Proposed by Lim, Kang, and Faloutsos (ICDM 2009), it identifies high-degree "hub" vertices and separates them from the rest of the graph, creating a structure amenable to block-based compression and processing.

## Algorithm

SlashBurn iteratively peels hub vertices from the graph:

1. **Identify hubs**: Find the top-k vertices with the highest degree in the current graph.
2. **Remove hubs**: Remove these k hub vertices and all their edges.
3. **Find connected components**: After removing the hubs, the remaining graph typically breaks into multiple disconnected components.
4. **Assign positions**:
   - Hub vertices are placed at the **front** of the ordering (top-left of the matrix)
   - The **largest connected component** (GCC) is retained for the next iteration
   - Smaller components ("spokes") are placed at the **end** of the ordering (bottom-right)
5. **Recurse**: Repeat on the largest connected component until it becomes small enough.

The result is a "wing" structure: a narrow band of hub-related entries at the top and left, and a block-diagonal structure of disconnected components in the remaining part.

### Greedy Mode

The greedy variant (default) uses a more aggressive hub selection strategy, choosing vertices that maximize graph disconnection rather than purely by degree.

### Complexity

- **Time**: O(k * iterations * nnz) where iterations = O(n/k). With k = n/1000, this is roughly O(nnz * 1000).
- **Space**: O(n + nnz)

## Why It Works

Real-world graphs (especially power-law / scale-free graphs) have a small number of hub vertices connected to many other vertices. Removing these hubs fragments the graph into many small components. By separating hubs from spokes:

- **Compression**: The remaining blocks can be compressed efficiently because they have simple structure
- **Block structure**: The spoke components form a natural block-diagonal pattern
- **Hub separation**: Dense hub rows are concentrated together, which can improve memory access patterns

## Effect on Matrix Structure

- **Wing/arrow structure**: A narrow band of hub-related entries at top-left, block-diagonal elsewhere
- **Good for power-law graphs**: Very effective on social networks, web graphs, and other scale-free graphs
- **Less effective on regular graphs**: Meshes and other regular structures lack the hub-spoke pattern that SlashBurn exploits
- **May increase bandwidth**: Hub vertices connected to distant parts of the graph can create wide-spread entries

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/slashburn_perm.cpp`)

Uses the SparseBase library's `SlashburnReorder` class:

```cpp
// Default k = 0.1% of n (following BEAR reference implementation)
if (k <= 0) {
    k = std::max(1, n / 1000);
}
reorder::SlashburnReorder<int, int, float> reorderer(k, greedy, false);
auto perm = reorderer.GetReorder(csr, {&cpu_context}, true);
```

- Requires a **square matrix**
- Uses `GetReorder()` instead of `Reorder()` (different API pattern than other SparseBase reorderers)
- SparseBase returns Old-to-New; inverted to New-to-Old
- Includes validation of permutation values (bounds checking)
- Build dependency: detected at CMake time via `HAVE_SLASHBURN_REORDER`
- Output: 1-based permutation file

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| k | n/1000 (0.1% of vertices) | `--k K` | Number of hubs to remove per iteration |
| greedy | true | `--greedy 0\|1` | Use greedy hub selection strategy |

### Permutation Type

- **Symmetric** (`P*A*P^T`): Applied to both rows and columns of square matrices.

## References

- Lim, Y., Kang, U., and Faloutsos, C. "SlashBurn: Graph Compression and Mining beyond Caveman Communities." *IEEE ICDM*, 2009.
- Shin, K., Lim, S., Lee, J., and Kang, U. "BEAR: Block Elimination Approach for Random Walk with Restart on Large Graphs." *Proceedings of the ACM SIGMOD*, 2015.
