# Groot: Graph-Centric Row Reordering

## Overview

Groot is a row reordering algorithm specifically designed to improve block density for **Tensor Core** SpMM operations. Proposed by Chen et al. (EuroSys 2025), it builds a k-nearest-neighbor (k-NN) graph based on row similarity, extracts a Minimum Spanning Tree (MST), and traverses it via DFS to produce a row ordering where similar rows are adjacent.

## Algorithm

Groot operates in four stages:

### 1. k-NN Graph Construction

For each row of the sparse matrix, find the k most similar rows using **Jaccard similarity** (default) or Hamming distance:

- **Jaccard similarity**: `J(A, B) = |A intersect B| / |A union B|`, where A and B are the sets of nonzero column indices for two rows
- Uses **PyNNDescent** for approximate k-NN search in O(N log N) time with Numba JIT
- Default k = 16 nearest neighbors per row
- Custom sparse metric implementations (`sparse.py`) optimize Jaccard/Hamming computation for sparse vectors
- k-NN results can be cached to disk for reuse

### 2. MST Extraction (optional, default: enabled)

Extract a Minimum Spanning Tree from the k-NN graph:
- Convert similarity to distance: `distance = 1 - similarity`
- Symmetrize the k-NN graph (take minimum distance for each edge)
- Compute MST using `scipy.sparse.csgraph.minimum_spanning_tree`
- The MST preserves the strongest (most similar) connections while creating a tree structure

### 3. Graph Traversal

Traverse the MST (or full k-NN graph if MST is disabled) using **DFS** (default) or BFS:
- **Starting node**: Node with maximum degree in the graph (default), random, or first
- **Neighbor ordering**: At each node, visit neighbors sorted by descending similarity (most similar first)
- **Disconnected components**: After the initial traversal, iterate over unvisited nodes and start new traversals
- The traversal order becomes the new row permutation

### 4. Permutation Output

The DFS/BFS traversal order gives a **New-to-Old** mapping: `reorder_id[new_position] = old_row_id`. This is converted to a 1-based `.perm` file.

### Complexity

- **Time**: O(N log N) for k-NN via PyNNDescent, O(N * k) for MST construction, O(N) for traversal
- **Space**: O(N * k) for the k-NN graph

## Why It Works

Tensor Core units (TCU) on NVIDIA GPUs operate on small dense matrix tiles (e.g., 16x16). For sparse matrices stored in block formats, the Tensor Core utilization depends on how many nonzeros fall within each tile. Groot maximizes this by:

1. **Grouping similar rows**: Rows with similar nonzero patterns are placed adjacent, filling more positions within block tiles
2. **MST as skeleton**: The MST ensures the traversal follows the strongest similarity paths, avoiding "jumps" to dissimilar rows
3. **DFS depth-first**: DFS naturally creates long chains of similar rows, better than BFS which can mix similarity levels within a level

## Effect on Matrix Structure

- **Increases block density**: More nonzeros per block at all block sizes (4x4, 8x8, 16x16, etc.)
- **Improves Tensor Core utilization**: Higher fraction of tile elements are nonzero
- **Row-only reordering**: Only permutes rows; column order is unchanged (though symmetric application is possible)
- **Particularly effective for**: Matrices with groups of rows sharing similar column patterns

## Implementation in MtxPerm

### Wrapper (`GROOT/reorder.py`)

Thin wrapper that handles format conversion:

```python
# Convert MTX to NPZ format
A = mmread(matrix_path).tocoo()
np.savez(npz_path, src_li=A.row, dst_li=A.col, num_nodes=A.shape[0])

# Invoke groot.py
cmd = [sys.executable, groot_py, "--dataset", dataset_name,
       "--input_dir", tmpdir, "--output_dir", tmpdir, "--knn", str(knn)]
subprocess.run(cmd)

# Read result and convert to .perm
reorder_id = np.load(reorder_npz)["reorder_id"]
perm_1based = reorder_id + 1
```

### Core Algorithm (`GROOT/Groot-EuroSys25/groot.py`)

The full implementation uses PyNNDescent, SciPy MST, and custom traversal:

```python
# k-NN construction
index = NNDescent(scipy_csr, metric=sparse_jaccard,
                  n_neighbors=knn+1, random_state=2022)

# MST extraction (optional)
mst = minimum_spanning_tree(adj_matrix_csr)

# DFS traversal
def dfs(node, graph, visited, reorder):
    visited.add(node)
    reorder.append(node)
    neighbors = sorted(graph[node], key=lambda x: x[1], reverse=True)
    for neighbor_id, sim in neighbors:
        if neighbor_id not in visited:
            dfs(neighbor_id, graph, visited, reorder)
```

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| knn | 16 | `--knn K` | Number of nearest neighbors per row |
| similarity_metric | jaccard | `--similarity_metric` | `jaccard` or `hamming` |
| use_mst | true | `--no_mst` | Whether to extract MST from k-NN graph |
| traversal | dfs | `--traversal` | `dfs` or `bfs` |
| start_node | max_degree | `--start_node` | `max_degree`, `random`, or `first` |

### Dependencies

- PyNNDescent (approximate k-NN)
- SciPy (MST, sparse matrices)
- Numba (JIT compilation for sparse metrics)
- Conda environment: `GROOT`

### Permutation Type

- **Row** (`P*A`) or **Symmetric** (`P*A*P^T`): Designed for row-only reordering but can be applied symmetrically.

## References

- Chen, Y., Xie, J., Teng, S., Zeng, W., and Yu, J. X. "Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplications on Tensor Cores." *Proceedings of the Twentieth European Conference on Computer Systems (EuroSys)*, pp. 803-817, 2025.
