# AccOrder: Acc-SpMM Reordering

## Overview

AccOrder is the reordering algorithm from **Acc-SpMM**, which combines **Rabbit Order** community detection with a **greedy common-neighbor traversal** to produce cache-friendly orderings. The algorithm first identifies communities using Rabbit Order's label propagation, then traverses the graph greedily by following the neighbor with the most common neighbors to the current vertex.

## Algorithm

AccOrder operates in two phases:

### Phase 1: Community Detection (Rabbit Order)

Uses the Rabbit Order algorithm (see [RabbitOrder.md](RabbitOrder.md)) to detect communities:
1. **Label propagation**: Vertices iteratively adopt the most common label among their neighbors
2. **Community identification**: Groups of vertices with the same label form communities
3. **Hierarchical clustering**: Communities are recursively merged into a hierarchy

### Phase 2: Greedy Common-Neighbor Traversal

After community detection, `re2order_vertex()` performs a greedy traversal:
1. **Start from a vertex** (typically one with high degree)
2. **At each step**: Among unvisited neighbors, choose the one with the **most common neighbors** with the current vertex
3. **If stuck** (no unvisited neighbors): Jump to the unvisited vertex with the most common neighbors to any visited vertex
4. **Track neighbor sets**: Maintain `all_vertex_nbrs` map for efficient common-neighbor counting

This greedy approach ensures that structurally similar vertices (sharing many neighbors) are placed consecutively.

### Phase 3: Permutation Inversion

The traversal produces a `new2old` mapping (deque). `handle_vertex_remap()` converts this to an `ori2new` mapping suitable for the pipeline.

### Complexity

- **Time**: O(n * d^2) in the worst case for common-neighbor computation, where d is the maximum degree
- **Space**: O(n * d) for neighbor set storage

## Why It Works

AccOrder targets two aspects of SpMM performance:

1. **Community-based grouping** (Phase 1): Rabbit Order identifies densely connected clusters. Vertices in the same community are likely to share many column indices, leading to better x-vector cache reuse.
2. **Common-neighbor ordering** (Phase 2): The greedy traversal further refines the ordering within communities by placing vertices that share the most neighbors adjacent to each other. This maximizes overlap in memory access patterns between consecutive rows.

## Effect on Matrix Structure

- **Dense diagonal blocks**: Communities form blocks with high internal density
- **Smooth transitions**: The common-neighbor traversal ensures gradual changes between consecutive rows
- **Good for irregular matrices**: The community detection handles matrices without obvious geometric structure
- **Combined benefits**: Gets the hierarchical benefits of Rabbit Order plus the fine-grained locality of common-neighbor ordering

## Implementation in MtxPerm

### C++ Wrapper (`ACCORDER/accorder_perm.cpp`)

Reads MTX, writes edge list, calls the AccOrder library:

```cpp
// Read MTX and write edge list to temp file
auto coo = mtx_io::read_mtx(mtx_path);
coo.write_edge_list(edge_file);

// Run AccOrder reordering
auto adj = read_graph(edge_file);
std::deque<rabbit_order::vint> new2old = re2order_vertex(std::move(adj));
std::vector<rabbit_order::vint> ori2new = handle_vertex_remap(std::move(new2old));

// Write .perm file (1-based)
for (vint i = 0; i < n; ++i)
    ofs << (ori2new[i] + 1);
```

### Build

```bash
bash MtxPerm/ACCORDER/install_accorder.sh
```

Downloads AccOrder sources from Zenodo, extracts the `order/` directory, and builds with CMake.

### Source Code

The AccOrder library consists of:
- `my_order.hpp`: Core `re2order_vertex()` function and common-neighbor traversal
- `edge_list.hpp`: Graph loading from edge list files
- `rabbit_order.hpp`: Rabbit Order implementation (community detection)

All from the Acc-SpMM Zenodo repository (zenodo.org/records/14214504).

### Runtime Preprocess

Jobs require module loading via `accorder_preprocess.sh`:
```bash
source MtxPerm/ACCORDER/accorder_preprocess.sh
```
This loads `numactl/2.0.16-GCCcore-12.3.0` (Boost is header-only, only needed at build time).

### Build Notes

The install script:
- Uses system compiler (`/usr/bin/g++`) to avoid EasyBuild binutils "Illegal instruction" errors
- Strips EasyBuild binutils from PATH
- Defines `BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE` and `BOOST_ATOMIC_DETAIL_NO_HAS_UNIQUE_OBJECT_REPRESENTATIONS` to work around Boost 1.82 stricter `boost::atomic` requirements with rabbit_order's non-trivially-copyable `atom` struct

### Dependencies

- OpenMP (parallelism)
- Boost 1.82 (module `Boost/1.82.0-GCC-12.3.0`, used by Rabbit Order internals)
- libnuma (module `numactl/2.0.16-GCCcore-12.3.0`, NUMA-aware memory allocation)
- C++17

### Parameters

No tunable parameters. The algorithm uses default settings for both Rabbit Order community detection and the greedy traversal.

### Permutation Type

- **Symmetric** (`P*A*P^T`): Applied to both rows and columns. The algorithm treats the matrix as an undirected graph.

## References

- Acc-SpMM: Accelerating Sparse Matrix Multiplication with Accurate Reordering. Zenodo: 10.5281/zenodo.14214504.
- Arai, J., et al. "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis." *IPDPS*, 2016. (underlying community detection)
