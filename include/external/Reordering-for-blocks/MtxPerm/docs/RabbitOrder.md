# Rabbit Order

## Overview

Rabbit Order is a cache-aware graph reordering algorithm designed to improve spatial and temporal locality during graph traversals and sparse matrix operations. Proposed by Arai et al. (IPDPS 2016), it combines community detection with hierarchical clustering to produce orderings that maximize cache reuse.

## Algorithm

Rabbit Order operates in two phases:

### Phase 1: Community Detection via Label Propagation

1. **Label propagation**: Each vertex starts with its own label. In each iteration, vertices adopt the label most common among their neighbors (weighted by edge weight). This is repeated until convergence.
2. **Community identification**: Vertices with the same label form a community. This detects densely-connected groups in the graph.

### Phase 2: Hierarchical Ordering

1. **Community graph construction**: Create a coarsened graph where each community becomes a super-vertex, and edges between communities represent inter-community connections.
2. **Recursive ordering**: Recursively apply community detection and ordering on the community graph, building a hierarchical decomposition.
3. **Leaf ordering**: At the finest level, vertices within the same community receive consecutive indices. The hierarchical structure ensures that closely-related communities also receive nearby indices.

The result is a multi-scale ordering: vertices in the same community are adjacent, and related communities are nearby.

### Complexity

- **Time**: O(n + nnz) per label propagation iteration, typically converges in a few iterations. The hierarchical recursion adds a log factor.
- **Space**: O(n + nnz)

## Why It Works

Graph algorithms and SpMV/SpMM exhibit two types of locality:
- **Spatial locality**: Accessing consecutive memory locations (row data in CSR)
- **Temporal locality**: Reusing recently accessed data (x-vector elements in SpMV)

Rabbit Order's community-based ordering ensures:
- Vertices in the same community are stored consecutively, so accessing one vertex's neighbors likely hits cache from a previous vertex's access
- The hierarchical structure preserves locality at multiple scales, matching the cache hierarchy (L1/L2/L3)

## Effect on Matrix Structure

- **Block-diagonal tendency**: Communities form dense diagonal blocks
- **Hierarchical block structure**: Related communities form larger blocks at coarser scales
- **Good for power-law graphs**: Particularly effective on social networks and web graphs where community structure is pronounced
- **Preserves intra-community structure**: Vertices within a community maintain their relative ordering from the label propagation

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/rabbit_perm.cpp`)

Uses the SparseBase library's `RabbitReorder` class, which wraps the original Rabbit Order implementation:

```cpp
reorder::RabbitReorderParams params;  // Empty params (no tunable parameters)
int* perm = bases::ReorderBase::Reorder<reorder::RabbitReorder>(
    params, csr, {&cpu_context}, true);
```

- Requires a **square matrix**
- No tunable parameters
- SparseBase returns Old-to-New; inverted to New-to-Old
- Build dependencies: Boost, libnuma, detected at CMake time via `HAVE_RABBIT_REORDER`
- Output: 1-based permutation file

### Original Source (`SPARSEBASE/rabbit_build/`)

The Rabbit Order implementation is built from the original source by Arai et al. (NTT Corporation):
- `rabbit_order.hpp`: Core algorithm (label propagation, hierarchical ordering)
- Dependencies: Boost, libnuma, OpenMP
- Also used internally by AccOrder's community detection phase (see [AccOrder.md](AccOrder.md))

### Build Notes

Rabbit Order is built as part of SparseBase (`setup_sparsebase.sh --with-rabbit`):
1. `install_rabbit.sh` clones the [original repo](https://github.com/araij/rabbit_order) into `rabbit_build/` and copies the header to `rabbit_install/include/`
2. SparseBase is configured with `-DUSE_RABBIT_ORDER=ON -DRABBIT_ORDER_INC_DIR=rabbit_install/include`
3. The tools build (`setup_sparsebase.sh --build-only`) compiles `src/rabbit_perm.cpp` against SparseBase + rabbit_order.hpp

The build uses the system compiler (`/usr/bin/g++`) with modules `Boost/1.82.0-GCC-12.3.0` and `numactl/2.0.16-GCCcore-12.3.0` loaded (see `load_modules.sh`). The EasyBuild binutils may need to be stripped from PATH on some nodes to avoid "Illegal instruction" errors in the assembler (same issue as ACCORDER).

### Parameters

No tunable parameters. The algorithm auto-detects community structure.

### Permutation Type

- **Symmetric** (`P*A*P^T`): Applied to both rows and columns of square matrices.

## References

- Arai, J., Shiokawa, H., Yamamuro, T., Onizuka, M., and Iwamura, S. "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis." *IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2016.
