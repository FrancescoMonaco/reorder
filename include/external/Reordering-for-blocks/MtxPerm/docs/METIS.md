# METIS Graph Partitioning

## Overview

METIS is a multilevel graph partitioning algorithm that divides a graph into k roughly equal-sized partitions while minimizing edge cuts. In this project, METIS is used as a **reordering** tool: after partitioning the graph, vertices are renumbered by partition ID so that vertices in the same partition receive consecutive indices.

## Algorithm

METIS uses a three-phase **multilevel** approach:

### 1. Coarsening Phase
The graph is repeatedly contracted by merging pairs of adjacent vertices (heavy-edge matching), creating a hierarchy of smaller graphs. Each coarsening step roughly halves the number of vertices.

### 2. Initial Partitioning Phase
The coarsest graph (typically a few hundred vertices) is partitioned using direct methods (e.g., greedy graph growing, spectral bisection).

### 3. Uncoarsening/Refinement Phase
The partition is projected back through the hierarchy. At each level, a Kernighan-Lin / Fiduccia-Mattheyses (KL/FM) refinement algorithm improves the partition by moving vertices between parts to reduce the cut.

### Partition-to-Permutation Conversion

After partitioning, vertices are sorted by their partition ID using a **stable sort**, maintaining relative order within each partition. This groups vertices in the same partition together in the final ordering.

### Complexity

- **Time**: O(n + nnz) for each coarsening/uncoarsening step, typically O(n log k) total
- **Space**: O(n + nnz)

## Why It Works

Graph partitioning groups densely connected vertices together while placing sparse connections at partition boundaries. When used as a matrix reordering:

- Vertices in the same partition access similar memory regions during SpMV/SpMM
- Cross-partition edges (boundary entries) are minimized
- Within each partition, the original relative order is preserved (stable sort)

## Effect on Matrix Structure

- **Block-diagonal tendency**: Creates a roughly block-diagonal matrix with small off-diagonal blocks
- **Improves cache locality**: Vertices in the same partition share many neighbors, so x-vector accesses are localized
- **Good for parallel processing**: Each partition can be processed independently with minimal communication
- **Number of partitions matters**: More partitions = finer grouping but less bandwidth reduction; fewer = coarser blocks

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/metis_part_perm.cpp`)

Uses METIS k-way partitioning via SparseBase's `MetisPartition` class:

```cpp
partition::MetisPartitionParams params;
params.num_partitions = nparts;   // default: 128
params.objtype = objtype;          // METIS_OBJTYPE_CUT or METIS_OBJTYPE_VOL

partition::MetisPartition<int, int, float> partitioner;
auto partition_result = partitioner.Partition(csr, &params, {&cpu_context}, true);

// Sort vertices by partition assignment
std::stable_sort(vertex_partition.begin(), vertex_partition.end(),
    [](auto& a, auto& b) { return a.second < b.second; });
```

- Requires a **square matrix**
- Uses k-way partitioning (not nested dissection)
- Stable sort preserves vertex order within each partition
- Build dependency: METIS library (detected at CMake time)
- Output: 1-based permutation file

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| nparts | 128 | `--nparts N` | Number of partitions (higher = finer grouping) |
| objtype | cut | `--objtype TYPE` | Optimization objective: `cut` (minimize edge cuts) or `vol` (minimize communication volume) |

### Permutation Type

- **Symmetric** (`P*A*P^T`): Same permutation applied to both rows and columns.

## References

- Karypis, G. and Kumar, V. "A fast and high quality multilevel scheme for partitioning irregular graphs." *SIAM Journal on Scientific Computing*, 20(1):359-392, 1998.
- Karypis, G. and Kumar, V. "METIS: A Software Package for Partitioning Unstructured Graphs, Partitioning Meshes, and Computing Fill-Reducing Orderings of Sparse Matrices." University of Minnesota, 1998.
