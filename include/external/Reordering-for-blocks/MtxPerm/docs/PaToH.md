# PaToH Hypergraph Partitioning

## Overview

PaToH (Partitioning Tool for Hypergraphs) is a multilevel hypergraph partitioning tool. Unlike graph partitioning (METIS), PaToH operates on **hypergraphs** where a single hyperedge can connect more than two vertices. When applied to sparse matrices, each row is a vertex and each column is a hyperedge connecting all rows that have a nonzero in that column. This model directly captures the communication pattern of SpMV/SpMM.

## Algorithm

PaToH follows the same three-phase multilevel paradigm as METIS, but on hypergraphs:

### 1. Coarsening Phase
Vertices are merged using hypergraph-aware matching (e.g., heavy connectivity matching). The hypergraph is contracted by merging matched vertices and their associated hyperedges.

### 2. Initial Partitioning Phase
The coarsest hypergraph is partitioned using greedy bisection or other direct methods.

### 3. Uncoarsening/Refinement Phase
The partition is projected back through the hierarchy with FM-style (Fiduccia-Mattheyses) refinement optimizing either the **cut** or **connectivity** objective.

### Optimization Objectives

- **Cut**: Minimize the total number of cut hyperedges (hyperedges spanning multiple partitions)
- **Connectivity (lambda-1)**: Minimize `sum(lambda_e - 1)` over all hyperedges e, where `lambda_e` is the number of partitions that hyperedge e spans. This directly corresponds to the **communication volume** in parallel SpMV.

### Partition-to-Permutation Conversion

Vertices are sorted by `(partition_id, vertex_id)`, grouping all vertices in the same partition together while maintaining vertex order within partitions.

## Why It Works

The hypergraph model is a more accurate representation of SpMV communication than the graph model:

- A column j in the sparse matrix corresponds to a hyperedge connecting all rows that access `x[j]`
- Minimizing connectivity minimizes the number of partitions that need `x[j]`, directly reducing cache misses and communication
- Graph partitioning (METIS) can only model pairwise relationships, while hypergraph partitioning captures the multi-way sharing pattern

## Effect on Matrix Structure

- **Near-block-diagonal**: Similar to METIS but with better optimization of column sharing
- **Optimizes x-vector reuse**: The connectivity objective directly minimizes redundant x-vector accesses across partitions
- **Better for irregular matrices**: Hypergraph model handles highly irregular sparsity patterns better than graph-based methods

## Implementation in MtxPerm

### SparseBase C++ (`SPARSEBASE/src/patoh_perm.cpp`)

Uses PaToH via SparseBase's `PatohPartition` class:

```cpp
partition::PatohPartitionParams params;
params.num_partitions = num_parts;  // default: 128
params.objective = partition::patoh::CON;  // connectivity objective
params.param_init = partition::patoh::QUALITY;

partition::PatohPartition<int, int, float> partitioner;
int* partition = partitioner.Partition(csr, &params, {&cpu_context}, true);

// Sort by (partition_id, vertex_id)
std::sort(part_vertex.begin(), part_vertex.end());
```

- Works on any matrix (no square requirement in theory, though input is loaded via CSR)
- Default objective: **connectivity** (optimizes for SpMV/SpMM x-vector reuse)
- Quality mode for more accurate partitioning
- Build dependency: PaToH library (detected at CMake time via `HAVE_PATOH_PARTITION`)
- Output: 1-based permutation file

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| num_partitions | 128 | `--nparts N` | Number of partitions |
| objective | connectivity | `--objective OBJ` | `connectivity` (minimize communication volume) or `cut` (minimize cut hyperedges) |

### Permutation Type

- **Symmetric** (`P*A*P^T`): Same permutation applied to both rows and columns.

## References

- Catalyurek, U. V. and Aykanat, C. "PaToH: Partitioning Tool for Hypergraphs." Technical Report BU-CE-9915, Bilkent University, 1999.
- Catalyurek, U. V. and Aykanat, C. "Hypergraph-partitioning-based decomposition for parallel sparse-matrix vector multiplication." *IEEE TPDS*, 10(7):673-693, 1999.
