# Mt-KaHyPar: Multilevel Graph Partitioning Reordering

## Overview

Mt-KaHyPar (Multi-Threaded Karlsruhe Hypergraph Partitioner) is a state-of-the-art
multilevel partitioner for (hyper)graphs. In this testbed it is used as a
**partitioning-based reordering**: the matrix's sparsity pattern is partitioned
into `k` blocks, and rows are reordered so that vertices sharing a block are
contiguous. This is the same family as METIS (`SB_metis`) and PaToH (`SB_patoh`)
and produces a block-diagonal layout.

Reference:
- Gottesbüren, Heuer, Maß, Sanders, Schlag. *Scalable High-Quality Hypergraph
  Partitioning.* ACM Transactions on Algorithms 20(1), 2024.
- Repository: https://github.com/kahypar/mt-kahypar (MIT license)

## Algorithm

1. **Graph construction.** For a square matrix, each index is a graph node and
   every stored nonzero `A[i][j]` with `i != j` becomes an undirected edge
   `{i, j}`. Self-loops (the diagonal) are dropped and edges are deduplicated.
2. **Multilevel partitioning.** Mt-KaHyPar contracts the graph, partitions the
   (small) coarse graph, then uncontracts while running localized refinement.
   It optimizes an objective (cut-net / connectivity / SOED) under a balance
   constraint of `1 + epsilon`.
3. **Permutation.** Each node's block id is read back with
   `mt_kahypar_get_partition`. Vertices are stable-sorted by block id, giving a
   new-to-old permutation `perm[new_position] = original_row_id`.

## Why It Works

- **Block-diagonal structure.** Vertices in the same block are placed together,
  so most nonzeros fall inside a few dense diagonal blocks.
- **Fewer inter-block edges** (small cut) means fewer long-range jumps during
  SpMM/SpMV, improving cache and TLB behavior.
- **High-quality partitions** at scale, thanks to flow-based refinement and the
  parallel multilevel scheme.

## Effect on Matrix Structure

- Increases block density at all block sizes.
- Typically reduces bandwidth relative to the original / random orderings.
- **Row-only permutation** (single-line `.perm`), but can also be applied
  symmetrically (`P A Pᵀ`) since it requires a square matrix.

## Implementation in MtxPerm

### Wrapper (`KAHYPAR/kahypar_perm.cpp`)

A self-contained C++ tool (no SparseBase dependency):

- Minimal Matrix Market reader (pattern/integer/real/complex, comment skipping).
- Builds the symmetric edge set, constructs the hypergraph in memory via the
  mt-kahypar **C API** (`mt_kahypar_create_hypergraph`), partitions
  (`mt_kahypar_partition`), reads blocks (`mt_kahypar_get_partition`), and writes
  the 1-based `.perm`.
- Prints `<Timer>[loading]` and `<Timer>[reordering]` (parsed by
  `parse_results.py`).

Key API calls (from `include/mtkahypar.h`):

```c
mt_kahypar_initialize(nthreads, /*numa=*/true);
mt_kahypar_context_t* ctx = mt_kahypar_context_from_preset(DEFAULT);
mt_kahypar_set_partitioning_parameters(ctx, k, epsilon, CUT);
mt_kahypar_hypergraph_t hg = mt_kahypar_create_hypergraph(
    ctx, num_nodes, num_edges, edge_indices, edges, edge_w, node_w, &err);
mt_kahypar_partitioned_hypergraph_t phg = mt_kahypar_partition(hg, ctx, &err);
mt_kahypar_get_partition(phg, block.data());
```

### Build (`KAHYPAR/CMakeLists.txt`, `KAHYPAR/install.sh`)

`install.sh` clones mt-kahypar, builds and installs the shared `mtkahypar`
library into a **local** prefix (`mt-kahypar-install/`, no root), then builds
`kahypar_perm` into `KAHYPAR/build/`. `kahypar_preprocess.sh` (Bash) and
`kahypar_preprocess.fish` export `LD_LIBRARY_PATH` for runtime.

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| k | 128 | `--nparts` | number of blocks |
| epsilon | 0.03 | `--epsilon` | allowed imbalance |
| preset | `default` | `--preset` | `default`, `quality`, `deterministic`, `large_k` |
| objective | `cut` | `--objective` | `cut`, `km1`, `soed` |
| seed | 42 | `--seed` | random seed |
| threads | all cores | `--threads` | thread count |

### Dependencies

- mt-kahypar (C++20), TBB, hwloc, Boost headers (all handled by `install.sh`).

### Permutation Type

- **Row** (`P*A`) or **Symmetric** (`P*A*Pᵀ`); requires a square matrix.

## References

- https://github.com/kahypar/mt-kahypar
- Gottesbüren et al., *Scalable High-Quality Hypergraph Partitioning*, ACM TALG 2024.
