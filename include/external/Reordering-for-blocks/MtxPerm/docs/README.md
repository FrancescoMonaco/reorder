# Reordering Techniques Documentation

Detailed documentation for each sparse matrix reordering algorithm implemented in MtxPerm.

## Traditional / Graph-Theoretic Reorderings

| Algorithm | File | Category | Requires Square | Key Optimization Target |
|-----------|------|----------|:-:|------------------------|
| [RCM](RCM.md) | `SPARSEBASE/src/rcm_perm.cpp`, `RCM/rcm_perm.py` | Bandwidth reduction | Yes | Bandwidth, cache locality |
| [AMD](AMD.md) | `SPARSEBASE/src/amd_perm.cpp` | Fill reduction | Yes | Factorization fill-in |
| [Degree](Degree.md) | `SPARSEBASE/src/degree_perm.cpp`, `RANDOM/GB_degree_permutation.py` | Sorting | No | Row length regularity |
| [Gray Code](GrayCode.md) | `SPARSEBASE/src/gray_perm.cpp` | Pattern similarity | Yes | Vertical adjacency, block density |

## Partitioning-Based Reorderings

| Algorithm | File | Category | Requires Square | Key Optimization Target |
|-----------|------|----------|:-:|------------------------|
| [METIS](METIS.md) | `SPARSEBASE/src/metis_part_perm.cpp` | Graph partitioning | Yes | Block-diagonal structure |
| [PaToH](PaToH.md) | `SPARSEBASE/src/patoh_perm.cpp` | Hypergraph partitioning | No | Communication volume / x-vector reuse |

## Community / Cache-Aware Reorderings

| Algorithm | File | Category | Requires Square | Key Optimization Target |
|-----------|------|----------|:-:|------------------------|
| [Rabbit Order](RabbitOrder.md) | `SPARSEBASE/src/rabbit_perm.cpp` | Community detection | Yes | Cache locality (multi-level) |
| [SlashBurn](SlashBurn.md) | `SPARSEBASE/src/slashburn_perm.cpp` | Hub separation | Yes | Graph compression, block structure |
| [AccOrder](AccOrder.md) | `ACCORDER/accorder_perm.cpp` | Community + greedy traversal | Yes | Cache locality + neighbor overlap |

## Tensor Core-Optimized Reorderings

| Algorithm | File | Category | Requires Square | Key Optimization Target |
|-----------|------|----------|:-:|------------------------|
| [Groot](GROOT.md) | `GROOT/reorder.py` | k-NN + MST + DFS | No | Block density for Tensor Cores |
| [SPARTA](SPARTA.md) | `SPARTA/reorder.py` | Block-aware grouping | No | Block density for specific tile sizes |
| [DTC-LSH](DTC-LSH.md) | `DTC-LSH/reorder.py` | MinHash LSH clustering | No | Tensor Core tile fill |
| [BLEST-BFS](BLEST-BFS.md) | `BLEST/blest/blest_driver.cu` | Gorder + Jaccard | Yes | BFS cache locality + Tensor Cores |

## Baselines

| Algorithm | File | Category | Requires Square | Key Optimization Target |
|-----------|------|----------|:-:|------------------------|
| [Random](Random.md) | `RANDOM/GB_random_permutation_{1d,2d}.py` | Baseline | No | None (control) |

## Common Implementation Patterns

All reordering tools in this project follow a common pattern:

1. **Input**: Matrix Market (`.mtx`) sparse matrix file
2. **Output**: Permutation file (`.perm`) with 1-based, space-separated indices
3. **Permutation direction**: New-to-Old mapping (`perm[new_position] = old_index`)
4. **SparseBase tools**: Return Old-to-New from the library; the wrapper inverts to New-to-Old
5. **Timing**: Each tool prints `<Timer>[loading]` and `<Timer>[reordering]` in milliseconds
