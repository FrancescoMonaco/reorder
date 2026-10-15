# KAHYPAR/

Mt-KaHyPar (Multi-Threaded Karlsruhe Hypergraph Partitioner) partitioning-based
permutation generation.

## Algorithm

`kahypar_perm` treats the sparsity pattern of a **square** matrix as a symmetric
graph (each row/column index is a node; every stored nonzero `A[i][j]`, `i != j`,
is an undirected edge). The graph is partitioned into `k` blocks with Mt-KaHyPar,
and the permutation is produced by ordering vertices by their block assignment
(stable sort, so relative order inside a block is preserved). This yields a
block-diagonal layout that improves cache locality and enables block-based
parallel processing — the same strategy as the METIS/PaToH partition tools.

Reference:
- Gottesbüren, Heuer, Maß, Sanders, Schlag. *Scalable High-Quality Hypergraph
  Partitioning.* ACM Trans. Algorithms 20(1), 2024.
- Repo: https://github.com/kahypar/mt-kahypar (MIT)

## Prerequisites

- CMake >= 3.20, a C++20 compiler
- TBB, hwloc (pulled in automatically if you use the micromamba path below)

## Install

```bash
cd MtxPerm/KAHYPAR
./install.sh              # clone + build mt-kahypar + kahypar_perm
./install.sh --clean      # wipe local artifacts and rebuild from scratch
```

The mt-kahypar shared library is installed into the local prefix
`mt-kahypar-install/` (no root required) and the tool is built into `build/`.
Pin a specific release with `MK_TAG=<tag> ./install.sh`.

## Usage

```bash
./include/external/Reordering-for-blocks/MtxPerm/KAHYPAR/build/kahypar_perm \
    input.mtx output.perm --nparts 128
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--nparts N` | 128 | number of blocks/partitions |
| `--epsilon E` | 0.03 | allowed imbalance |
| `--preset P` | `default` | `default`, `quality`, `deterministic`, `large_k` |
| `--objective O` | `cut` | `cut`, `km1`, `soed` |
| `--threads T` | all cores | number of threads |
| `--seed S` | 42 | random seed |

## Output

Single-line, 1-based, space-separated `.perm` file (new-to-old mapping), the
same format used by every other tool in the pipeline. The tool also prints
`<Timer>[loading]` and `<Timer>[reordering]` lines consumed by
`scripts/parse_results.py`.

If the shared library cannot be located at runtime, source the preprocess
helper first. There are two variants — one for Bash, one for fish:

```bash
# Bash
source MtxPerm/KAHYPAR/kahypar_preprocess.sh
```

```fish
# Fish
source MtxPerm/KAHYPAR/kahypar_preprocess.fish
```
