# Reordering for Blocks.

This repository contains the experimental suite for the paper Reordering and Blocking for High Performance Linear Algebra (currently under review). 
It provides a comprehensive research framework for evaluating *sparse matrix reordering algorithms*. It studies the structural effect of reordering on matrices, and how these structural changes impact linear algebra primitives on GPUs.

## Purpose

This repository systematically:
1. Downloads and manages sparse matrices from the SuiteSparse Matrix Collection
2. Generates permutations using various reordering algorithms
3. Analyzes matrix structural properties (bandwidth, block density, locality)
4. Benchmarks the performance of reordered matrices across multiple GPU kernels
5. Correlates reordering-induced structural changes with performance gains

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| [`scripts/`](scripts/README.md) | Result parsing, plotting, and job monitoring utilities |
| [`yamls/`](yamls/README.md) | Experiment configuration files for SbatchMan |
| [`operators/`](operators/) | SpMM kernel implementations and Python wrappers |
| [`MtxPerm/`](MtxPerm/) | Permutation generation tools |
| [`datasets/`](datasets/README.md) | Sparse matrices from SuiteSparse collection |
| [`results/`](results/README.md) | Aggregated CSV results from experiments |
| [`perms/`](perms/README.md) | Generated permutation files organized by algorithm |
| [`perms_random/`](perms_random/) | Permutations computed on random-scrambled matrices |
| [`plots/`](plots/) | Generated visualization plots |
| [`ccutils/`](ccutils/) | Header-only C++/CUDA utility library |
| [`distributed_mmio/`](distributed_mmio/) | Matrix Market I/O library |
| [`SbatchMan/`](SbatchMan/) | SLURM batch job management tool |
| [`reports/`](reports/) | SLURM job status reports |
| [`test/`](test/) | Test matrices, permutations, and fixtures |

## Quick Reference for Key Files

### Configuration
- **[yamls/perms.yaml](yamls/perms.yaml)** - All reordering algorithms and generation commands
- **[yamls/configs.yaml](yamls/configs.yaml)** - SLURM job configurations (GPU, CPU, memory)
- **[yamls/matrices.yaml](yamls/matrices.yaml)** - Matrix download criteria
- **[yamls/operations_*.yaml](yamls/)** - SpMM benchmark configurations
- **[yamls/analysis_*.yaml](yamls/)** - Matrix analysis configurations
- **[yamls/perms_random.yaml](yamls/perms_random.yaml)** - Permutation generation on random-scrambled matrices
- **[yamls/analysis_random_*.yaml](yamls/)** - Random-base analysis configurations

### Core Scripts
- **[scripts/parse_results.py](scripts/parse_results.py)** - Aggregates job outputs into CSVs
- **[scripts/plot.py](scripts/plot.py)** - Generates correlation and performance plots
- **[scripts/correlation_table.py](scripts/correlation_table.py)** - Generates LaTeX correlation tables (metrics vs. GFLOPS)
- **[scripts/job_check.py](scripts/job_check.py)** - Monitors SLURM job status

### Operator Wrappers
- **[operators/cusparse_csr_spmm.py](operators/cusparse_csr_spmm.py)** - cuSPARSE CSR SpMM
- **[operators/cusparse_bsr_spmm.py](operators/cusparse_bsr_spmm.py)** - cuSPARSE BSR SpMM
- **[operators/flashsparse_spmm.py](operators/flashsparse_spmm.py)** - FlashSparse (Tensor Core)
- **[operators/dtc_spmm.py](operators/dtc_spmm.py)** - DTC-SpMM
- **[operators/aspt_spmm.py](operators/aspt_spmm.py)** - ASpT (Adaptive Sparse Tiling)
- **[operators/smat_spmm.py](operators/smat_spmm.py)** - SMaT (Sparse Matrix Tensor Core)
- **[operators/config.py](operators/config.py)** - Shared configuration and utilities

### Permutation Tools
- **[MtxPerm/SPARSEBASE/](MtxPerm/SPARSEBASE/)** - C++ reordering via SparseBase library
- **[MtxPerm/RANDOM/](MtxPerm/RANDOM/)** - GraphBLAS random permutations
- **[MtxPerm/SPARTA/](MtxPerm/SPARTA/)** - SPARTA blocking algorithm
- **[MtxPerm/GROOT/](MtxPerm/GROOT/)** - Groot graph-centric row reordering (EuroSys'25)
- **[MtxPerm/ANALYSIS/](MtxPerm/ANALYSIS/)** - Matrix structure analysis
- **[MtxPerm/pre_permute.py](MtxPerm/pre_permute.py)** - Pre-permute a matrix and write new `.mtx` (used by random-base experiments)

## Reordering Algorithms

Defined in [yamls/perms.yaml](yamls/perms.yaml):

| Algorithm | Description |
|-----------|--------|-------------|
| `identity` | - | No reordering (baseline) |
| `random1D` | Random row permutation |
| `random2D` | Random row + column permutation |
| `SB_rcm` | Reverse Cuthill-McKee (bandwidth reduction) |
| `SB_degree` | Degree-based ordering |
| `SB_gray` | Gray code reordering |
| `SB_amd` | Approximate Minimum Degree |
| `SB_metis` | METIS graph partitioning |
| `SB_rabbit` | Rabbit reordering |
| `SB_slashburn` | SlashBurn algorithm |
| `SPARTA_reorder` | Sparse blocking for tensor cores |
| `GROOT_reorder` | k-NN + MST + DFS row reordering for tensor cores (EuroSys'25) |

## Permutation Types

Three modes for applying permutations:

- **ROW**: `P * A` - Permute rows only
- **SYMMETRIC**: `P * A * P^T` - Same permutation for rows and columns (square matrices)
- **ASYMMETRIC**: `P_row * A * P_col^T` - Different permutations for rows and columns

## SpMM Kernels Benchmarked

| Kernel | Description |
|--------|-------------|
| `CUSPARSE_SPMM_CSR` | cuSPARSE CSR format |
| `CUSPARSE_SPMM_BSR` | cuSPARSE Block Sparse Row |
| `FLASHSPARSE_SPMM` | FlashSparse Tensor Core (PPoPP'25) |
| `DTC_SPMM` | Dynamic Tensor Core SpMM |
| `ASPT_SPMM` | Adaptive Sparse Tiling |
| `SMAT_SPMM` | Sparse Matrix Tensor Core |

## Data Formats

| Format | Description |
|--------|-------------|
| `.mtx` | Matrix Market format (standard sparse matrix) |
| `.perm` | Permutation file: 1-based indices, space-separated, 1-2 lines |
| `.csv` | Result files |
| `.yaml` | SbatchMan experiment configurations |

## Permutation File Format

```
# Single line = symmetric or row-only permutation
3 1 4 2 5

# Two lines = asymmetric permutation (row perm, then col perm)
3 1 4 2 5
2 4 1 5 3
```

Indices are **1-based**.

### Analysis YAMLs

| YAML | Description |
|------|-------------|
| [`analysis_random_no_reorder.yaml`](yamls/analysis_random_no_reorder.yaml) | Baseline: random-symmetric-permuted matrix (no further reordering) |
| [`analysis_random_row.yaml`](yamls/analysis_random_row.yaml) | Random-sym base + ROW reordering on top |
| [`analysis_random_symmetric.yaml`](yamls/analysis_random_symmetric.yaml) | Random-sym base + SYMMETRIC reordering on top |

### Output

Permutations are stored in `perms_random/<algorithm>/<matrix_name>.perm`.

## Typical Workflow

1. **Download matrices**: Configure [yamls/matrices.yaml](yamls/matrices.yaml)
2. **Generate permutations**: Configure [yamls/perms.yaml](yamls/perms.yaml), run reorderings with `sbatchman launch --file yamls/perm_*.yaml`
3. **Run analysis**: Use `sbatchman launch --file yamls/analysis_*.yaml` to compute matrix metrics
4. **Run benchmarks**: Use `sbatchman launch --file yamls/operations_*.yaml` to benchmark SpMM kernels
5. **Parse results**: `python scripts/parse_results.py`
6. **Generate plots**: `python scripts/plot.py --one-per-family`

## Dependencies

### Python
- PyTorch (with CUDA), CuPy
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- python-graphblas
- ssgetpy (SuiteSparse download)

### C++/CUDA
- CUDA Toolkit (≥11.8)
- SparseBase library
- CMake (≥3.12)
- OpenMP
- Optional: METIS, PaToH

## Environment

Designed for SLURM cluster execution. Job configurations in [yamls/configs.yaml](yamls/configs.yaml).
