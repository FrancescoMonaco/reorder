# yamls/

Experiment configuration files for SbatchMan job management.

## Configuration Files

### configs.yaml

SLURM job configurations defining resource requirements:
- `gpu`: GPU partition settings (A100, time limits, memory)
- `cpu`: CPU-only partition settings

### matrices.yaml

Matrix download criteria for SuiteSparse collection:
- NNZ (non-zero) count ranges
- Matrix properties (square, symmetric, etc.)
- Download destination paths

### perms.yaml

**Key file** - Defines all reordering algorithms and their generation commands.

Structure:
```yaml
perms:
  algorithm_name:
    command: "path/to/tool --args {mtx} {output}"
    type: symmetric|asymmetric|row
```

Algorithms defined: `identity`, `random1D`, `random2D`, `SB_rcm`, `SB_degree`, `SB_gray`, `SB_amd`, `SB_metis`, `SB_rabbit`, `SB_slashburn`, `SB_patoh`, `SPARTA_reorder`, `GROOT_reorder`, `TCA_reorder`, `CLUB_reorder`, `CLUB_jaccard`

### Permutation file shape vs. perm_type

| File lines | Written by | Evaluate with |
|------------|-----------|----------------|
| 1 line (row perm) | every algorithm including `CLUB_reorder` | `--perm-type ROW` (rows) or `SYMMETRIC` (rows+cols, square matrices) |
| 2 lines (row perm / col perm) | `CLUB_jaccard` (`--technique jaccard`, two-sided) | `--perm-type ASYMMETRIC` — both lines applied: `A' = P_row * A * P_col^T` |

The `ASYMMETRIC` loader also accepts **single-line** files and treats them as
row-only (columns untouched). This is intentional: it lets row-only algorithms
be dropped into the ASYMMETRIC tables for an apples-to-apples comparison with
two-sided `CLUB_jaccard` runs, evaluated by exactly the same code path.
Conversely, reading a two-line file with `--perm-type ROW` applies line 1 only,
which gives the row-only view of the same two-sided run.

## Experiment YAMLs

### Analysis Experiments

Compute matrix structural metrics after applying permutations:

| File | Permutation Type |
|------|------------------|
| `analysis_no_reorder.yaml` | Identity (baseline) |
| `analysis_row_reorder.yaml` | Row-only permutation |
| `analysis_symmetric_reorder.yaml` | Symmetric permutation |
| `analysis_asymmetric_reorder.yaml` | Asymmetric permutation |

### Operation Experiments

Benchmark SpMM kernels with permuted matrices:

| File | Permutation Type |
|------|------------------|
| `operations_no_reorder.yaml` | Identity (baseline) |
| `operations_row_reorder.yaml` | Row-only permutation |
| `operations_symmetric_reorder.yaml` | Symmetric permutation |
| `operations_asymmetric_reorder.yaml` | Asymmetric permutation |

## Running the CLUB pipelines

```bash
# 1. generate permutations (row-only stripe baseline + two-sided jaccard)
sbatchman launch --file yamls/perms.yaml

# 2. structural analysis
sbatchman launch --file yamls/analysis_row_reorder.yaml        # ROW view
sbatchman launch --file yamls/analysis_asymmetric_reorder.yaml # row+col view

# 3. SpMM benchmarks
sbatchman launch --file yamls/operations_row_reorder.yaml
sbatchman launch --file yamls/operations_asymmetric_reorder.yaml

# 4. aggregate + plots (--asymmetric selects the new pipeline)
python scripts/parse_results.py
python scripts/plot.py --asymmetric
python scripts/plot.py --row
```

`parse_results.py` labels rows by tag, so the `*_ASYMMETRIC` tags land in
`results_analysis.csv` / `results_operations.csv` with
`perm_type == 'ASYMMETRIC'` and coexist with the `ROW` / `SYMMETRIC` rows.

To run `CLUB_jaccard` in a shell yourself:

```bash
python include/external/Reordering-for-blocks/MtxPerm/CLUB/reorder.py \
    /data/matrices/datasets/3elt_dual/3elt_dual.mtx /tmp/3elt_dual.perm \
    --technique jaccard --bs 16 --tau 0.5 --max-iters 4
# add --row-only for a single-line (row-only) permutation
```

## YAML Structure Pattern

```yaml
variables:
  mtx: 'datasets/matrices_list_mtx.txt'  # List of matrix paths
  perm: [SB_rcm, SB_degree, ...]         # Permutation algorithms
  n_cols: [32, 256, 1024]                # Dense matrix widths
  block_size: [32]                       # BSR block size

preprocess: |
  source ~/.venv/bin/activate
  mtx_name=$(basename {mtx} .mtx)
  module load CUDA/

jobs:
  - config: "gpu"
    config_jobs:
      - tag: "KERNEL_NAME"
        command: "python3 operators/kernel.py {mtx} --args"
```

## Variable Expansion

- `{mtx}`: Expands to each matrix path from file list
- `{perm}`: Expands to each permutation algorithm
- `{n_cols}`: Expands to each dense matrix width
- `$mtx_name`: Shell variable set in preprocess
