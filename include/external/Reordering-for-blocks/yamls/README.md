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

Algorithms defined: `identity`, `random1D`, `random2D`, `SB_rcm`, `SB_degree`, `SB_gray`, `SB_amd`, `SB_metis`, `SB_rabbit`, `SB_slashburn`, `SB_patoh`, `SPARTA_reorder`, `GROOT_reorder`, `TCA_reorder`, `CLUB_reorder`

### Permutation file shape vs. perm_type

| File lines | Written by | Evaluate with |
|------------|-----------|----------------|
| 1 line (row perm) | every algorithm including `CLUB_reorder` | `--perm-type ROW` (rows) or `SYMMETRIC` (rows+cols, square matrices) |

`CLUB_reorder` runs the **hybrid** pipeline (`reorder.py --technique hybrid`:
IDF-filtered MinHash micro-clustering + weighted bipartite macro-RCM + SIMD
Jaccard slab packing) and writes a **single-line row perm**. The `ASYMMETRIC`
loader accepts single-line files as row-only (columns untouched), so `CLUB_reorder`
can be dropped into the ASYMMETRIC tables for comparison, evaluated with exactly
the same code path as the other row-only algorithms.

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
# 1. generate permutations (row-only hybrid CLUB baseline)
sbatchman launch --file yamls/perms.yaml

# 2. structural analysis
sbatchman launch --file yamls/analysis_row_reorder.yaml        # ROW view
sbatchman launch --file yamls/analysis_asymmetric_reorder.yaml # row-only view

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

To run `CLUB_reorder` (hybrid) in a shell yourself:

```bash
python include/external/Reordering-for-blocks/MtxPerm/CLUB/reorder.py \
    /data/matrices/datasets/3elt_dual/3elt_dual.mtx /tmp/3elt_dual.perm \
    --technique hybrid --w 4 --w-block 32 --lookahead 64 --hub-frac 0.05
# (also available: --technique classic [reorder2], --technique multilevel)
```

## Choosing a matrix suite (`--ai` / `--extended` / `--standard`)

Every experiment YAML declares **all** of its matrix suites in a single `mtx`
`map`, keyed by a `suite` variable. This requires **SbatchMan >= 2.0**. The suite is chosen at launch time with the
`scripts/launch.py` wrapper:

`--ai`, `--extended`, `--standard` define different sets of matrices:

The suites map to these list files:

| Suite     | List file                          | Matrices |
|-----------|------------------------------------|----------|
| `ai`      | `ai_matrices.txt`                  | 1045     |
| `extended`| `matrices_extended.txt`            | 389      |
| `standard`| `matrices_list_filtered.txt`       | 52       |

`scripts/launch.py --<suite>` is sugar for sbatchman's `--variable` filter, so
the same selection can be done directly:

```bash
sbatchman launch -f include/external/Reordering-for-blocks/yamls/perms_random_ai.yaml \
    -v suite=ai --dry-run


## YAML Structure Pattern

```yaml
variables:
  suite: [ai, extended, standard]        # Matrix suite selector
  mtx:                                   # One list file per suite
    map:
      ai: /data/matrices/datasets/ai_matrices.txt
      extended: /data/matrices/datasets/matrices_extended.txt
      standard: /data/matrices/datasets/matrices_list_filtered.txt
  perm: [SB_rcm, SB_degree, ...]         # Permutation algorithms
  n_cols: [32, 256, 1024]                # Dense matrix widths
  block_size: [32]                       # BSR block size

preprocess: |
  micromamba activate reorder_env
  mtx_name=$(basename {mtx[suite]} .mtx)

jobs:
  - config: "gpu"
    config_jobs:
      - tag: "KERNEL_NAME"
        command: "python3 operators/kernel.py {mtx[suite]} --args"
```

## Variable Expansion

- `{mtx[suite]}`: Expands to each matrix path from the suite's list file
- `{perm}`: Expands to each permutation algorithm
- `{n_cols}`: Expands to each dense matrix width
- `$mtx_name`: Shell variable set in preprocess
