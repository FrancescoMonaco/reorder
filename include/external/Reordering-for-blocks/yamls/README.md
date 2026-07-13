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

Algorithms defined: `identity`, `random1D`, `random2D`, `SB_rcm`, `SB_degree`, `SB_gray`, `SB_amd`, `SB_metis`, `SB_rabbit`, `SB_slashburn`, `SPARTA_reorder`

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
