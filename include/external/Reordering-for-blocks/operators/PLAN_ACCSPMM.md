# Plan: Add AccSpMM as a New SpMM Kernel

## Context

AccSpMM (PPoPP 2025) is a TF32 tensor-core SpMM kernel. The public repo (`github.com/YaoJianyu77/AccSpMM`) does **not** include the reordering algorithm described in the paper ("data-affinity-based reordering") — only the SpMM kernel itself. We integrate it as a kernel-only baseline, similar to how SMAT and ASpT are integrated.

**AccSpMM interface:** `./mma <matrix.mtx> <feature_dim>` — reads MTX, outputs timing to `result.csv` (appended) as `matrix_name,feature_dim,elapsed_time_us,throughput_GFLOPS`.

**Requirements:** CUDA >= 11.8, supported GPUs: A800, H100, RTX4090.

## Files to Create/Modify

### 1. Build & install script: `operators/install_accspmm.sh`
- Clone `https://github.com/YaoJianyu77/AccSpMM.git` into `operators/AccSpMM/`
- `mkdir build && cd build && cmake .. && make`
- Binary lands at `operators/AccSpMM/mma`

### 2. Utility module: `operators/accspmm_utils.py`
Pattern: follow `operators/smat_utils.py`
- `find_accspmm_binary()` — search `operators/AccSpMM/mma`, `operators/AccSpMM/build/mma`
- `run_accspmm_spmm(matrix_path, perm_path, perm_type, base_perm_path, base_perm_type, n_cols, ...)`
  1. `load_and_permute_matrix()` from `cusparse_utils` (reuse existing)
  2. Write permuted matrix to temp `.mtx` file via `scipy.io.mmwrite`
  3. Call `./mma <tmp.mtx> <n_cols>` via `subprocess.run`, capture stdout/stderr
  4. Parse `result.csv` (written by AccSpMM in CWD) — extract `elapsed_time_us`, convert to ms
  5. Also parse stdout for any timing output as fallback
  6. Clean up temp files and `result.csv`
  7. Return `{'loading_ms', 'write_ms', 'accspmm_kernel_ms'}`

**Key detail:** AccSpMM writes `result.csv` in CWD (appended). Run in a temp directory to isolate output, similar to ASpT pattern (`aspt_spmm.py:53`).

### 3. Wrapper script: `operators/accspmm_spmm.py`
Pattern: follow `operators/smat_spmm.py`
- Standard argparse: `matrix_path`, `--perm`, `--perm-type`, `--base-perm`, `--base-perm-type`, `--n-cols`
- Call `run_accspmm_spmm()`
- Print `<Timer>[loading]`, `<Timer>[operation]` via `print_timer()` from `cusparse_utils`

### 4. YAML experiment configs — add AccSpMM job entries

Add to each operations YAML (same pattern as `SMAT_SPMM_*`):

| File | Tag |
|---|---|
| `yamls/operations_row_reorder.yaml` | `ACCSPMM_SPMM_ROW` |
| `yamls/operations_symmetric_reorder.yaml` | `ACCSPMM_SPMM_SYMMETRIC` |
| `yamls/operations_asymmetric_reorder.yaml` | `ACCSPMM_SPMM_ASYMMETRIC` |
| `yamls/operations_no_reorder.yaml` | `ACCSPMM_SPMM_NOREORDER` |
| `yamls/operations_random_row.yaml` | `ACCSPMM_SPMM_RANDOM_ROW` |
| `yamls/operations_random_symmetric.yaml` | `ACCSPMM_SPMM_RANDOM_SYMMETRIC` |

Entry format (example for row):
```yaml
- tag: "ACCSPMM_SPMM_ROW"
  command: "python3 operators/accspmm_spmm.py {mtx} --perm perms/{perm}/$mtx_name.perm --perm-type ROW --n-cols {n_cols}"
```

### 5. No changes needed in `scripts/parse_results.py`
The timer format `<Timer>[operation] XXX.XXXXXX ms` is already auto-parsed. The tag-based categorization will pick up `ACCSPMM_SPMM_*` automatically as an operation job.

## Verification

1. **Build:** `bash operators/install_accspmm.sh` (on a compute node with CUDA >= 11.8)
2. **Smoke test:** `python3 operators/accspmm_spmm.py datasets/<any_small_matrix>.mtx --n-cols 32` — should print `<Timer>[loading]` and `<Timer>[operation]`
3. **With permutation:** `python3 operators/accspmm_spmm.py datasets/<matrix>.mtx --perm perms/SB_rcm/<matrix>.perm --perm-type ROW --n-cols 32`
4. **Full pipeline:** use `/test-all` on one matrix to verify end-to-end

## Open Items

- **Reordering:** The paper describes "data-affinity-based reordering" but the code is not in the public repo. Contact authors if needed.
- **GPU compatibility:** Only tested on A800, H100, RTX4090. May need CMake arch flags adjusted for other GPUs.
- **n_cols mapping:** AccSpMM calls it `feature_dim`. Verify it maps directly to our `--n-cols` parameter.
