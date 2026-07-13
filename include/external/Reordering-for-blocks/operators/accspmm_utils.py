#!/usr/bin/env python3
"""
Utilities for AccSpMM (PPoPP 2025 TF32 Tensor Core SpMM).
Handles matrix preparation and calling the AccSpMM binary.
"""

import os
import sys
import csv
import subprocess
import tempfile
import time
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cusparse_utils import load_and_permute_matrix


# Path to AccSpMM binary - adjust if needed
ACCSPMM_BINARY_PATHS = [
    os.path.join(os.path.dirname(__file__), "AccSpMM", "mma"),
    os.path.join(os.path.dirname(__file__), "AccSpMM", "build", "mma"),
]


def find_accspmm_binary():
    """Find the AccSpMM binary in expected locations."""
    for path in ACCSPMM_BINARY_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"AccSpMM binary not found. Searched locations:\n" +
        "\n".join(f"  - {p}" for p in ACCSPMM_BINARY_PATHS) +
        "\n\nPlease run: bash operators/install_accspmm.sh"
    )


def run_accspmm_spmm(matrix_path, perm_path=None, perm_type='ROW',
                      base_perm_path=None, base_perm_type='SYMMETRIC',
                      n_cols=32, dtype=np.float32):
    """
    Run AccSpMM SpMM operation on a matrix with optional permutation.

    AccSpMM only accepts MTX files, so we:
    1. Load and permute the matrix in Python
    2. Write permuted matrix to temporary MTX file
    3. Call ./mma <tmp.mtx> <n_cols> in a temp directory
    4. Parse result.csv output for timing

    AccSpMM interface: ./mma <matrix.mtx> <feature_dim>
    Output: appends to result.csv in CWD with columns:
        matrix_name,feature_dim,elapsed_time_us,throughput_GFLOPS

    Args:
        matrix_path: Path to original Matrix Market file
        perm_path: Optional path to permutation file
        perm_type: Type of permutation ('ROW', 'SYMMETRIC', or 'ASYMMETRIC')
        base_perm_path: Optional path to base permutation file (applied first)
        base_perm_type: Type of base permutation
        n_cols: Number of columns in dense matrix B (feature_dim)
        dtype: Data type for matrix loading

    Returns:
        Dictionary with timing results: {
            'loading_ms': time to load and permute,
            'write_ms': time to write temp MTX file,
            'accspmm_kernel_ms': kernel execution time from AccSpMM output,
            'accspmm_output': raw stdout+stderr output
        }
    """
    # Step 1: Load and permute matrix
    t0 = time.perf_counter()
    A_cpu = load_and_permute_matrix(matrix_path, perm_path, perm_type, dtype,
                                    base_perm_path=base_perm_path,
                                    base_perm_type=base_perm_type)
    loading_ms = (time.perf_counter() - t0) * 1000

    # Step 2: Write permuted matrix to temporary MTX file
    if not isinstance(A_cpu, csr_matrix):
        A_cpu = A_cpu.tocsr()
    A_cpu.sort_indices()

    with tempfile.NamedTemporaryFile(suffix='.mtx', delete=False) as tmp_file:
        tmp_mtx_path = tmp_file.name

    try:
        t0 = time.perf_counter()
        mmwrite(tmp_mtx_path, A_cpu)
        write_ms = (time.perf_counter() - t0) * 1000

        # Step 3: Find binary and run in temp directory to isolate result.csv
        accspmm_binary = find_accspmm_binary()

        # Run in a temp directory so result.csv doesn't pollute CWD
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [accspmm_binary, tmp_mtx_path, str(n_cols)]

            result = subprocess.run(
                cmd,
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"AccSpMM binary failed with return code {result.returncode}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )

            output = result.stdout + result.stderr

            # Step 4: Parse result.csv written by AccSpMM
            kernel_ms = None
            result_csv_path = os.path.join(tmp_dir, "result.csv")

            if os.path.exists(result_csv_path):
                with open(result_csv_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # Format: matrix_name,feature_dim,elapsed_time_ms,throughput_GFLOPS
                        # Note: the third column is in milliseconds (verified empirically
                        # against GFLOPS: GFLOPS = 2*nnz*ncols / (time_s * 1e9))
                        if len(row) >= 3:
                            try:
                                kernel_ms = float(row[2])
                            except (ValueError, IndexError):
                                pass

            # Fallback: try to parse timing from stdout
            if kernel_ms is None:
                import re
                for line in output.split('\n'):
                    match = re.search(r'(\d+\.?\d*)\s*us', line)
                    if match:
                        kernel_ms = float(match.group(1)) / 1000.0
                        break
                    match = re.search(r'(\d+\.?\d*)\s*ms', line)
                    if match:
                        kernel_ms = float(match.group(1))
                        break

        return {
            'loading_ms': loading_ms,
            'write_ms': write_ms,
            'accspmm_kernel_ms': kernel_ms,
            'accspmm_output': output,
        }

    finally:
        if os.path.exists(tmp_mtx_path):
            os.unlink(tmp_mtx_path)
