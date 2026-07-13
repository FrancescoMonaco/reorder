#!/usr/bin/env python3
"""
Utilities for SMaT (Sparse Matrix Tensor Core) operators.
Handles matrix preparation and calling the smat CUDA binary.
"""

import os
import subprocess
import tempfile
import time
import numpy as np
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix
from cusparse_utils import load_and_permute_matrix, print_timer


# Path to smat binary - adjust if needed
SMAT_BINARY_PATHS = [
    os.path.join(os.path.dirname(__file__), "smat/src/cuda_hgemm/output/bin/hgemm"),
    os.path.join(os.path.dirname(__file__), "smat/build/bin/hgemm"),
]


def find_smat_binary():
    """Find the smat binary in expected locations."""
    for path in SMAT_BINARY_PATHS:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"SMaT binary not found. Searched locations:\n" +
        "\n".join(f"  - {p}" for p in SMAT_BINARY_PATHS) +
        "\n\nPlease run: bash operators/install_smat.sh"
    )


def setup_smat_environment():
    """
    Set up environment variables for smat binary execution.
    Adds locally installed gflags library path to LD_LIBRARY_PATH.
    """
    env = os.environ.copy()
    
    # Check for locally installed gflags
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gflags_lib_path = os.path.join(script_dir, "smat/gflags_install/lib")
    
    if os.path.exists(gflags_lib_path):
        # Add gflags lib path to LD_LIBRARY_PATH
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{gflags_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = gflags_lib_path
    
    return env


def run_smat_spmm(matrix_path, perm_path=None, perm_type='ROW',
                  base_perm_path=None, base_perm_type='SYMMETRIC',
                  n_cols=32, blocksize=8, n_iterations=5,
                  alpha=1.0, beta=0.0, dtype=np.float32):
    """
    Run SMaT SpMM operation on a matrix with optional permutation.
    
    SMaT only accepts MTX files, so we:
    1. Load and permute the matrix in Python (like cusparse)
    2. Write permuted matrix to temporary MTX file
    3. Call smat binary with the temporary file
    4. Parse timing output from smat's HLOG messages
    
    NOTE: SMaT binary uses CUDA events internally for accurate GPU timing.
    The timing is measured inside the binary with proper synchronization,
    not from the Python subprocess wrapper. We parse the profiling time
    from smat's output logs.
    
    Args:
        matrix_path: Path to original Matrix Market file
        perm_path: Optional path to permutation file
        perm_type: Type of permutation ('ROW', 'SYMMETRIC', or 'ASYMMETRIC')
        n_cols: Number of columns in dense matrix B for SpMM
        blocksize: Block size for tensor core optimization
        n_iterations: Number of timing iterations (smat will average internally)
        alpha: Alpha scalar (currently not configurable in smat binary)
        beta: Beta scalar (currently not configurable in smat binary)
        dtype: Data type (smat uses float16 internally for tensor cores)
    
    Returns:
        Dictionary with timing results: {
            'loading_ms': time to load and permute,
            'smat_total_ms': total smat execution time (wall clock),
            'smat_kernel_ms': kernel execution time from smat's internal timing (GPU events)
        }
    """
    # Step 1: Load and permute matrix using shared utilities
    t0 = time.perf_counter()
    A_cpu = load_and_permute_matrix(matrix_path, perm_path, perm_type, dtype,
                                    base_perm_path=base_perm_path, base_perm_type=base_perm_type)
    loading_ms = (time.perf_counter() - t0) * 1000
    
    # Step 2: Write permuted matrix to temporary MTX file
    # SMaT binary only accepts MTX format as input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mtx', delete=False) as tmp_file:
        tmp_mtx_path = tmp_file.name
    
    try:
        # Convert to CSR if not already
        if not isinstance(A_cpu, csr_matrix):
            A_cpu = A_cpu.tocsr()
        
        # Ensure indices are sorted before FP16 conversion
        # Permutation operations may leave indices unsorted, and scipy's
        # astype() requires canonical format (sorted indices, no duplicates)
        A_cpu.sort_indices()
        
        # SMaT uses FP16 for tensor cores, so convert to float16 before writing
        # Check for overflow BEFORE conversion to avoid scipy warnings
        max_val = np.abs(A_cpu.data).max()
        FP16_MAX = 65504.0  # Maximum representable value in float16
        
        if max_val > FP16_MAX:
            # Scale matrix to fit in FP16 range
            scale_factor = (FP16_MAX * 0.99) / max_val  # Use 0.99 for safety margin
            A_scaled = A_cpu * scale_factor
            A_fp16 = A_scaled.astype(np.float16)
            print(f"WARNING: Matrix values scaled by {scale_factor:.6e} to fit in FP16 range (max: {max_val:.6e})")
        else:
            # Safe to convert directly
            A_fp16 = A_cpu.astype(np.float16)
        
        # Write to MTX file
        t0 = time.perf_counter()
        mmwrite(tmp_mtx_path, A_fp16)
        write_ms = (time.perf_counter() - t0) * 1000
        
        # Step 3: Find and call smat binary
        smat_binary = find_smat_binary()
        
        # SMaT command line arguments based on github.com/spcl/smat
        # The binary appears to use gflags, common args:
        # --filename=<path> : input matrix file
        # --n_cols=<int> : number of columns in dense B matrix
        # There may be other flags - we'll capture stdout/stderr for timing info
        
        cmd = [
            smat_binary,
            f"--filename={tmp_mtx_path}",
        ]
        
        # Run smat binary
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=setup_smat_environment()  # Set up environment with gflags lib path
        )
        total_ms = (time.perf_counter() - t0) * 1000
        
        # Check for errors
        if result.returncode != 0:
            raise RuntimeError(
                f"SMaT binary failed with return code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        
        # Step 4: Parse output for timing information
        # SMaT uses HLOG macro and logs profiling time
        # Pattern: "exit, profiling time: XXX.XXX ms"
        # Also writes to results_smat.csv: "filename,time_ms"
        kernel_ms = None
        output = result.stdout + result.stderr
        
        # Parse SMaT output for timing
        # Look for: "profiling time: XXX.XXX ms" or similar patterns
        for line in output.split('\n'):
            # Match patterns like "profiling time: 123.456 ms"
            if 'profiling time' in line.lower() and 'ms' in line.lower():
                try:
                    # Extract number before 'ms'
                    parts = line.split('ms')[0].rsplit(':', 1)
                    if len(parts) == 2:
                        time_str = parts[1].strip()
                        kernel_ms = float(time_str)
                        break
                except (ValueError, IndexError):
                    pass
            
            # Fallback: look for just a float followed by ms
            if kernel_ms is None and 'ms' in line.lower():
                try:
                    # Try to find pattern like "XXX.XXX ms"
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*ms', line, re.IGNORECASE)
                    if match:
                        kernel_ms = float(match.group(1))
                except:
                    pass
        
        return {
            'loading_ms': loading_ms,
            'write_ms': write_ms,
            'smat_total_ms': total_ms,
            'smat_kernel_ms': kernel_ms,
            'smat_output': output
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_mtx_path):
            os.unlink(tmp_mtx_path)


def parse_smat_output(output_text):
    """
    Parse SMaT output to extract timing information.
    
    The exact format depends on smat's output, which may vary.
    This function attempts to extract key metrics.
    """
    timings = {}
    
    for line in output_text.split('\n'):
        line_lower = line.lower()
        
        # Look for various timing patterns
        if 'time' in line_lower:
            # Try to extract number + unit
            try:
                if 'ms' in line_lower:
                    parts = line_lower.split('ms')[0].split()
                    value = float(parts[-1])
                    
                    if 'kernel' in line_lower:
                        timings['kernel_ms'] = value
                    elif 'total' in line_lower:
                        timings['total_ms'] = value
                    elif 'transfer' in line_lower:
                        timings['transfer_ms'] = value
                        
                elif 's' in line_lower and 'ms' not in line_lower:
                    # Might be seconds
                    parts = line_lower.split('s')[0].split()
                    value = float(parts[-1]) * 1000  # Convert to ms
                    
                    if 'kernel' in line_lower:
                        timings['kernel_ms'] = value
                    elif 'total' in line_lower:
                        timings['total_ms'] = value
            except:
                continue
    
    return timings
