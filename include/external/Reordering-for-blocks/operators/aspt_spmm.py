#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import scipy.io
import scipy.sparse
import time

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cusparse_utils import load_and_permute_matrix, print_timer
from config import SPMM_N_COLS_DEFAULT, PERM_TYPE_DEFAULT

def main():
    parser = argparse.ArgumentParser(description='ASpT SpMM Benchmark')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')
    parser.add_argument('--n-iterations', type=int, default=100, help='Number of iterations (ignored by ASpT binary, but kept for compatibility)')
    
    args = parser.parse_args()

    # 1. Load and permute matrix
    t0 = time.perf_counter()
    try:
        A = load_and_permute_matrix(args.matrix_path, args.perm, args.perm_type,
                                    base_perm_path=args.base_perm, base_perm_type=args.base_perm_type)
    except Exception as e:
        print(f"Error loading matrix: {e}", file=sys.stderr)
        sys.exit(1)
    loading_ms = (time.perf_counter() - t0) * 1000

    # 2. Prepare environment
    script_dir = Path(__file__).parent.absolute()
    
    # 3. Write matrix to temp file
    # ASpT expects a Matrix Market file
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp_mtx:
        scipy.io.mmwrite(tmp_mtx, A)
        tmp_mtx_path = tmp_mtx.name
    transfer_ms = (time.perf_counter() - t0) * 1000

    # 4. Run ASpT
    # We need to run in a temp dir to capture the output file cleanly
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Find binary
        # It should be in operators/ASpT/ASpT_SpMM_GPU/sspmm_32
        binary = script_dir / "ASpT" / "ASpT_SpMM_GPU" / "sspmm_32"
        
        if not binary.exists():
             # Try looking in other places just in case
             if (script_dir / "sspmm_32").exists():
                 binary = script_dir / "sspmm_32"
             else:
                 print(f"Error: ASpT binary not found at {binary}. Please run install_aspt.sh", file=sys.stderr)
                 os.remove(tmp_mtx_path)
                 sys.exit(1)

        # ASpT arguments: <matrix_file> <n_cols>
        cmd = [str(binary), tmp_mtx_path, str(args.n_cols)]
        
        try:
            # Run in tmp_dir so output file is created there
            # ASpT writes to SpMM_GPU_SP.out in CWD
            subprocess.run(cmd, cwd=tmp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Read output file
            out_file = Path(tmp_dir) / "SpMM_GPU_SP.out"
            if out_file.exists():
                with open(out_file, 'r') as f:
                    content = f.read().strip()
                    # Format: GFLOPS,ErrorRate,
                    parts = content.split(',')
                    if len(parts) >= 1:
                        try:
                            gflops = float(parts[0])
                            
                            # Convert GFLOPS to time_ms
                            # GFLOPS = (2 * nnz * n_cols) / (time_ms * 1e6)
                            # time_ms = (2 * nnz * n_cols) / (GFLOPS * 1e6)
                            nnz = A.nnz
                            n_cols = args.n_cols
                            if gflops > 0:
                                time_ms = (2 * nnz * n_cols) / (gflops * 1e6)
                            else:
                                time_ms = 0
                                
                            print_timer("loading", loading_ms)
                            print_timer("transfer", transfer_ms)
                            print_timer("operation", time_ms)
                        except ValueError:
                             print(f"Error: Could not parse GFLOPS from output: {content}", file=sys.stderr)
                    else:
                        print(f"Error: Invalid output format: {content}", file=sys.stderr)
            else:
                print("Error: Output file SpMM_GPU_SP.out not generated", file=sys.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running ASpT: {e}", file=sys.stderr)
            print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
        finally:
            if os.path.exists(tmp_mtx_path):
                os.remove(tmp_mtx_path)

if __name__ == "__main__":
    main()
