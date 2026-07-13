#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE CSR SpMM using CuPy.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
import time
from cusparse_utils import load_and_permute_matrix, convert_to_gpu, time_operation, print_timer
from config import ALPHA_DEFAULT, BETA_DEFAULT, SPMM_N_COLS_DEFAULT, N_ITERATIONS_DEFAULT, PERM_TYPE_DEFAULT


def main():
    parser = argparse.ArgumentParser(description='Minimal cuSPARSE CSR SpMM using CuPy')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--alpha', type=float, default=ALPHA_DEFAULT, help=f'Alpha scalar (default: {ALPHA_DEFAULT})')
    parser.add_argument('--beta', type=float, default=BETA_DEFAULT, help=f'Beta scalar (default: {BETA_DEFAULT})')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')
    parser.add_argument('--n-iterations', type=int, default=N_ITERATIONS_DEFAULT, help=f'Number of timing iterations (default: {N_ITERATIONS_DEFAULT})')
    
    args = parser.parse_args()
    
    # Load and permute matrix
    t0 = time.perf_counter()
    A_cpu = load_and_permute_matrix(args.matrix_path, args.perm, args.perm_type,
                                    base_perm_path=args.base_perm, base_perm_type=args.base_perm_type)
    loading_ms = (time.perf_counter() - t0) * 1000
    m, n = A_cpu.shape

    # Convert to CSR and transfer to GPU
    t0 = time.perf_counter()
    A_gpu = convert_to_gpu(A_cpu, 'csr')
    transfer_ms = (time.perf_counter() - t0) * 1000
    
    # Create dense matrix B (n x n_cols) and C (m x n_cols) on GPU
    B_gpu = cp.random.randn(n, args.n_cols, dtype=cp.float32)
    C_gpu = cp.random.randn(m, args.n_cols, dtype=cp.float32)
    
    # Time the operation
    def spmm_op():
        return args.alpha * A_gpu.dot(B_gpu) + args.beta * C_gpu
    
    avg_time_ms = time_operation(spmm_op, args.n_iterations)
    
    print_timer("loading", loading_ms)
    print_timer("transfer", transfer_ms)
    print_timer("operation", avg_time_ms)
    return 0


if __name__ == '__main__':
    sys.exit(main())
