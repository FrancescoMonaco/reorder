#!/usr/bin/env python3
"""
SMaT (Sparse Matrix Tensor Core) SpMM using Python wrapper.
Applies permutations in Python, then calls smat CUDA binary.
"""

import sys
import argparse
from smat_utils import run_smat_spmm, parse_smat_output
from cusparse_utils import print_timer
from config import ALPHA_DEFAULT, BETA_DEFAULT, SPMM_N_COLS_DEFAULT, N_ITERATIONS_DEFAULT, BSR_BLOCKSIZE_DEFAULT, PERM_TYPE_DEFAULT


def main():
    parser = argparse.ArgumentParser(description='SMaT SpMM using Tensor Cores')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--alpha', type=float, default=ALPHA_DEFAULT, help=f'Alpha scalar (default: {ALPHA_DEFAULT})')
    parser.add_argument('--beta', type=float, default=BETA_DEFAULT, help=f'Beta scalar (default: {BETA_DEFAULT})')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')
    parser.add_argument('--n-iterations', type=int, default=N_ITERATIONS_DEFAULT, help=f'Number of timing iterations (default: {N_ITERATIONS_DEFAULT})')
    parser.add_argument('--blocksize', type=int, default=BSR_BLOCKSIZE_DEFAULT, help=f'Block size for Tensor Core optimization (default: {BSR_BLOCKSIZE_DEFAULT})')
    
    args = parser.parse_args()
    
    try:
        # Run SMaT SpMM
        results = run_smat_spmm(
            matrix_path=args.matrix_path,
            perm_path=args.perm,
            perm_type=args.perm_type,
            base_perm_path=args.base_perm,
            base_perm_type=args.base_perm_type,
            n_cols=args.n_cols,
            blocksize=args.blocksize,
            n_iterations=args.n_iterations,
            alpha=args.alpha,
            beta=args.beta
        )
        
        # Print timing results in consistent format
        print_timer("loading", results['loading_ms'])
        print_timer("write", results['write_ms'])
        
        # Use kernel time if available, otherwise total time
        operation_ms = results.get('smat_kernel_ms') or results['smat_total_ms']
        print_timer("operation", operation_ms)
        
        # Print smat output in a box
        if results['smat_output'].strip():
            print("\n┌─── SMaT Output " + "─" * 60 + "┐")
            for line in results['smat_output'].strip().split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTo install SMaT, run: bash operators/install_smat.sh", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running SMaT: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
