#!/usr/bin/env python3
"""
AccSpMM (PPoPP 2025 TF32 Tensor Core) SpMM using Python wrapper.
Applies permutations in Python, then calls AccSpMM CUDA binary.
"""

import sys
import os
import argparse

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from accspmm_utils import run_accspmm_spmm
from cusparse_utils import print_timer
from config import SPMM_N_COLS_DEFAULT, PERM_TYPE_DEFAULT


def main():
    parser = argparse.ArgumentParser(description='AccSpMM SpMM using TF32 Tensor Cores')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')

    args = parser.parse_args()

    try:
        results = run_accspmm_spmm(
            matrix_path=args.matrix_path,
            perm_path=args.perm,
            perm_type=args.perm_type,
            base_perm_path=args.base_perm,
            base_perm_type=args.base_perm_type,
            n_cols=args.n_cols,
        )

        print_timer("loading", results['loading_ms'])
        print_timer("write", results['write_ms'])

        if results['accspmm_kernel_ms'] is not None:
            print_timer("operation", results['accspmm_kernel_ms'])
        else:
            print("WARNING: Could not parse kernel timing from AccSpMM output", file=sys.stderr)

        # Print AccSpMM output in a box
        if results['accspmm_output'].strip():
            print("\n┌─── AccSpMM Output " + "─" * 57 + "┐")
            for line in results['accspmm_output'].strip().split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘\n")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTo install AccSpMM, run: bash operators/install_accspmm.sh", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running AccSpMM: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
