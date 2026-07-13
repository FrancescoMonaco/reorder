#!/usr/bin/env python3
"""
Pre-permute a sparse matrix and write the result as a new .mtx file.

Usage:
    python3 MtxPerm/pre_permute.py <input.mtx> <output.mtx> --perm <perm_file> --perm-type SYMMETRIC
"""

import sys
import argparse
from pathlib import Path

from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_permute_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Apply a permutation to a sparse matrix and write the result'
    )
    parser.add_argument('input', help='Path to input Matrix Market file')
    parser.add_argument('output', help='Path to output Matrix Market file')
    parser.add_argument('--perm', required=True, help='Path to permutation file')
    parser.add_argument(
        '--perm-type',
        choices=['ROW', 'SYMMETRIC', 'ASYMMETRIC'],
        default='SYMMETRIC',
        help='Type of permutation (default: SYMMETRIC)'
    )

    args = parser.parse_args()

    A = load_and_permute_matrix(args.input, args.perm, args.perm_type)
    mmwrite(args.output, A)

    return 0


if __name__ == '__main__':
    sys.exit(main())
