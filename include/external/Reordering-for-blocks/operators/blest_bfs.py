#!/usr/bin/env python3
"""
BLEST BFS Benchmark wrapper.

Loads and permutes the matrix in Python, writes a temp .mtx file,
then invokes the blest_driver binary with --no-reorder.
"""

import sys
import os
import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import scipy.io
import scipy.sparse

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cusparse_utils import load_and_permute_matrix, print_timer
from config import PERM_TYPE_DEFAULT


def main():
    parser = argparse.ArgumentParser(description='BLEST BFS Benchmark')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT,
                        help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--n-sources', type=int, default=64, help='Number of BFS source vertices (default: 64)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for source selection (default: 42)')

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

    # 2. Find binary
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    binary = project_root / "MtxPerm" / "BLEST" / "blest" / "build" / "blest_driver"

    if not binary.exists():
        print(f"Error: blest_driver binary not found at {binary}. Please run MtxPerm/BLEST/install.sh",
              file=sys.stderr)
        sys.exit(1)

    # 3. Write matrix to temp file
    # BLEST expects symmetric matrices; ensure we write the full (symmetrized) matrix
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp_mtx:
        # mmwrite with symmetry='general' to write full matrix (BLEST symmetrizes internally)
        scipy.io.mmwrite(tmp_mtx, A, symmetry='general')
        tmp_mtx_path = tmp_mtx.name
    transfer_ms = (time.perf_counter() - t0) * 1000

    # 4. Run blest_driver with --no-reorder (external perm already applied)
    cmd = [
        str(binary),
        tmp_mtx_path,
        "--no-reorder",
        "--n-sources", str(args.n_sources),
        "--seed", str(args.seed),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print loading time (Python-side matrix load + permute + write)
        print_timer("loading", loading_ms + transfer_ms)

        # Parse and re-print <Timer> lines from the binary output
        for line in result.stdout.splitlines():
            if "<Timer>" in line:
                # Skip the binary's own loading timer (we report our own)
                if "[loading]" in line:
                    continue
                print(line, flush=True)

        if result.returncode != 0:
            print(f"Error running blest_driver (exit code {result.returncode})", file=sys.stderr)
            if result.stderr:
                print(f"Stderr: {result.stderr}", file=sys.stderr)

    except Exception as e:
        print(f"Error running blest_driver: {e}", file=sys.stderr)
    finally:
        if os.path.exists(tmp_mtx_path):
            os.remove(tmp_mtx_path)


if __name__ == "__main__":
    main()
