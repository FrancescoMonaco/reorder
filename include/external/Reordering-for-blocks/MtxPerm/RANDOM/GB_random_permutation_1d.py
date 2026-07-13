#!/usr/bin/env python3
"""Generate a random row permutation for a Matrix Market sparse matrix.

Generates a random permutation for the rows of a matrix (works for both square
and rectangular matrices). Output is 1-based indices compatible with reordering tools.

Requirements
------------
    pip install numpy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np


def read_mtx_dimensions(matrix_path: Path) -> tuple[int, int]:
    """Read matrix dimensions from a Matrix Market file header."""
    with open(matrix_path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.split()
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Could not parse dimensions from {matrix_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a random row permutation for a Matrix Market matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix", type=Path, help="Path to input .mtx matrix")
    parser.add_argument("output", type=Path, help="Output permutation file (1-based)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    # Read matrix dimensions from header (instant, no full load)
    t_start = time.perf_counter()
    n_rows, _ = read_mtx_dimensions(args.matrix)
    t_load = (time.perf_counter() - t_start) * 1000.0
    print(f"<Timer>[loading] {t_load:.6f} ms")

    # Generate random permutation (reordering only)
    t_start = time.perf_counter()
    perm = np.random.permutation(n_rows) + 1  # 1-based
    t_reorder = (time.perf_counter() - t_start) * 1000.0
    print(f"<Timer>[reordering] {t_reorder:.6f} ms")
    
    # Save permutation (space-separated on single line, 1-based)
    # This format works for both ROW and SYMMETRIC permutation types
    with open(args.output, 'w') as f:
        f.write(' '.join(map(str, perm)) + '\n')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
