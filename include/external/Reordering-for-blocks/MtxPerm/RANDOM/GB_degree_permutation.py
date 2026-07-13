#!/usr/bin/env python3
"""Generate a degree-based row permutation for a Matrix Market sparse matrix.

Sorts vertices by ascending row degree (number of nonzeros per row) using a
stable sort, producing the same ordering as the SparseBase degree_perm binary.

Requirements
------------
    pip install numpy scipy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a degree-based row permutation for a Matrix Market matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix", type=Path, help="Path to input .mtx matrix")
    parser.add_argument("output", type=Path, help="Output permutation file (1-based)")

    args = parser.parse_args()

    # Load matrix as CSR (mmread expands symmetric matrices automatically)
    t_start = time.perf_counter()
    A = scipy.io.mmread(args.matrix).tocsr()
    t_load = (time.perf_counter() - t_start) * 1000.0
    print(f"<Timer>[loading] {t_load:.6f} ms")

    # Compute row degrees and sort ascending (stable)
    t_start = time.perf_counter()
    degrees = np.diff(A.indptr)
    perm = np.argsort(degrees, kind='stable') + 1  # 1-based
    t_reorder = (time.perf_counter() - t_start) * 1000.0
    print(f"<Timer>[reordering] {t_reorder:.6f} ms")

    # Save permutation (space-separated on single line, 1-based)
    with open(args.output, 'w') as f:
        f.write(' '.join(map(str, perm)) + '\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())
