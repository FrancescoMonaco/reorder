#!/usr/bin/env python3
"""Generate a Reverse Cuthill-McKee (RCM) permutation for a Matrix Market sparse matrix.

Uses scipy.sparse.csgraph.reverse_cuthill_mckee which is a fast, well-tested
implementation. Produces a New-to-Old permutation (1-based) compatible with the
reordering pipeline.

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
from scipy.sparse.csgraph import reverse_cuthill_mckee


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an RCM permutation for a Matrix Market matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix", type=Path, help="Path to input .mtx matrix")
    parser.add_argument("output", type=Path, help="Output permutation file (1-based)")

    args = parser.parse_args()

    # Load matrix as CSC (mmread expands symmetric matrices automatically)
    t_start = time.perf_counter()
    A = scipy.io.mmread(args.matrix).tocsc()
    t_load = (time.perf_counter() - t_start) * 1000.0
    print(f"<Timer>[loading] {t_load:.6f} ms")

    m, n = A.shape
    if m != n:
        print(f"Error: RCM requires a square matrix. Got {m} x {n}", file=sys.stderr)
        return 1

    # Compute RCM ordering
    # reverse_cuthill_mckee returns a New-to-Old permutation (0-based)
    # which is exactly what our pipeline expects
    t_start = time.perf_counter()
    perm = reverse_cuthill_mckee(A, symmetric_mode=True)
    t_reorder = (time.perf_counter() - t_start) * 1000.0
    print(f"<Timer>[reordering] {t_reorder:.6f} ms")

    # Convert to 1-based and save (space-separated on single line)
    perm_1based = perm + 1
    with open(args.output, 'w') as f:
        f.write(' '.join(map(str, perm_1based)) + '\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())
