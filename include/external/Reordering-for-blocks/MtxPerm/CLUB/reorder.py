#!/usr/bin/env python3
"""
CLUB reordering wrapper.

Loads a Matrix Market file, runs the C++ CLUB reorder2 implementation,
and writes the resulting row permutation as a 1-based .perm file.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[5]))

from MtxPerm.utils import save_permutation

try:
    from CLUB._reorder_impl import reorder2, multilevel_mask, micro_macro_mask
except ImportError as exc:
    print(
        "Error: could not import reorder._reorder_impl. "
        "Build the CMake target before running CLUB/reorder.py.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

# Reordering technique to use. One of:
#   "micromacro"  - single-window sketch + micro-macro clustering (BFS on
#                   the macro graph of micro-clusters)
#   "multilevel"  - multi-resolution sketch (list of window sizes) +
#                   lexicographic clustering
#   "classic"     - two-sided alternating reordering (reorder2)
TECHNIQUE = "micromacro"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLUB reordering permutation for sparse matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix_path", help="Path to Matrix Market file")
    parser.add_argument("output_path", help="Path to save permutation file")
    parser.add_argument("--w", type=int, default=32, help="Sketch window size")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=4,
        help="Maximum number of alternating row/column passes",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default=None,
        choices=["micromacro", "multilevel", "classic"],
        help=(
            "Reordering technique to use. Overrides the module-level "
            "TECHNIQUE variable: 'micromacro' (single-window sketch + "
            "micro-macro clustering), 'multilevel' (multi-resolution sketch "
            "+ lexicographic clustering), 'classic' (two-sided reorder2)"
        ),
    )
    parser.add_argument(
        "--micro-threshold",
        type=int,
        default=0,
        help="Micro-cluster size threshold for 'micromacro' (0 = auto: sqrt(n))",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    matrix_path = Path(args.matrix_path).resolve()
    output_path = Path(args.output_path).resolve()
    verbose = not args.quiet

    if not matrix_path.exists():
        print(f"Error: Matrix file not found: {matrix_path}", file=sys.stderr)
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    technique = args.technique if args.technique is not None else TECHNIQUE

    try:
        if verbose:
            print(f"Loading matrix: {matrix_path}")
        t0 = time.perf_counter()
        matrix = scipy.io.mmread(str(matrix_path)).tocsr()
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"<Timer>[loading] {load_ms:.6f} ms")

        if verbose:
            print(f"Running CLUB reordering (technique={technique})...")
        t0 = time.perf_counter()
        if technique == "classic":
            perm = reorder2(str(matrix_path), args.w, args.max_iters)
        elif technique == "multilevel":
            perm = multilevel_mask(str(matrix_path), [32, 16, 4, 1])
        elif technique == "micromacro":
            perm = micro_macro_mask(str(matrix_path), args.w, args.micro_threshold)
        else:
            raise ValueError(f"Unknown technique: {technique!r}")
        perm = np.asarray(perm)
        reorder_ms = (time.perf_counter() - t0) * 1000
        print(f"<Timer>[reordering] {reorder_ms:.6f} ms")

        if len(perm) != matrix.shape[0]:
            raise RuntimeError(
                f"Permutation length ({len(perm)}) does not match matrix rows ({matrix.shape[0]})"
            )

        save_permutation(output_path, perm)

        if verbose:
            print(f"Permutation saved to {output_path}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()