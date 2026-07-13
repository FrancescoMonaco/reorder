#!/usr/bin/env python3
"""
Groot Reordering Wrapper

Thin wrapper around the official groot.py from Groot-EuroSys25.
Converts MTX input to the NPZ format groot.py expects, invokes it
as a subprocess, and converts the output to .perm format.

Reference:
    Chen et al., "Groot: Graph-Centric Row Reordering with Tree for
    Sparse Matrix Multiplications on Tensor Cores", EuroSys 2025

Repository: https://github.com/yuang-chen/Groot-EuroSys25
"""

import sys
import os
import argparse
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from scipy.io import mmread


def main():
    parser = argparse.ArgumentParser(
        description='Generate Groot reordering permutation for sparse matrices',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('output_path', help='Path to save permutation file')
    parser.add_argument('--knn', type=int, default=16,
                        help='Number of nearest neighbors for KNN graph')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    groot_py = script_dir / "Groot-EuroSys25" / "groot.py"

    if not groot_py.exists():
        print(f"Error: groot.py not found at {groot_py}", file=sys.stderr)
        print("Run install.sh first to clone the Groot-EuroSys25 repository.", file=sys.stderr)
        sys.exit(1)

    matrix_path = Path(args.matrix_path).resolve()
    output_path = Path(args.output_path).resolve()
    verbose = not args.quiet

    if not matrix_path.exists():
        print(f"Error: Matrix file not found: {matrix_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = "matrix"
    knn = args.knn
    # groot.py default suffix: knn.k{knn}.mst.jaccard.dfs
    suffix = f"knn.k{knn}.mst.jaccard.dfs"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # --- Load MTX and convert to NPZ ---
            if verbose:
                print(f"Loading matrix: {matrix_path}")
            t0 = time.perf_counter()
            A = mmread(str(matrix_path)).tocoo()
            load_ms = (time.perf_counter() - t0) * 1000
            print(f"<Timer>[loading] {load_ms:.6f} ms")

            npz_path = os.path.join(tmpdir, f"{dataset_name}.npz")
            np.savez(npz_path,
                     src_li=A.row.astype(np.int64),
                     dst_li=A.col.astype(np.int64),
                     num_nodes=np.array(A.shape[0]))

            # --- Call groot.py ---
            if verbose:
                print(f"Running groot.py (knn={knn})...")
            t0 = time.perf_counter()

            cache_dir = os.path.join(tmpdir, "knn_cache")
            cmd = [
                sys.executable, str(groot_py),
                "--dataset", dataset_name,
                "--input_dir", tmpdir,
                "--output_dir", tmpdir,
                "--cache_dir", cache_dir,
                "--knn", str(knn),
            ]
            result = subprocess.run(
                cmd,
                capture_output=not verbose,
                text=True,
            )
            if result.returncode != 0:
                print("Error: groot.py failed", file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                sys.exit(1)

            reorder_ms = (time.perf_counter() - t0) * 1000
            print(f"<Timer>[reordering] {reorder_ms:.6f} ms")

            # --- Read reorder_id and convert to .perm ---
            reorder_npz = os.path.join(
                tmpdir, f"{dataset_name}.{suffix}.reorder_id.npz"
            )
            if not os.path.exists(reorder_npz):
                print(f"Error: Expected output not found: {reorder_npz}",
                      file=sys.stderr)
                sys.exit(1)

            reorder_id = np.load(reorder_npz)["reorder_id"]
            # reorder_id is the traversal order: reorder_id[i] = original node
            # visited at position i. Convert to 1-based permutation.
            perm_1based = reorder_id + 1
            with open(output_path, 'w') as f:
                f.write(' '.join(map(str, perm_1based)) + '\n')

            if verbose:
                print(f"Permutation saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
