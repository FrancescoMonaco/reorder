#!/usr/bin/env python3
"""
TCA (Tensor Core Aware) Reordering Wrapper

Thin wrapper around the original TCA_reorder.py from DTC-SpMM (ASPLOS'24).
Converts MTX input to the NPZ format TCA_reorder.py expects, invokes it
as a subprocess, and converts the output to .perm format.

The algorithm performs two-level MinHash LSH clustering:
  Level 1 (TCU-aware): Clusters rows by sparsity pattern similarity
  Level 2 (Cache-aware): Re-clusters for better cache locality

Reference:
    Fan et al., "DTC-SpMM: Bridging the Gap in Accelerating General Sparse
    Matrix Multiplication with Tensor Cores", ASPLOS 2024

Original code: operators/DTC-SpMM/reordering/TCA_reorder.py
"""

import sys
import os
import re
import argparse
import tempfile
import time
import numpy as np
from pathlib import Path
from scipy.io import mmread


def main():
    parser = argparse.ArgumentParser(
        description='Generate TCA reordering permutation for sparse matrices',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('output_path', help='Path to save permutation file')
    parser.add_argument('--thres', type=int, default=128,
                        help='Cluster size threshold')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    tca_py = script_dir / ".." / ".." / "operators" / "DTC-SpMM" / "reordering" / "TCA_reorder.py"
    tca_py = tca_py.resolve()

    if not tca_py.exists():
        print(f"Error: TCA_reorder.py not found at {tca_py}", file=sys.stderr)
        print("Ensure DTC-SpMM is cloned (operators/install_dtc.sh).", file=sys.stderr)
        sys.exit(1)

    matrix_path = Path(args.matrix_path).resolve()
    output_path = Path(args.output_path).resolve()
    verbose = not args.quiet

    if not matrix_path.exists():
        print(f"Error: Matrix file not found: {matrix_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = "matrix"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # --- Load MTX and convert to NPZ (format TCA_reorder.py expects) ---
            if verbose:
                print(f"Loading matrix: {matrix_path}")
            t0 = time.perf_counter()
            A = mmread(str(matrix_path)).tocoo()
            load_ms = (time.perf_counter() - t0) * 1000
            print(f"<Timer>[loading] {load_ms:.6f} ms")

            npz_path = os.path.join(tmpdir, dataset_name + ".npz")
            np.savez(npz_path,
                     src_li=A.row.astype(np.int64),
                     dst_li=A.col.astype(np.int64),
                     num_nodes=np.array(A.shape[0]))

            if verbose:
                print(f"Matrix: {A.shape[0]} rows, {A.shape[1]} cols, {A.nnz} nnz")

            # --- Patch and call TCA_reorder.py ---
            # The original script has hardcoded paths for input/output.
            # We create a patched copy that reads from our temp directory.
            if verbose:
                print(f"Running TCA_reorder.py (thres={args.thres})...")

            patched_script = os.path.join(tmpdir, "TCA_reorder_patched.py")
            with open(tca_py, 'r') as f:
                original_code = f.read()

            # Replace the hardcoded input path
            patched_code = original_code.replace(
                'path = osp.join("/mnt/raid/fanruibo/g_dataset/", dataset + ".npz")',
                f'path = osp.join("{tmpdir}/", dataset + ".npz")'
            )
            # Replace the hardcoded output path
            patched_code = patched_code.replace(
                'np.savez(osp.join("/mnt/raid/fanruibo/asplos_dtc_dataset/", dataset + ".reorder_id.npz"), reorder_id = reorder)',
                f'np.savez(osp.join("{tmpdir}/", dataset + ".reorder_id.npz"), reorder_id = reorder)'
            )
            patched_code = patched_code.replace(
                'np.savez(osp.join("/mnt/raid/fanruibo/asplos_dtc_dataset/", dataset + ".reorder.npz"), src_li = new_row_ind, dst_li = new_col_ind, num_nodes = num_row)',
                f'np.savez(osp.join("{tmpdir}/", dataset + ".reorder.npz"), src_li = new_row_ind, dst_li = new_col_ind, num_nodes = num_row)'
            )

            with open(patched_script, 'w') as f:
                f.write(patched_code)

            import subprocess
            cmd = [
                sys.executable, patched_script,
                "--dataset", dataset_name,
                "--thres", str(args.thres),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if verbose and result.stdout:
                print(result.stdout, end='')
            if result.returncode != 0:
                print("Error: TCA_reorder.py failed", file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                sys.exit(1)

            # Parse timing lines reported by TCA_reorder.py itself
            # e.g. "init LSH time (s) 1.189" or "clustering time (s):  0.219"
            time_pattern = re.compile(r'time \(s\):?\s+([0-9.]+)')
            tca_total_s = sum(float(m.group(1))
                              for m in time_pattern.finditer(result.stdout))
            print(f"<Timer>[reordering] {tca_total_s * 1000:.6f} ms")

            # --- Read reorder_id and convert to .perm ---
            reorder_npz = os.path.join(tmpdir, dataset_name + ".reorder_id.npz")
            if not os.path.exists(reorder_npz):
                print(f"Error: Expected output not found: {reorder_npz}",
                      file=sys.stderr)
                sys.exit(1)

            reorder_id = np.load(reorder_npz)["reorder_id"]
            # reorder_id[i] = original node visited at position i (0-based)
            # Convert to 1-based permutation
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
