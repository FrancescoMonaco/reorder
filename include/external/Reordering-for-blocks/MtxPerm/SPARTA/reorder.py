import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
import numpy as np

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from MtxPerm.utils import save_permutation

def main():
    parser = argparse.ArgumentParser(description='Generate SPARTA reordering permutation')
    parser.add_argument('matrix_path', help='Path to matrix market file')
    parser.add_argument('output_path', help='Path to save permutation file')
    parser.add_argument('--sparta-bin', default='MtxPerm/SPARTA/repo/programs/general/Matrix_Blocking', help='Path to SPARTA binary')
    parser.add_argument('--block-size', type=int, default=32, help='Block size for reordering')
    parser.add_argument('--tau', type=float, default=0.5, help='Similarity threshold (tau)')
    parser.add_argument('--algo', type=int, default=5, help='Blocking algorithm (default: 5 for DenseAMP)')
    
    args = parser.parse_args()
    
    matrix_path = Path(args.matrix_path)
    output_path = Path(args.output_path)
    sparta_bin = Path(args.sparta_bin).resolve()
    
    if not sparta_bin.exists():
        print(f"Error: SPARTA binary not found at {sparta_bin}")
        sys.exit(1)
        
    # Run SPARTA
    # -f: input file (.mtx supported with -R 1)
    # -R: matrix format (1 for mtx)
    # -b: col block size
    # -B: row block size (we set both to block_size)
    # -t: tau
    # -o: output prefix
    # -a: algorithm
    # -v: verbose (0=min)
    
    # SPARTA appends .g to the output filename for the grouping
    # We'll use a temp output file
    temp_output_prefix = output_path.with_suffix('.sparta_out')
    
    cmd = [
        str(sparta_bin),
        '-f', str(matrix_path),
        '-R', '1',
        '-b', str(args.block_size),
        '-B', str(args.block_size),
        '-t', str(args.tau),
        '-o', str(temp_output_prefix),
        '-a', str(args.algo),
        '-v', '1' 
    ]
    
    print(f"Running SPARTA: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"SPARTA failed with error code {e.returncode}")
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        sys.exit(1)

    # Print SPARTA's stdout (contains internal timing info)
    if result.stdout:
        print(result.stdout, end='')

    # Parse SPARTA's internal timer (blocking algorithm only, in microseconds)
    # The binary prints "timer: VALUE" where VALUE is in microseconds
    import re
    timer_match = re.search(r'^timer:\s+([0-9.e+\-]+)', result.stdout or '', re.MULTILINE)
    if timer_match:
        time_us = float(timer_match.group(1))
        time_ms = time_us / 1000.0
        print(f"<Timer>[reordering] {time_ms:.6f} ms")
        
    print(f"SPARTA run complete. Checking outputs...")
    
    # The grouping file should be at temp_output_prefix + ".g"
    grouping_file = Path(str(temp_output_prefix) + ".g")    
    try:
        # Read grouping (one integer per line)
        with open(grouping_file, 'r') as f:
            grouping = [int(line.strip()) for line in f if line.strip()]
        
        grouping = np.array(grouping)
        
        # Convert grouping to permutation
        # Sort indices based on grouping value
        # We want to keep rows in the same group together.
        # Stable sort is preferred to maintain relative order of rows within the same group (optional but good)
        perm = np.argsort(grouping, kind='stable').astype(np.int32)
        
        print(f"Saving permutation to {output_path}")
        save_permutation(args.output_path, perm)
        
    except Exception as e:
        print(f"Error parsing grouping file: {e}")
        # We don't have mat.shape[0] here anymore, but we can try to get it from the grouping if it was partially read
        # or just fail.
        sys.exit(1)

    # Clean up temp output files
    if grouping_file.exists():
        grouping_file.unlink()
    if temp_output_prefix.exists():
        temp_output_prefix.unlink()

if __name__ == '__main__':
    main()
