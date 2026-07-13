#!/usr/bin/env python3
"""
Count Non-Zero Blocks at Different Block Sizes

uses scipy.sparse matrices

"""

import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np

# Add operators directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'operators'))
from cusparse_utils import load_and_permute_matrix


def analyze_block_structure(matrix_path, block_sizes=None, perm_path=None, perm_type='ROW'):
    """
    Analyze block structure at multiple block sizes using scipy.sparse.
    
    Args:
        matrix_path: Path to Matrix Market file
        block_sizes: List of block sizes to analyze (default: powers of 2 from 4 to 128)
        perm_path: Optional path to permutation file (1-indexed)
        perm_type: Type of permutation ('ROW', 'SYMMETRIC', or 'ASYMMETRIC')
        
    Returns:
        Dictionary with analysis results
    """
    if block_sizes is None:
        block_sizes = [4, 8, 16, 32, 64, 128]
    
    # Load matrix with scipy (load_and_permute_matrix returns scipy.sparse matrix)
    A_scipy = load_and_permute_matrix(str(matrix_path), perm_path, perm_type)
    m, n = A_scipy.shape
    nnz = A_scipy.nnz
    
    # Extract indices once for all block sizes (avoid repeated extraction)
    A_scipy = A_scipy.tocoo()
    rows, cols = A_scipy.row, A_scipy.col
    
    results = {
        "matrix": str(Path(matrix_path).name),
        "rows": m,
        "cols": n,
        "nnz": nnz,
        "block_analysis": []
    }
    
    for bs in block_sizes:
        # Calculate number of block rows and columns
        block_rows = (m + bs - 1) // bs
        block_cols = (n + bs - 1) // bs
        total_blocks = block_rows * block_cols
        
        # Compute block indices using numpy vectorization
        block_row_indices = rows // bs
        block_col_indices = cols // bs
        
        # Count unique (block_row, block_col) pairs efficiently
        # Use pairing function to create unique integers, then count unique values
        # This is faster than np.unique on 2D array for large matrices
        block_ids = block_row_indices * block_cols + block_col_indices
        nonzero_blocks = len(np.unique(block_ids))
        
        block_info = {
            "block_size": bs,
            "block_rows": block_rows,
            "block_cols": block_cols,
            "total_blocks": total_blocks,
            "nonzero_blocks": nonzero_blocks,
        }
        
        results["block_analysis"].append(block_info)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Count non-zero blocks at different block sizes'
    )
    parser.add_argument('matrix', help='Path to Matrix Market file')
    parser.add_argument('--perm', help='Path to permutation file (optional)')
    parser.add_argument(
        '--perm-type',
        choices=['ROW', 'SYMMETRIC', 'ASYMMETRIC'],
        default='ROW',
        help='Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: ROW)'
    )
    parser.add_argument(
        '--block-sizes', '-b',
        type=int,
        nargs='+',
        help='Block sizes to analyze (default: 4 8 16 32 64 128)'
    )
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    try:
        t0 = time.perf_counter()
        results = analyze_block_structure(args.matrix, args.block_sizes, args.perm, args.perm_type)
        analysis_ms = (time.perf_counter() - t0) * 1000
        
        # Print results in easily parseable format
        print(f"MATRIX: {results['matrix']}")
        print(f"ROWS: {results['rows']}")
        print(f"COLS: {results['cols']}")
        print(f"NNZ: {results['nnz']}")
        
        for block_info in results['block_analysis']:
            bs = block_info['block_size']
            nb = block_info['nonzero_blocks']
            print(f"BLOCK_SIZE_{bs}: {nb}")
        
        print(f"<Timer>[analysis] {analysis_ms:.6f} ms", flush=True)
        
        # Also write full JSON to file if requested
        if args.output:
            indent = 2 if args.pretty else None
            json_output = json.dumps(results, indent=indent)
            with open(args.output, 'w') as f:
                f.write(json_output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
