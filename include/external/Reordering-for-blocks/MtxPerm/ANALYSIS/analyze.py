#!/usr/bin/env python3
"""
Analyze sparse matrix properties: Block structure, Bandwidth, Locality.
"""

import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np

# Add MtxPerm directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_and_permute_matrix, apply_permutation

# Import local analysis utils
from analysis_utils import analyze_block_structure, analyze_bandwidth, analyze_locality, analyze_access_distances

def main():
    parser = argparse.ArgumentParser(
        description='Analyze sparse matrix properties'
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
    parser.add_argument('--base-perm', help='Base permutation file (applied first)')
    parser.add_argument(
        '--base-perm-type',
        choices=['ROW', 'SYMMETRIC', 'ASYMMETRIC'],
        default='SYMMETRIC',
        help='Type of base permutation (default: SYMMETRIC)'
    )
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    try:
        t0 = time.perf_counter()
        
        # Load matrix (optionally with base permutation applied first)
        print(f"Loading matrix: {args.matrix}...", file=sys.stderr)
        A_scipy = load_and_permute_matrix(
            args.matrix, args.perm, args.perm_type,
            base_perm_path=args.base_perm, base_perm_type=args.base_perm_type
        )
        m, n = A_scipy.shape
        nnz = A_scipy.nnz
        
        loading_ms = (time.perf_counter() - t0) * 1000

        # Initialize results
        results = {
            "time_loading_ms": loading_ms,
            "matrix": str(Path(args.matrix).name),
            "rows": m,
            "cols": n,
            "nnz": nnz,
            "density": nnz / (m * n) if m*n > 0 else 0,
            "permutation": args.perm if args.perm else "None",
            "perm_type": args.perm_type,
            "base_permutation": args.base_perm if args.base_perm else "None",
            "base_perm_type": args.base_perm_type
        }
        
        # 1. Block Structure Analysis
        print("Analyzing block structure...", file=sys.stderr)
        t_block = time.perf_counter()
        results["block_analysis"] = analyze_block_structure(A_scipy, args.block_sizes)
        results["time_block_analysis_ms"] = (time.perf_counter() - t_block) * 1000
        
        # 2. Bandwidth Analysis
        print("Analyzing bandwidth...", file=sys.stderr)
        t_bw = time.perf_counter()
        results["bandwidth"] = analyze_bandwidth(A_scipy)
        results["time_bandwidth_analysis_ms"] = (time.perf_counter() - t_bw) * 1000
        
        # 3. Locality Analysis
        print("Analyzing locality...", file=sys.stderr)
        t_loc = time.perf_counter()
        results["locality"] = analyze_locality(A_scipy)
        results["time_locality_analysis_ms"] = (time.perf_counter() - t_loc) * 1000

        # 4. Access Distance Analysis
        print("Analyzing access distances...", file=sys.stderr)
        t_dist = time.perf_counter()
        results["access_distances"] = analyze_access_distances(A_scipy)
        results["time_access_distances_ms"] = (time.perf_counter() - t_dist) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        results["total_analysis_time_ms"] = total_ms
        
        # Output results
        indent = 2 if args.pretty else None
        json_output = json.dumps(results, indent=indent)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(json_output)
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
