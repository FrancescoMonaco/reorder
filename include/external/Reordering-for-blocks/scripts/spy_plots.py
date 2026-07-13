#!/usr/bin/env python3
"""
Generate spy plots showing matrix structure before and after reordering.

This script creates visualizations of sparse matrix nonzero patterns
to illustrate the effect of different reordering algorithms.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import mmread
from scipy.sparse import csr_matrix
from settings import PERMS, get_perm_display, get_perm_color
from plot_utils import set_professional_style


def load_permutation_file(perm_path, perm_type='SYMMETRIC'):
    """
    Load permutation file and return row and/or column permutations.
    
    Args:
        perm_path: Path to permutation file
        perm_type: Type of permutation:
            'ROW' - single line with row permutation (only permute rows)
            'SYMMETRIC' - single line, apply to both rows and cols
            'ASYMMETRIC' - two lines: first=rows, second=cols
    
    Returns:
        Tuple of (row_perm, col_perm) as numpy arrays (0-indexed)
    """
    perm_type = perm_type.upper()
    
    with open(perm_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if perm_type == 'ROW':
        perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        return perm, None
    elif perm_type == 'SYMMETRIC':
        perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        return perm, perm.copy()
    elif perm_type == 'ASYMMETRIC':
        if len(lines) < 2:
            raise ValueError(f"ASYMMETRIC permutation requires 2 lines, found {len(lines)}")
        row_perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        col_perm = np.fromstring(lines[1], sep=' ', dtype=np.int64) - 1
        return row_perm, col_perm
    else:
        raise ValueError(f"Unknown permutation type: {perm_type}")


def apply_permutation(A, row_perm, col_perm):
    """Apply row and column permutations to a sparse matrix."""
    if row_perm is not None:
        A = A[row_perm, :]
    if col_perm is not None:
        A = A[:, col_perm]
    return A


def _get_canonical_order(labels):
    """Return *labels* sorted in canonical PERMS order (Original first).

    Unknown labels are appended alphabetically at the end.
    """
    seen = set()
    canonical = []
    for name in ['Original'] + [p['display'] for p in PERMS.values()]:
        if name not in seen:
            seen.add(name)
            canonical.append(name)
    order = [l for l in canonical if l in labels]
    extra = sorted(set(labels) - set(order))
    return order + extra


def _label_color(label):
    """Return the PERMS color for a display-name label, grey for Original."""
    if label == 'Original':
        return '#888888'
    # Reverse-lookup: display name -> color
    for p in PERMS.values():
        if p['display'] == label:
            return p['color']
    return '#333333'


def create_spy_plot(matrices_dict, output_path, matrix_name, figsize=None, markersize=0.5):
    """
    Create a multi-panel spy plot showing original and reordered matrices.

    Args:
        matrices_dict: Dict of {label: sparse_matrix}  (labels are display names)
        output_path: Path to save the figure
        matrix_name: Name of the matrix for the title
        figsize: Figure size (auto-calculated if None)
        markersize: Size of markers in spy plot
    """
    # Sort panels in canonical order
    ordered_labels = _get_canonical_order(list(matrices_dict.keys()))

    n_plots = len(ordered_labels)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3.5 * n_cols, 3.5 * n_rows + 0.6)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    first_A = matrices_dict[ordered_labels[0]]
    for idx, label in enumerate(ordered_labels):
        A = matrices_dict[label]
        ax = axes[idx]
        color = _label_color(label)
        ax.spy(A, markersize=markersize, aspect='equal', color=color)
        ax.set_title(label, fontsize=14, fontweight='bold', color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    matrix_stem = Path(matrix_name).stem
    fig.suptitle(f'{matrix_stem}  ({first_A.shape[0]:,} $\\times$ {first_A.shape[1]:,},  nnz = {first_A.nnz:,})',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def find_permutation_file(perm_dir, matrix_name, algorithm):
    """
    Find the permutation file for a given matrix and algorithm.
    
    Expected directory structure (based on YAML configs):
        perm_dir/
            ALGORITHM/
                MATRIX_NAME.perm
    
    Examples:
        perms/SB_gray/bcsstk10.perm
        perms/GROOT_reorder/bcsstk10.perm
    """
    matrix_base = Path(matrix_name).stem  # Remove .mtx extension
    
    # Try different path patterns based on actual project structure
    patterns = [
        # Primary pattern: perms/ALGORITHM/matrix.perm
        perm_dir / algorithm / f"{matrix_base}.perm",
        # Alternative: perms/ALGORITHM/matrix_reorder.perm
        perm_dir / algorithm / f"{matrix_base}_reorder.perm",
        # Nested: perms/ALGORITHM/matrix/matrix.perm
        perm_dir / algorithm / matrix_base / f"{matrix_base}.perm",
        # GROOT style: perms/GROOT_reorder/matrix.perm
        perm_dir / f"{algorithm}_reorder" / f"{matrix_base}.perm",
        # Lowercase algorithm
        perm_dir / algorithm.lower() / f"{matrix_base}.perm",
    ]
    
    for pattern in patterns:
        if pattern.exists():
            return pattern
    
    return None


def get_perm_type_for_algorithm(algorithm, analysis_df, matrix_name):
    """Get the permutation type for a given algorithm from analysis data.

    Prefers SYMMETRIC when both ROW and SYMMETRIC entries exist for the
    same (matrix, algorithm) pair.
    """
    rows = analysis_df[(analysis_df['matrix'] == matrix_name) &
                       (analysis_df['perm'] == algorithm)]
    if not rows.empty:
        perm_types = set(rows['perm_type'].unique())
        if 'SYMMETRIC' in perm_types:
            return 'SYMMETRIC'
        return rows.iloc[0]['perm_type']
    return 'SYMMETRIC'  # Default


def main():
    parser = argparse.ArgumentParser(
        description="Generate spy plots for matrices before/after reordering."
    )
    parser.add_argument(
        "--matrices-dir", type=str, required=True,
        help="Base directory containing matrix files"
    )
    parser.add_argument(
        "--perms-dir", type=str, required=True,
        help="Base directory containing permutation files organized by algorithm"
    )
    parser.add_argument(
        "--analysis-csv", type=str, default="results/results_analysis.csv",
        help="Path to analysis CSV for perm_type info"
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots/spy_plots",
        help="Output directory for spy plots"
    )
    parser.add_argument(
        "--matrices", type=str, nargs="+", default=None,
        help="Specific matrices to plot (default: read from spy_matrices.txt)"
    )
    parser.add_argument(
        "--matrices-file", type=str, default=None,
        help="File containing matrix names, one per line (default: scripts/spy_matrices.txt)"
    )
    parser.add_argument(
        "--algorithms", type=str, nargs="+", default=None,
        help="Specific algorithms to include (default: all available)"
    )
    parser.add_argument(
        "--max-size", type=int, default=20000,
        help="Maximum matrix dimension for plotting (default: 20000)"
    )
    parser.add_argument(
        "--markersize", type=float, default=0.005,
        help="Marker size for spy plot (default: 0.005)"
    )
    parser.add_argument(
        "--use-curated", action="store_true",
        help="Use a curated list of matrices known to show interesting reordering patterns"
    )
    
    args = parser.parse_args()
    
    # Curated list of matrices that demonstrate interesting reordering effects
    # Selected for: diverse structures, reasonable size for visualization, 
    # and significant improvement potential
    CURATED_MATRICES = [
        "bcsstk10.mtx",      # FEM: structural, shows block structure after reorder
        "bcsstk33.mtx",      # FEM: larger structural problem
        "can_1054.mtx",      # FEM: oil reservoir simulation
        "ca-GrQc.mtx",       # Social: arXiv collaboration network
        "email-Enron.mtx",   # Social: email network (power-law degree)
        "delaunay_n13.mtx",  # Mesh: Delaunay triangulation
        "ex19.mtx",          # CFD: computational fluid dynamics
        "memplus.mtx",       # Circuit: memory circuit
        "circuit_2.mtx",     # Circuit: circuit simulation
        "USpowerGrid.mtx",   # Infrastructure: US power grid
    ]
    
    matrices_dir = Path(args.matrices_dir)
    perms_dir = Path(args.perms_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use publication-quality style from the plotting pipeline
    set_professional_style()
    
    # Load analysis data
    print(f"Loading analysis data from {args.analysis_csv}...")
    try:
        df = pd.read_csv(args.analysis_csv)
        df['perm'] = df['perm'].fillna('None').astype(str)
    except Exception as e:
        print(f"Warning: Could not load analysis CSV: {e}")
        df = pd.DataFrame()
    
    # Algorithms excluded from spy plots (not informative / not in scope)
    SPY_EXCLUDE = {'SPARTA_reorder', 'random2D', 'ACCORDER_reorder', 'graphblas'}

    # Get available algorithms from perms directory or use defaults
    if args.algorithms:
        algorithms = [a for a in args.algorithms if a not in SPY_EXCLUDE]
    else:
        # Default algorithms based on analysis data
        # These match the permutation names in results_analysis.csv
        algorithms = [
            'SB_rcm',        # Reverse Cuthill-McKee (bandwidth reduction)
            'SB_amd',        # Approximate Minimum Degree
            'SB_metis',      # Graph partitioning
            'SB_gray',       # Gray code ordering
            'SB_degree',     # Degree-based ordering
            'SB_rabbit',     # Rabbit ordering
            'SB_patoh',      # PaToH hypergraph partitioning
            'GROOT_reorder', # GROOT GPU reordering
            'TCA_reorder',   # DTC-LSH reordering
            'random1D',      # Random permutation (baseline)
        ]

        # Also check what's actually in the perms directory
        if perms_dir.exists():
            available = [d.name for d in perms_dir.iterdir() if d.is_dir()]
            # Keep only algorithms that exist, excluding blacklisted ones
            algorithms = [a for a in algorithms if a in available and a not in SPY_EXCLUDE] + \
                        [a for a in available if a not in algorithms and a not in SPY_EXCLUDE]
    
    print(f"Algorithms to process: {algorithms}")
    
    # Select matrices to process
    if args.matrices:
        selected_matrices = args.matrices
    elif args.matrices_file:
        # Read from specified file
        with open(args.matrices_file, 'r') as f:
            selected_matrices = [line.strip() for line in f
                                if line.strip() and not line.startswith('#')]
    elif (Path(__file__).parent / 'spy_matrices.txt').exists():
        # Default: read from spy_matrices.txt in scripts directory
        default_file = Path(__file__).parent / 'spy_matrices.txt'
        print(f"Using default matrix list: {default_file}")
        with open(default_file, 'r') as f:
            selected_matrices = [line.strip() for line in f
                                if line.strip() and not line.startswith('#')]
    elif args.use_curated:
        # Use curated list, filter to those available in analysis
        if not df.empty:
            available = set(df['matrix'].unique())
            selected_matrices = [m for m in CURATED_MATRICES if m in available]
            if not selected_matrices:
                print("Warning: No curated matrices found in analysis data.")
                print("Available matrices sample:", list(available)[:10])
                sys.exit(1)
        else:
            selected_matrices = CURATED_MATRICES
    else:
        # Auto-select diverse matrices from analysis data
        if not df.empty:
            orig = df[df['perm'] == 'None'].copy()
            orig = orig[(orig['rows'] >= 1000) & (orig['rows'] <= args.max_size)]
            
            # Select matrices that have permutation data
            matrix_counts = df.groupby('matrix')['perm'].nunique()
            good_matrices = matrix_counts[matrix_counts >= 5].index
            orig = orig[orig['matrix'].isin(good_matrices)]
            
            # Try to select diverse matrices:
            # - Different sizes
            # - Different densities
            # - Different application domains (from different families)
            if len(orig) > 10:
                # Bin by size and density, sample from each
                orig['size_bin'] = pd.cut(orig['rows'], bins=5, labels=False)
                orig['density_bin'] = pd.cut(orig['density'], bins=5, labels=False)
                
                np.random.seed(42)
                # Stratified sample
                sampled = orig.groupby(['size_bin', 'density_bin'], observed=True).apply(
                    lambda x: x.sample(min(1, len(x)), random_state=42)
                ).reset_index(drop=True)
                
                if len(sampled) < 10:
                    # Fill with random if stratified didn't give enough
                    remaining = orig[~orig['matrix'].isin(sampled['matrix'])]
                    extra = remaining.sample(min(10 - len(sampled), len(remaining)), random_state=42)
                    sampled = pd.concat([sampled, extra])
                
                orig = sampled.head(10)
            
            selected_matrices = orig['matrix'].tolist()
        else:
            print("No analysis data and no matrices specified. Please use --matrices.")
            sys.exit(1)
    
    print(f"Selected matrices: {selected_matrices}")
    
    # Process each matrix
    for matrix_name in selected_matrices:
        print(f"\n{'='*60}")
        print(f"Processing: {matrix_name}")
        print('='*60)
        
        # Find matrix file
        matrix_path = None
        for pattern in [
            matrices_dir / "**" / matrix_name,
            matrices_dir / matrix_name,
        ]:
            matches = list(matrices_dir.glob(f"**/{matrix_name}"))
            if matches:
                matrix_path = matches[0]
                break
        
        if matrix_path is None:
            print(f"  Warning: Matrix file not found for {matrix_name}")
            continue
        
        print(f"  Loading matrix from: {matrix_path}")
        
        try:
            A_original = mmread(matrix_path).tocsr()
        except Exception as e:
            print(f"  Error loading matrix: {e}")
            continue
        
        print(f"  Shape: {A_original.shape}, NNZ: {A_original.nnz:,}")
        
        # Skip if too large
        if max(A_original.shape) > args.max_size:
            print(f"  Skipping: matrix too large (>{args.max_size})")
            continue
        
        # Collect reordered versions (keyed by display name)
        matrices_dict = {"Original": A_original}

        for algo in algorithms:
            perm_file = find_permutation_file(perms_dir, matrix_name, algo)

            if perm_file is None:
                # Try with algorithm name variations
                algo_variations = [
                    algo,
                    algo.upper(),
                    algo.lower(),
                    f"{algo}_reorder",
                ]
                for var in algo_variations:
                    perm_file = find_permutation_file(perms_dir, matrix_name, var)
                    if perm_file:
                        break

            if perm_file is None:
                print(f"  No permutation found for algorithm: {algo}")
                continue

            print(f"  Found permutation: {perm_file}")

            # Get permutation type
            perm_type = get_perm_type_for_algorithm(algo, df, matrix_name)
            print(f"    Perm type: {perm_type}")

            try:
                row_perm, col_perm = load_permutation_file(perm_file, perm_type)
                A_reordered = apply_permutation(A_original.copy(), row_perm, col_perm)
                display_name = get_perm_display(algo)
                matrices_dict[display_name] = A_reordered
                print(f"    Successfully applied {algo} ({display_name}) permutation")
            except Exception as e:
                print(f"    Error applying permutation: {e}")
        
        if len(matrices_dict) <= 1:
            print("  No permutations found, skipping spy plot")
            continue
        
        # Create spy plot
        matrix_base = Path(matrix_name).stem
        output_path = output_dir / f"{matrix_base}_spy.png"
        
        print(f"  Creating spy plot with {len(matrices_dict)} panels...")
        create_spy_plot(
            matrices_dict, 
            output_path, 
            matrix_name,
            markersize=args.markersize
        )
        print(f"  Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"All spy plots saved to: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
