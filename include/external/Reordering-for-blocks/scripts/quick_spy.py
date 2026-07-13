#!/usr/bin/env python3
"""
Quick on-demand spy plots: one figure per (matrix, algorithm) pair.

Usage examples:
    python3 scripts/quick_spy.py bcsstk10 random rcm groot
    python3 scripts/quick_spy.py --markersize 0.3 cage12 amd metis sparta
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).parent))
from settings import PERMS, get_perm_display, get_perm_color
from plot_utils import set_professional_style
from spy_plots import load_permutation_file, apply_permutation, find_permutation_file

# Build a lookup: lowercase display name / alias -> perm key
_ALIAS = {}
for key, info in PERMS.items():
    _ALIAS[info['display'].lower()] = key
    _ALIAS[key.lower()] = key
# Common shorthand aliases
_ALIAS['random'] = 'random1D'
_ALIAS['groot'] = 'GROOT_reorder'
_ALIAS['sparta'] = 'SPARTA_reorder'
_ALIAS['dtc-lsh'] = 'TCA_reorder'
_ALIAS['dtclsh'] = 'TCA_reorder'
_ALIAS['tca'] = 'TCA_reorder'
_ALIAS['slashburn'] = 'SB_slashburn'
_ALIAS['rcm'] = 'SB_rcm'
_ALIAS['amd'] = 'SB_amd'
_ALIAS['metis'] = 'SB_metis'
_ALIAS['gray'] = 'SB_gray'
_ALIAS['degree'] = 'SB_degree'
_ALIAS['rabbit'] = 'SB_rabbit'
_ALIAS['patoh'] = 'SB_patoh'

DATASETS_DIR = Path(__file__).resolve().parent.parent / 'datasets'
PERMS_DIR = Path(__file__).resolve().parent.parent / 'perms'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'plots' / 'spy_quick'
ANALYSIS_CSV = Path(__file__).resolve().parent.parent / 'results' / 'results_analysis.csv'


def resolve_algorithm(name):
    """Resolve a user-friendly algorithm name to the internal perm key."""
    hit = _ALIAS.get(name.lower())
    if hit:
        return hit
    # Fuzzy: check if name is a substring of any key
    for alias, key in _ALIAS.items():
        if name.lower() in alias:
            return key
    return None


def find_matrix(name):
    """Find a .mtx file matching *name* under datasets/."""
    stem = Path(name).stem
    matches = list(DATASETS_DIR.glob(f'**/{stem}.mtx'))
    return matches[0] if matches else None


def get_perm_type(algorithm, matrix_name):
    """Look up perm_type from analysis CSV, default to SYMMETRIC.

    Prefers SYMMETRIC when both ROW and SYMMETRIC entries exist.
    """
    try:
        import pandas as pd
        df = pd.read_csv(ANALYSIS_CSV)
        rows = df[(df['perm'] == algorithm) & (df['matrix'] == matrix_name)]
        if not rows.empty:
            perm_types = set(rows['perm_type'].unique())
            if 'SYMMETRIC' in perm_types:
                return 'SYMMETRIC'
            return rows.iloc[0]['perm_type']
    except Exception:
        pass
    return 'SYMMETRIC'


def make_spy(A, title, color, output_path, markersize):
    """Save a single spy plot using imshow for pixel-accurate rendering."""
    from matplotlib.colors import to_rgb
    n_rows, n_cols = A.shape
    # Render into a binary image, then colorize
    img_size = 1500  # pixels for the image
    # Map matrix coords to pixel coords
    A_coo = A.tocoo()
    pr = (A_coo.row * (img_size - 1) / max(n_rows - 1, 1)).astype(int)
    pc = (A_coo.col * (img_size - 1) / max(n_cols - 1, 1)).astype(int)
    # White background, colored nonzeros
    rgb = to_rgb(color)
    img = np.ones((img_size, img_size, 3), dtype=np.float32)
    img[pr, pc] = rgb

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, interpolation='nearest', aspect='equal',
              extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])
    ax.set_title(title, fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel(f'{n_cols:,} cols', fontsize=10)
    ax.set_ylabel(f'{n_rows:,} rows', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Quick on-demand spy plots.')
    parser.add_argument('matrix', help='Matrix name (e.g. bcsstk10)')
    parser.add_argument('algorithms', nargs='+',
                        help='Algorithms (e.g. random rcm groot)')
    parser.add_argument('--markersize', type=float, default=0.05)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--original', action='store_true', default=True,
                        help='Also generate a spy plot of the original matrix (default: yes)')
    parser.add_argument('--no-original', action='store_false', dest='original')
    args = parser.parse_args()

    set_professional_style()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_stem = Path(args.matrix).stem
    mtx_name = f'{matrix_stem}.mtx'

    # Find and load matrix
    mtx_path = find_matrix(args.matrix)
    if mtx_path is None:
        print(f'Error: matrix "{args.matrix}" not found under {DATASETS_DIR}')
        sys.exit(1)

    print(f'Loading {mtx_path} ...')
    A = mmread(mtx_path).tocsr()
    print(f'  Shape: {A.shape}, NNZ: {A.nnz:,}')

    # Original
    if args.original:
        out_path = out_dir / f'{matrix_stem}_original.png'
        make_spy(A, f'{matrix_stem} — Original', '#888888', out_path,
                 args.markersize)

    # Each algorithm
    for algo_input in args.algorithms:
        algo_key = resolve_algorithm(algo_input)
        if algo_key is None:
            print(f'  Warning: unknown algorithm "{algo_input}", skipping')
            continue

        perm_file = find_permutation_file(PERMS_DIR, mtx_name, algo_key)
        if perm_file is None:
            print(f'  Warning: no permutation file for {algo_key} / {mtx_name}')
            continue

        perm_type = get_perm_type(algo_key, mtx_name)
        try:
            row_perm, col_perm = load_permutation_file(perm_file, perm_type)
            A_perm = apply_permutation(A.copy(), row_perm, col_perm)
        except Exception as e:
            print(f'  Error applying {algo_key}: {e}')
            continue

        display = get_perm_display(algo_key)
        color = get_perm_color(algo_key)
        out_path = out_dir / f'{matrix_stem}_{algo_key}.png'
        make_spy(A_perm, f'{matrix_stem} — {display}', color, out_path,
                 args.markersize)

    print('Done.')


if __name__ == '__main__':
    main()
