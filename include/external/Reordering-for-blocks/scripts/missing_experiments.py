"""
Missing Experiments Report

Walk through the experiment pipeline and show what's missing at each stage:

1. Analysis: the base set of matrices that have been analyzed.
2. Filtered: matrices that survive filter_config.yaml filters.
3. Operations: for each kernel (algo), which filtered matrices are missing
   — overall and per reordering.
"""

import argparse
import re
import sys
from collections import defaultdict

import pandas as pd

from plot_utils import load_filter_config, apply_filters
from settings import KERNEL_NAMES, get_perm_display


# ── helpers ─────────────────────────────────────────────────────────────────

def _kernel_id(algo: str, block_size) -> str:
    """Strip reordering suffixes from algo to get the kernel identity."""
    algo = re.sub(r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)', '', algo)
    if pd.notna(block_size) and block_size > 0:
        return f"{algo}_bs{int(block_size)}"
    return algo


def _display_kernel(kid: str) -> str:
    return KERNEL_NAMES.get(kid, kid)


def _display_perm(perm: str) -> str:
    if perm == 'None':
        return 'None (original)'
    return get_perm_display(perm)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--filter-config', default=None,
                    help='Path to filter_config.yaml')
    ap.add_argument('--no-filter', action='store_true',
                    help='Skip filtering (use all matrices)')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='List individual missing matrices')
    args = ap.parse_args()

    # ── load config & paths ─────────────────────────────────────────────
    cfg = load_filter_config(args.filter_config)
    data = cfg.get('data', {})
    ops_csv = data.get('operations_csv', 'results/results_operations.csv')
    ana_csv = data.get('analysis_csv', 'results/results_analysis.csv')
    mat_list = data.get('matrices_list', 'datasets/matrices_list_mtx.txt')

    # ── load CSVs ───────────────────────────────────────────────────────
    try:
        df_op = pd.read_csv(ops_csv)
        df_an = pd.read_csv(ana_csv)
    except Exception as e:
        print(f"Error reading CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    for df in (df_op, df_an):
        df['matrix'] = df['matrix'].astype(str)
        df['perm'] = df['perm'].fillna('None').astype(str)
        df['perm_type'] = df['perm_type'].fillna('UNKNOWN').astype(str)

    # ── Stage 1: Analysis (base set) ───────────────────────────────────
    analysis_matrices = sorted(df_an['matrix'].unique())
    print('=' * 65)
    print('STAGE 1 — Analysis (base set of matrices)')
    print('=' * 65)
    print(f'  Matrices with analysis data: {len(analysis_matrices)}')

    # ── Stage 2: After filters ─────────────────────────────────────────
    if args.no_filter:
        filtered_matrices = set(analysis_matrices)
        print('\n  (filtering skipped)')
    else:
        filt = cfg.get('filters', {})
        _, df_an_f = apply_filters(
            df_op, df_an,
            matrices_list_path=mat_list,
            one_per_family=filt.get('one_per_family', True),
            square_only=filt.get('square_only', True),
            min_size=filt.get('min_size'),
            max_size=filt.get('max_size'),
            min_bandwidth=filt.get('min_bandwidth'),
            max_sparsity_factor=filt.get('max_sparsity_factor'),
            filter_diagonal=filt.get('filter_diagonal', True),
        )
        filtered_matrices = set(df_an_f['matrix'].unique())

    dropped = set(analysis_matrices) - filtered_matrices
    print(f'\n  Matrices after filtering: {len(filtered_matrices)}')
    print(f'  Dropped by filters:      {len(dropped)}')

    print('\n' + '=' * 65)
    print('STAGE 2 — These are the matrices we care about')
    print('=' * 65)
    print(f'  Target matrix count: {len(filtered_matrices)}')

    # ── Stage 3: Operations coverage ───────────────────────────────────
    # restrict operations to only the filtered matrices
    df_op_f = df_op[df_op['matrix'].isin(filtered_matrices)].copy()

    # derive kernel_id
    if 'algo' in df_op_f.columns:
        df_op_f['kernel_id'] = df_op_f.apply(
            lambda r: _kernel_id(r['algo'], r.get('block_size')), axis=1)
    else:
        df_op_f['kernel_id'] = 'UNKNOWN'

    kernels = sorted(df_op_f['kernel_id'].unique())

    # all (perm, perm_type) combos seen across data
    all_reorderings = sorted(
        df_op_f[['perm', 'perm_type']].drop_duplicates().itertuples(index=False, name=None))

    print('\n' + '=' * 65)
    print('STAGE 3 — Operations: missing matrices per kernel')
    print('=' * 65)
    print(f'  Kernels found:      {len(kernels)}')
    print(f'  Reorderings found:  {len(all_reorderings)}')
    print(f'  n_cols values:      '
          f'{sorted(df_op_f["n_cols"].dropna().unique().astype(int)) if "n_cols" in df_op_f.columns else "N/A"}')

    # Include n_cols in the grouping if present
    has_ncols = 'n_cols' in df_op_f.columns
    if has_ncols:
        ncols_values = sorted(df_op_f['n_cols'].dropna().unique().astype(int))
    else:
        ncols_values = [None]

    for n_cols in ncols_values:
        if has_ncols and len(ncols_values) > 1:
            print(f'\n  ── n_cols = {n_cols} ──')

        if has_ncols and n_cols is not None:
            df_slice = df_op_f[df_op_f['n_cols'] == n_cols]
        else:
            df_slice = df_op_f

        for kid in kernels:
            df_k = df_slice[df_slice['kernel_id'] == kid]
            present = set(df_k['matrix'].unique())
            missing_any = filtered_matrices - present

            display = _display_kernel(kid)
            print(f'\n  Kernel: {display} ({kid})')
            print(f'    Present: {len(present)} / {len(filtered_matrices)}'
                  f'   Missing (any reordering): {len(missing_any)}')

            if args.verbose and missing_any:
                for m in sorted(missing_any):
                    print(f'      - {m}')

            # per-reordering breakdown
            for perm, perm_type in all_reorderings:
                df_kr = df_k[(df_k['perm'] == perm) & (df_k['perm_type'] == perm_type)]
                present_r = set(df_kr['matrix'].unique())
                missing_r = filtered_matrices - present_r
                label = _display_perm(perm)
                print(f'    {label:>20s} ({perm_type:>9s}):  '
                      f'present {len(present_r):>4d}  '
                      f'missing {len(missing_r):>4d}')
                if args.verbose and missing_r:
                    for m in sorted(missing_r):
                        print(f'        - {m}')

    # ── Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)
    print(f'  Analysis matrices:           {len(analysis_matrices)}')
    print(f'  After filters (target set):  {len(filtered_matrices)}')
    total_expected = len(filtered_matrices) * len(kernels) * len(all_reorderings) * len(ncols_values)
    total_present = len(df_op_f)
    print(f'  Expected operation rows:     {total_expected}  '
          f'(= {len(filtered_matrices)} matrices × {len(kernels)} kernels '
          f'× {len(all_reorderings)} reorderings × {len(ncols_values)} n_cols)')
    print(f'  Actual operation rows:       {total_present}')
    print(f'  Missing operation rows:      {max(0, total_expected - total_present)}')


if __name__ == '__main__':
    main()
