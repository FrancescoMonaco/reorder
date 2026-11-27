"""
Plot Utilities - Clean Architecture (Tufte-styled)

This module provides:
1. Data loading and processing functions that prepare DataFrames with all derived columns
2. Generic Tufte-styled plot functions parameterized by column names
3. Metric configuration for display names and properties
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, FuncFormatter
from pathlib import Path
from scipy import stats
import re
import sys
import yaml
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from settings import (
    PALETTE, PERMS, ALL_METRICS,
    get_metric_display, get_perm_display, get_perm_color,
    use_log_scale,
)


# =============================================================================
# Tufte / Publication Plot Style
# =============================================================================

# Muted, harmonious semantic colors (replacing harsh primary red/green)
_COLOR_GOOD = '#2e7d5b'      # muted forest teal
_COLOR_BAD = '#b84a45'       # muted crimson / terracotta
_COLOR_NEUTRAL = '#a8a8a8'   # light neutral grey
_COLOR_BAR_GOOD = '#5a9e78'  # soft sage for stacked bars
_COLOR_BAR_BAD = '#c96a63'   # soft coral for stacked bars


def set_professional_style():
    """Configure matplotlib/seaborn for clean, Tufte-inspired publication plots.

    Call once at the start of a script (also used as ProcessPoolExecutor
    initializer so parallel workers inherit the exact same style).
    """
    sns.set_theme(style='ticks', font='serif', palette=PALETTE)
    mpl.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titlepad': 10,
        'axes.labelsize': 12,
        'axes.labelcolor': '#222222',
        'axes.titlecolor': '#111111',
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10,
        'legend.frameon': False,
        # Lines & markers
        'lines.linewidth': 1.5,
        'lines.markersize': 4.5,
        # Axes & spines (Tufte: minimal ink)
        'axes.linewidth': 0.7,
        'axes.edgecolor': '#444444',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.color': '#888888',
        'grid.alpha': 0.2,
        'grid.linewidth': 0.5,
        # Figure
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.06,
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '#444444',
        'ytick.color': '#444444',
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })


def apply_tufte_style(ax, grid_axis='y', offset_spines=False, keep_left=True):
    """Strip chartjunk from an Axes: despine, soften ticks, and place subtle gridlines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(keep_left)

    for side in ('bottom', 'left'):
        if ax.spines[side].get_visible():
            ax.spines[side].set_color('#444444')
            ax.spines[side].set_linewidth(0.7)
            if offset_spines:
                ax.spines[side].set_position(('outward', 5))

    ax.tick_params(axis='both', which='major', direction='out', length=3.5,
                   width=0.7, colors='#333333')
    ax.tick_params(axis='both', which='minor', direction='out', length=2.0,
                   width=0.5, colors='#666666')
    if not keep_left:
        ax.tick_params(axis='y', length=0)

    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, which='major',
                color='#888888', linestyle='-', linewidth=0.5, alpha=0.2)


def _log_scalar_formatter(x, pos):
    """Format log-scale tick values cleanly (0.1, 1, 10, 1k, 1M)."""
    if x == 0:
        return '0'
    if x >= 1e6:
        val = x / 1e6
        return f'{int(val)}M' if val == int(val) else f'{val:g}M'
    if x >= 1e4:
        val = x / 1e3
        return f'{int(val)}k' if val == int(val) else f'{val:g}k'
    if x >= 1 and x == int(x):
        return f'{int(x)}'
    return f'{x:g}'


def _format_log10_axis(ax, log_lo, log_hi, suffix=''):
    """Format a log10-transformed y-axis with intuitive linear tick labels."""
    p_lo = int(np.floor(log_lo))
    p_hi = int(np.ceil(log_hi))
    span = p_hi - p_lo
    if span <= 2:
        mults = [1.0, 2.0, 5.0]
    elif span <= 4:
        mults = [1.0, 2.0] if span <= 3 else [1.0]
    else:
        mults = [1.0]

    ticks = []
    for p in range(p_lo - 1, p_hi + 2):
        for m in mults:
            val = m * (10.0 ** p)
            lv = np.log10(val)
            if log_lo - 1e-9 <= lv <= log_hi + 1e-9:
                ticks.append(val)

    if len(ticks) < 3:
        ticks = [10.0 ** p for p in range(p_lo, p_hi + 1)]

    ax.set_yticks([np.log10(t) for t in ticks])
    ax.set_yticklabels([f'{_log_scalar_formatter(t, None)}{suffix}' for t in ticks])


def _style_violin_bodies(ax, alpha=0.72):
    """Soften violin fills and add crisp white edges."""
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(alpha)
            coll.set_edgecolor('white')
            coll.set_linewidth(0.6)


def format_log_axes(ax, which='both', dense=True):
    """Format log-scale axes with clean scalar ticks and subtle Tufte gridlines.

    Args:
        ax: Matplotlib Axes object.
        which: ``'x'``, ``'y'``, or ``'both'`` — which axes to format.
        dense: If True (default), label at 1, 2, 5 per decade.
               If False, label only at powers of 10.
    """
    targets = []
    if which in ('x', 'both') and ax.get_xscale() == 'log':
        targets.append(ax.xaxis)
    if which in ('y', 'both') and ax.get_yscale() == 'log':
        targets.append(ax.yaxis)

    scalar_fmt = FuncFormatter(_log_scalar_formatter)
    major_subs = [1.0, 2.0, 5.0] if dense else [1.0]

    for axis in targets:
        axis.set_major_locator(LogLocator(base=10, subs=major_subs, numticks=30))
        axis.set_major_formatter(scalar_fmt)
        axis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
        axis.set_minor_formatter(NullFormatter())

    ax.tick_params(which='major', length=3.8, width=0.7)
    ax.tick_params(which='minor', length=2.0, width=0.45)

    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#888888', alpha=0.22, linewidth=0.55)
    ax.grid(True, which='minor', color='#888888', alpha=0.06, linewidth=0.25)


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_data(ops_path, analysis_path):
    """Load operations and analysis CSVs and merge them."""
    try:
        df_analysis = pd.read_csv(analysis_path)
    except Exception as e:
        print(f"Error reading analysis CSV: {e}")
        sys.exit(1)

    if df_analysis.empty:
        print("Analysis CSV is empty.")
        sys.exit(1)

    try:
        df_op = pd.read_csv(ops_path)
    except FileNotFoundError:
        print(f"Operations CSV not found at {ops_path} — "
              "kernel/breakeven plots will be skipped.")
        df_op = pd.DataFrame()
    except Exception as e:
        print(f"Error reading operations CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df_op)} operation rows and {len(df_analysis)} analysis rows.")

    df_analysis['perm'] = df_analysis['perm'].fillna('None').astype(str)
    df_analysis['perm_type'] = df_analysis['perm_type'].fillna('UNKNOWN').astype(str)
    df_analysis['matrix'] = df_analysis['matrix'].astype(str)

    if df_op.empty:
        return pd.DataFrame(), df_analysis

    df_op['perm'] = df_op['perm'].fillna('None').astype(str)
    df_op['perm_type'] = df_op['perm_type'].fillna('UNKNOWN').astype(str)
    df_op['matrix'] = df_op['matrix'].astype(str)

    df = pd.merge(df_op, df_analysis, on=['matrix', 'perm', 'perm_type'], how='left')
    print(f"Merged DataFrame has {len(df)} rows.")

    if df.empty:
        print("Merged DataFrame is empty! Check if keys match in both CSVs.")

    return df, df_analysis


def add_base_metrics(df):
    """Add base derived metrics: gflops, kernel_id, strategy."""
    df = df.copy()

    if 'n_cols' in df.columns:
        df['n_cols'] = pd.to_numeric(df['n_cols'], errors='coerce').fillna(32)
    else:
        df['n_cols'] = 32

    df['gflops'] = (2 * df['nnz'] * df['n_cols']) / (df['time_operation_ms'] * 1e-3) / 1e9

    def get_kernel_id(row):
        algo = row['algo']
        algo = re.sub(r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)', '', algo)
        if pd.notnull(row.get('block_size')) and row['block_size'] > 0:
            return f"{algo}_bs{int(row['block_size'])}"
        return algo

    df['kernel_id'] = df.apply(get_kernel_id, axis=1)
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )
    return df


def add_relative_metrics(df):
    """Add relative/normalized metrics."""
    df = df.copy()

    if 'bandwidth_max' in df.columns and 'rows' in df.columns:
        df['rel_bandwidth'] = df['bandwidth_max'] / df['rows']
    if 'locality_avg_row_spread' in df.columns and 'cols' in df.columns:
        df['rel_row_spread'] = df['locality_avg_row_spread'] / df['cols']
    if 'access_dist_reuse_distance_mean' in df.columns and 'cols' in df.columns:
        df['rel_reuse_distance'] = df['access_dist_reuse_distance_mean'] / df['cols']
    if 'access_dist_index_distance_mean' in df.columns and 'cols' in df.columns:
        df['rel_index_distance'] = df['access_dist_index_distance_mean'] / df['cols']

    return df


def add_speedup(df):
    """Add speedup column relative to Original (perm='None') baseline."""
    df = df.copy()
    original = df[df['strategy'] == 'Original'].groupby(['matrix', 'kernel_id', 'n_cols'])['gflops'].mean()
    original = original.reset_index().rename(columns={'gflops': 'gflops_original'})
    df = pd.merge(df, original, on=['matrix', 'kernel_id', 'n_cols'], how='left')
    df['speedup'] = df['gflops'] / df['gflops_original']
    return df


def build_metrics_config(df, include_extended=False):
    """Build a metrics config dict for improvement calculation."""
    metrics_config = {
        'bandwidth_max': {'improvement_name': 'bandwidth_improvement', 'higher_is_better': False},
        'bandwidth_avg': {'improvement_name': 'bandwidth_avg_improvement', 'higher_is_better': False},
        'locality_avg_row_spread': {'improvement_name': 'row_spread_improvement', 'higher_is_better': False},
        'locality_avg_col_spread': {'improvement_name': 'col_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
        'locality_profile': {'improvement_name': 'profile_improvement', 'higher_is_better': False},
        'access_dist_reuse_distance_mean': {'improvement_name': 'reuse_distance_improvement', 'higher_is_better': False},
        'access_dist_reuse_distance_median': {'improvement_name': 'reuse_distance_median_improvement', 'higher_is_better': False},
        'access_dist_index_distance_mean': {'improvement_name': 'index_distance_improvement', 'higher_is_better': False},
        'access_dist_index_distance_median': {'improvement_name': 'index_distance_median_improvement', 'higher_is_better': False},
    }

    if include_extended:
        metrics_config.update({
            'locality_max_row_spread': {'improvement_name': 'max_row_spread_improvement', 'higher_is_better': False},
            'locality_max_col_spread': {'improvement_name': 'max_col_spread_improvement', 'higher_is_better': False},
        })

    for col in df.columns:
        if col.startswith('block_density_'):
            bs = col.split('_')[-1]
            metrics_config[col] = {'improvement_name': f'density_improvement_{bs}', 'higher_is_better': True}

    for bs in [4, 8, 16, 32, 64, 128]:
        col = f'avg_blocks_per_row_{bs}'
        if col in df.columns:
            metrics_config[col] = {
                'improvement_name': f'avg_blocks_per_row_improvement_{bs}',
                'higher_is_better': False,
            }

    if include_extended:
        for bs in [4, 8, 16, 32, 64, 128]:
            col = f'max_blocks_per_row_{bs}'
            if col in df.columns:
                metrics_config[col] = {
                    'improvement_name': f'max_blocks_per_row_improvement_{bs}',
                    'higher_is_better': False,
                }

    return metrics_config


def compute_improvements(df, metrics_config):
    """Compute improvement columns from a metrics config."""
    df = df.copy()
    available_metrics = [m for m in metrics_config if m in df.columns]
    if not available_metrics:
        return df

    original = df[df['strategy'] == 'Original'][['matrix'] + available_metrics].drop_duplicates()
    original = original.groupby('matrix')[available_metrics].mean().reset_index()
    original = original.rename(columns={m: f'{m}_original' for m in available_metrics})

    df = pd.merge(df, original, on='matrix', how='left')

    for metric, config in metrics_config.items():
        if metric not in available_metrics:
            continue
        orig_col = f'{metric}_original'
        imp_col = config['improvement_name']
        if config['higher_is_better']:
            df[imp_col] = df[metric] / df[orig_col].replace(0, np.nan)
        else:
            df[imp_col] = df[orig_col] / df[metric].replace(0, np.nan)

    return df


def add_improvement_columns(df):
    """Add improvement columns comparing reordered to original."""
    metrics_config = build_metrics_config(df)
    return compute_improvements(df, metrics_config)


def add_size_class(df):
    """Add size_class column based on nnz."""
    df = df.copy()
    if 'nnz' not in df.columns:
        return df
    bins = [0, 5e4, 2e5, 1e6, np.inf]
    labels = ['<50K', '50K-200K', '200K-1M', '>1M']
    df['size_class'] = pd.cut(df['nnz'], bins=bins, labels=labels)
    return df


def prepare_full_dataframe(df):
    """Apply all transformations to prepare a fully processed DataFrame."""
    df = add_base_metrics(df)
    df = add_relative_metrics(df)
    df = add_speedup(df)
    df = add_improvement_columns(df)
    df = add_size_class(df)
    return df


# =============================================================================
# Filtering Functions
# =============================================================================

def load_matrix_family_map(matrices_list_path):
    """Load mapping from matrix name to family/group."""
    matrix_to_family = {}
    if not Path(matrices_list_path).exists():
        print(f"Warning: Matrices list file {matrices_list_path} not found.")
        return matrix_to_family

    with open(matrices_list_path, 'r') as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            path = path.replace('\\', '/')
            parts = path.split('/')
            if len(parts) >= 3:
                matrix_name = parts[-1]
                family = parts[-3]
                matrix_to_family[matrix_name] = family

    return matrix_to_family


def filter_one_per_family(df, matrices_list_path, keep_full_families=None):
    """Filter DataFrame to keep only one matrix per family."""
    if keep_full_families is None:
        keep_full_families = load_filter_config().get('filters', {}).get('keep_full_families', [])

    matrix_to_family = load_matrix_family_map(matrices_list_path)
    if not matrix_to_family:
        print("Warning: No matrix mapping found. Skipping family filtering.")
        return df

    df = df.copy()
    df['family'] = df['matrix'].map(matrix_to_family)

    unknown_mask = df['family'].isna()
    if unknown_mask.any():
        n_unknown = df.loc[unknown_mask, 'matrix'].nunique()
        df = df[~unknown_mask]
        print(f"  Dropped {n_unknown} matrices not in {matrices_list_path}")

    keep_full_mask = df['family'].isin(keep_full_families)
    df_keep_full = df[keep_full_mask]
    df_to_filter = df[~keep_full_mask]

    unique_matrices = df_to_filter[['matrix', 'family']].drop_duplicates().sort_values('matrix')
    selected = unique_matrices.groupby('family').first()['matrix'].tolist()
    all_selected = list(set(selected + df_keep_full['matrix'].unique().tolist()))

    result = df[df['matrix'].isin(all_selected)]
    print(f"Family filter: {len(df)} -> {len(result)} rows ({len(all_selected)} matrices)")
    return result


def filter_trivial_matrices(df, df_analysis, bandwidth_threshold=5):
    """Filter out matrices with original bandwidth below threshold."""
    if 'bandwidth_max' not in df_analysis.columns:
        return df
    trivial = df_analysis[
        (df_analysis['perm'] == 'None') &
        (df_analysis['bandwidth_max'] < bandwidth_threshold)
    ]['matrix'].unique()
    if len(trivial) > 0:
        print(f"Filtering {len(trivial)} trivial matrices (bandwidth < {bandwidth_threshold})")
        df = df[~df['matrix'].isin(trivial)]
    return df


def filter_sparse_matrices(df, df_analysis, nnz_factor=2):
    """Filter out very sparse matrices (nnz < factor * rows)."""
    if 'nnz' not in df_analysis.columns or 'rows' not in df_analysis.columns:
        return df
    sparse = df_analysis[
        (df_analysis['perm'] == 'None') &
        (df_analysis['nnz'] < nnz_factor * df_analysis['rows'])
    ]['matrix'].unique()
    if len(sparse) > 0:
        print(f"Filtering {len(sparse)} very sparse matrices (nnz < {nnz_factor}*N)")
        df = df[~df['matrix'].isin(sparse)]
    return df


def filter_diagonal_matrices(df, df_analysis):
    """Filter out purely diagonal matrices (nnz == rows for square matrices)."""
    if 'nnz' not in df_analysis.columns or 'rows' not in df_analysis.columns:
        return df
    diagonal = df_analysis[
        (df_analysis['perm'] == 'None') &
        (df_analysis['rows'] == df_analysis['cols']) &
        (df_analysis['nnz'] == df_analysis['rows'])
    ]['matrix'].unique()
    if len(diagonal) > 0:
        print(f"Filtering {len(diagonal)} purely diagonal matrices")
        df = df[~df['matrix'].isin(diagonal)]
    return df


def filter_square_only(df):
    """Keep only square matrices."""
    if 'rows' in df.columns and 'cols' in df.columns:
        result = df[df['rows'] == df['cols']]
        print(f"Square filter: {len(df)} -> {len(result)} rows")
        return result
    return df


def filter_min_size(df, min_rows):
    """Keep only matrices with at least min_rows rows."""
    if 'rows' in df.columns:
        result = df[df['rows'] >= min_rows]
        print(f"Min size filter (>={min_rows}): {len(df)} -> {len(result)} rows")
        return result
    return df


def filter_max_size(df, max_dim):
    """Remove matrices where rows or cols exceed *max_dim*."""
    if 'rows' in df.columns and 'cols' in df.columns:
        result = df[(df['rows'] <= max_dim) & (df['cols'] <= max_dim)]
        print(f"Max size filter (<={max_dim}): {len(df)} -> {len(result)} rows")
        return result
    return df


def _filter_matrices_from_both(df, df_analysis, matrices_to_remove, reason):
    """Remove matrices from both DataFrames and print a message."""
    if len(matrices_to_remove) > 0:
        print(f"Filtering {len(matrices_to_remove)} {reason}")
        if not df.empty:
            df = df[~df['matrix'].isin(matrices_to_remove)]
        df_analysis = df_analysis[~df_analysis['matrix'].isin(matrices_to_remove)]
    return df, df_analysis


# =============================================================================
# Filter Config Loading
# =============================================================================

_DEFAULT_FILTER_CONFIG = Path(__file__).resolve().parent / 'filter_config.yaml'


def load_filter_config(config_path=None):
    """Load the centralized filter configuration from a YAML file."""
    if config_path is None:
        config_path = _DEFAULT_FILTER_CONFIG
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: filter config {config_path} not found. Using built-in defaults.",
              file=sys.stderr)
        return {
            'data': {
                'operations_csv': 'results/results_operations.csv',
                'analysis_csv': 'results/results_analysis.csv',
                'matrices_list': 'datasets/matrices_list_mtx.txt',
            },
            'filters': {
                'one_per_family': True,
                'square_only': True,
                'min_size': None,
                'max_size': None,
                'min_bandwidth': None,
                'max_sparsity_factor': 2,
                'filter_diagonal': True,
                'keep_full_families': [],
            }
        }

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_filters(df, df_analysis, matrices_list_path=None,
                  one_per_family=True, square_only=True,
                  min_size=None, max_size=None, min_bandwidth=None,
                  max_sparsity_factor=None, filter_diagonal=True,
                  keep_full_families=None):
    """Apply all configured filters to both DataFrames."""
    ops_empty = df.empty

    if one_per_family and matrices_list_path:
        if not ops_empty:
            df = filter_one_per_family(df, matrices_list_path, keep_full_families)
        df_analysis = filter_one_per_family(df_analysis, matrices_list_path, keep_full_families)

    if min_bandwidth is not None and 'bandwidth_max' in df_analysis.columns:
        trivial_matrices = df_analysis[
            (df_analysis['perm'] == 'None') &
            (df_analysis['bandwidth_max'] < min_bandwidth)
        ]['matrix'].unique()
        df, df_analysis = _filter_matrices_from_both(
            df, df_analysis, trivial_matrices,
            f"trivial matrices (bandwidth < {min_bandwidth})")

    if max_sparsity_factor is not None and 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
        sparse_matrices = df_analysis[
            (df_analysis['perm'] == 'None') &
            (df_analysis['nnz'] < max_sparsity_factor * df_analysis['rows'])
        ]['matrix'].unique()
        df, df_analysis = _filter_matrices_from_both(
            df, df_analysis, sparse_matrices,
            f"very sparse matrices (nnz < {max_sparsity_factor}*rows)")

    if filter_diagonal and 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
        diagonal_matrices = df_analysis[
            (df_analysis['perm'] == 'None') &
            (df_analysis['rows'] == df_analysis['cols']) &
            (df_analysis['nnz'] == df_analysis['rows'])
        ]['matrix'].unique()
        df, df_analysis = _filter_matrices_from_both(
            df, df_analysis, diagonal_matrices, "purely diagonal matrices")

    if square_only:
        if not ops_empty:
            df = filter_square_only(df)
        df_analysis = filter_square_only(df_analysis)

    if min_size is not None:
        if not ops_empty:
            df = filter_min_size(df, min_size)
        df_analysis = filter_min_size(df_analysis, min_size)

    if max_size is not None:
        if not ops_empty:
            df = filter_max_size(df, max_size)
        df_analysis = filter_max_size(df_analysis, max_size)

    return df, df_analysis


def _apply_exclude_perms(df, df_analysis, exclude_perms):
    """Drop rows whose perm matches the exclude_perms list."""
    if not exclude_perms:
        return df, df_analysis
    if isinstance(exclude_perms, dict):
        excluded = set()
        for v in exclude_perms.values():
            excluded.update(v)
    else:
        excluded = set(exclude_perms)
    n0 = len(df) + len(df_analysis)
    if not df.empty:
        df = df[~df['perm'].isin(excluded)]
    df_analysis = df_analysis[~df_analysis['perm'].isin(excluded)]
    n1 = len(df) + len(df_analysis)
    print(f"  exclude_perms: removed {n0 - n1} rows for {excluded}")
    return df, df_analysis


def split_by_perm_type(df, df_analysis, perm_type, pipeline_cfg=None):
    """Filter to a single perm_type, duplicating Original rows."""
    def _filter(frame):
        if frame.empty:
            return frame
        original = frame[frame['perm'] == 'None'].copy()
        reordered = frame[(frame['perm'] != 'None')
                          & (frame['perm_type'] == perm_type)]
        original['perm_type'] = perm_type
        return pd.concat([original, reordered], ignore_index=True)

    df_out = _filter(df)
    df_analysis_out = _filter(df_analysis)

    if pipeline_cfg:
        exclude_perms = set(pipeline_cfg.get('exclude_perms', []))
        if exclude_perms:
            n0 = len(df_out) + len(df_analysis_out)
            if not df_out.empty:
                df_out = df_out[~df_out['perm'].isin(exclude_perms)]
            df_analysis_out = df_analysis_out[
                ~df_analysis_out['perm'].isin(exclude_perms)]
            n1 = len(df_out) + len(df_analysis_out)
            print(f"  pipeline exclude_perms: removed {n0 - n1} rows "
                  f"for {exclude_perms}")
        exclude_kernels = set(pipeline_cfg.get('exclude_kernels', []))
        if exclude_kernels and not df_out.empty:
            if 'kernel_id' in df_out.columns:
                df_out = df_out[~df_out['kernel_id'].isin(exclude_kernels)]
            elif 'algo' in df_out.columns:
                _strip = lambda a: re.sub(
                    r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)', '', a)
                mask = df_out['algo'].apply(_strip).isin(exclude_kernels)
                df_out = df_out[~mask]

    print(f"  After split_by_perm_type({perm_type}): "
          f"{len(df_out)} ops rows, {len(df_analysis_out)} analysis rows")
    return df_out, df_analysis_out


def load_and_filter_data(config_path=None, cli_overrides=None):
    """Single entry-point: load CSVs, apply every filter from config."""
    cfg = load_filter_config(config_path)
    if cli_overrides and cli_overrides.get('random'):
        data_cfg = dict(cfg.get('data_random', cfg.get('data', {})))
    else:
        data_cfg = dict(cfg.get('data', {}))
    filt_cfg = dict(cfg.get('filters', {}))

    if cli_overrides:
        for key in ('operations_csv', 'analysis_csv', 'matrices_list'):
            if cli_overrides.get(key) is not None:
                data_cfg[key] = cli_overrides[key]
        for key in filt_cfg:
            if key in cli_overrides and cli_overrides[key] is not None:
                filt_cfg[key] = cli_overrides[key]

    print("Loading data...")
    df, df_analysis = load_data(
        data_cfg.get('operations_csv', 'results/results_operations.csv'),
        data_cfg.get('analysis_csv', 'results/results_analysis.csv'),
    )

    print("\nApplying filters (from filter_config.yaml)...")
    df, df_analysis = apply_filters(
        df, df_analysis,
        matrices_list_path=data_cfg.get('matrices_list',
                                        'datasets/matrices_list_mtx.txt'),
        one_per_family=filt_cfg.get('one_per_family', True),
        square_only=filt_cfg.get('square_only', True),
        min_size=filt_cfg.get('min_size'),
        max_size=filt_cfg.get('max_size'),
        min_bandwidth=filt_cfg.get('min_bandwidth'),
        max_sparsity_factor=filt_cfg.get('max_sparsity_factor'),
        filter_diagonal=filt_cfg.get('filter_diagonal', True),
        keep_full_families=filt_cfg.get('keep_full_families', []),
    )

    exclude_perms = filt_cfg.get('exclude_perms', None)
    if exclude_perms:
        df, df_analysis = _apply_exclude_perms(df, df_analysis, exclude_perms)

    print(f"After filtering: {len(df)} operation rows, "
          f"{len(df_analysis)} analysis rows")

    return df, df_analysis, cfg


# =============================================================================
# Correlation Method Configuration
# =============================================================================

_VALID_CORR_METHODS = {'pearson', 'spearman', 'kendall'}
_correlation_method_cache = None


def get_correlation_method(cfg=None):
    """Return the configured correlation method (pearson/spearman/kendall)."""
    global _correlation_method_cache
    if _correlation_method_cache is not None:
        return _correlation_method_cache

    if cfg is None:
        cfg = load_filter_config()
    method = cfg.get('display', {}).get('correlation_method', 'pearson').lower()
    if method not in _VALID_CORR_METHODS:
        print(f"Warning: unknown correlation_method '{method}', falling back to 'pearson'",
              file=sys.stderr)
        method = 'pearson'
    _correlation_method_cache = method
    return method


def compute_correlation(x, y, method=None):
    """Compute correlation between *x* and *y* using the configured method."""
    if method is None:
        method = get_correlation_method()
    if method == 'pearson':
        return stats.pearsonr(x, y)
    elif method == 'spearman':
        return stats.spearmanr(x, y)
    elif method == 'kendall':
        return stats.kendalltau(x, y)
    raise ValueError(f"Unknown correlation method: {method}")


def correlation_display_symbol(method=None, log_x=False, log_y=False):
    """Return a short display symbol for the configured correlation method."""
    if method is None:
        method = get_correlation_method()
    base = {'pearson': 'r', 'spearman': r'\rho', 'kendall': r'\tau'}[method]
    if log_x and log_y:
        return base + r'_{\log}'
    if log_x:
        return base + r'_{\log x}'
    if log_y:
        return base + r'_{\log y}'
    return base


def correlation_display_name(method=None):
    """Return a human-readable name for the configured correlation method."""
    if method is None:
        method = get_correlation_method()
    return {
        'pearson': "Pearson's $r$",
        'spearman': "Spearman's $\\rho$",
        'kendall': "Kendall's $\\tau$",
    }[method]


# =============================================================================
# Generic Plot Functions (Tufte-styled)
# =============================================================================

def _setup_figure(figsize=(10, 6.5)):
    """Create figure with standard settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_figure(path, dpi=300):
    """Save figure and close."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {Path(path).name}")


def _exec_plot_task(task):
    """Execute a single (func, kwargs) plot task."""
    func, kwargs = task
    try:
        func(**kwargs)
    except Exception as e:
        print(f"  Error in {func.__name__}: {e}")


def parallel_execute(tasks, n_jobs=None):
    """Execute plot tasks in parallel using multiprocessing."""
    if not tasks:
        return
    if n_jobs is None:
        n_jobs = min(multiprocessing.cpu_count(), 8)
    if n_jobs <= 1 or len(tasks) <= 2:
        for task in tasks:
            _exec_plot_task(task)
        return
    print(f"  Running {len(tasks)} plots across {n_jobs} workers...")
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=set_professional_style) as pool:
        futs = [pool.submit(_exec_plot_task, t) for t in tasks]
        for f in as_completed(futs):
            if f.exception():
                print(f"  Error: {f.exception()}")


def scatter_with_correlation(df, x_col, y_col, output_path,
                             title=None, hue_col=None,
                             log_x=None, log_y=None,
                             show_correlation=True,
                             figsize=(9.5, 6.5)):
    """Create Tufte-styled scatter plot with correlation annotation."""
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)

    method = get_correlation_method()
    sym = correlation_display_symbol(method, log_x=log_x, log_y=log_y)
    corr_val = _correlation_for_scatter(plot_df[x_col], plot_df[y_col], method,
                                        log_x=log_x, log_y=log_y)

    fig, ax = _setup_figure(figsize)

    if hue_col and hue_col in plot_df.columns:
        pal = get_strategy_palette() if hue_col == 'strategy' else "Set2"
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col,
                        alpha=0.68, s=26, edgecolor='white', linewidth=0.35,
                        ax=ax, palette=pal)
        ax.legend(title=None, frameon=False, loc='best')
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col,
                        alpha=0.68, s=26, edgecolor='white', linewidth=0.35,
                        ax=ax)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    apply_tufte_style(ax, grid_axis='both', offset_spines=True)
    if log_x or log_y:
        format_log_axes(ax)

    ax.set_xlabel(get_metric_display(x_col), labelpad=8)
    ax.set_ylabel(get_metric_display(y_col), labelpad=8)

    if title is None:
        title = f"{get_metric_display(y_col)} vs {get_metric_display(x_col)}"
    if show_correlation:
        title += f"   (${sym} = {corr_val:.3f}$)"
    ax.set_title(title)

    _save_figure(output_path)


def boxplot_by_category(df, x_col, y_col, output_path,
                        title=None, order=None,
                        baseline=None, show_points=False,
                        clip_percentile=None,
                        log_y=False,
                        ylim=None,
                        palette=None,
                        figsize=(11, 4.8)):
    """Create a Tufte-styled violin plot (log-space KDE when log_y=True)
    with embedded median/IQR box.

    Signature kept identical to the original boxplot_by_category for
    drop-in compatibility.
    """
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return

    plot_df = plot_df[np.isfinite(plot_df[y_col])]
    if log_y:
        plot_df = plot_df[plot_df[y_col] > 0]
        if plot_df.empty:
            print(f"Skipping {output_path}: no positive data for log scale")
            return

    if clip_percentile:
        lower = plot_df[y_col].quantile(clip_percentile[0] / 100)
        upper = plot_df[y_col].quantile(clip_percentile[1] / 100)
        plot_df = plot_df[(plot_df[y_col] >= lower) & (plot_df[y_col] <= upper)]

    if order is not None:
        order = [c for c in order if c in plot_df[x_col].unique()]
    if palette is None and x_col == 'strategy':
        palette = get_strategy_palette(order)

    fig, ax = _setup_figure(figsize)

    y_plot_col = y_col
    if log_y:
        y_plot_col = '_log10_y'
        plot_df[y_plot_col] = np.log10(plot_df[y_col])

    sns.violinplot(
        data=plot_df, x=x_col, y=y_plot_col, order=order,
        palette=palette, cut=0, inner='box', linewidth=0.75,
        saturation=0.85, ax=ax,
    )
    _style_violin_bodies(ax, alpha=0.72)

    if show_points:
        sns.stripplot(
            data=plot_df, x=x_col, y=y_plot_col, order=order,
            color='#222222', alpha=0.28, jitter=0.2, size=2.4, ax=ax,
        )

    if baseline is not None:
        b_val = np.log10(baseline) if (log_y and baseline > 0) else baseline
        ax.axhline(b_val, color='#333333', linestyle='--', linewidth=0.85,
                   alpha=0.65, zorder=2)

    if log_y:
        if ylim is not None:
            y_lo, y_hi = np.log10(ylim[0]), np.log10(ylim[1])
        else:
            p01 = plot_df[y_plot_col].quantile(0.01)
            p99 = plot_df[y_plot_col].quantile(0.99)
            pad = max(0.1, (p99 - p01) * 0.08)
            y_lo, y_hi = p01 - pad, p99 + pad
        ax.set_ylim(y_lo, y_hi)
        _format_log10_axis(ax, y_lo, y_hi)
    elif ylim is not None:
        ax.set_ylim(ylim)

    apply_tufte_style(ax, grid_axis='y', offset_spines=False)

    ax.set_xlabel('' if x_col == 'strategy' else get_metric_display(x_col), labelpad=8)
    ylabel = get_metric_display(y_col)
    if log_y:
        ylabel += ' (log scale)'
    ax.set_ylabel(ylabel, labelpad=8)

    if title is not None:
        if clip_percentile:
            title += f" ({clip_percentile[0]}–{clip_percentile[1]} pct)"
        ax.set_title(title)

    plt.xticks(rotation=30, ha='right')
    _save_figure(output_path)


def breakeven_boxplot(df_valid, df_harmful, x_col, y_col, output_path,
                      title=None, order=None,
                      cap_value=None,
                      clip_percentile=None,
                      palette=None,
                      figsize=(11, 7.6)):
    """Create Tufte-styled breakeven plot with two aligned subplots.

    **Top subplot**: log-space violin plot of break-even iteration counts
    (valid data only), revealing full distribution shape + embedded median/IQR.

    **Bottom subplot**: clean stacked percentage bars (muted sage beneficial
    vs muted terracotta harmful) with direct percentage labels.
    """
    plot_df = df_valid.dropna(subset=[x_col, y_col]).copy()
    plot_df = plot_df[np.isfinite(plot_df[y_col]) & (plot_df[y_col] > 0)]

    has_valid = not plot_df.empty
    has_harmful = df_harmful is not None and not df_harmful.empty

    if not has_valid and not has_harmful:
        print(f"Skipping {output_path}: no data")
        return

    if has_valid and clip_percentile:
        lower = plot_df[y_col].quantile(clip_percentile[0] / 100)
        upper = plot_df[y_col].quantile(clip_percentile[1] / 100)
        plot_df = plot_df[(plot_df[y_col] >= lower) & (plot_df[y_col] <= upper)]

    if palette is None and x_col == 'strategy':
        palette = get_strategy_palette(order)

    if order is not None:
        cats = list(order)
    elif has_valid:
        cats = sorted(plot_df[x_col].unique())
    elif has_harmful:
        cats = sorted(df_harmful[x_col].unique())
    else:
        cats = []
    positions = list(range(len(cats)))

    valid_pcts, harm_pcts = [], []
    for cat in cats:
        n_v = int((plot_df[x_col] == cat).sum()) if has_valid else 0
        n_h = int((df_harmful[x_col] == cat).sum()) if has_harmful else 0
        tot = n_v + n_h
        valid_pcts.append(100.0 * n_v / tot if tot > 0 else 100.0)
        harm_pcts.append(100.0 * n_h / tot if tot > 0 else 0.0)

    fig, (ax_top, ax_bar) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [3.2, 1], 'hspace': 0.12},
    )

    # ===== Top subplot: log10-space violin plot =====
    if has_valid:
        plot_df['_log_be'] = np.log10(plot_df[y_col])
        sns.violinplot(
            data=plot_df, x=x_col, y='_log_be', order=cats,
            palette=palette, cut=0, inner='box', linewidth=0.75,
            saturation=0.85, ax=ax_top,
        )
        _style_violin_bodies(ax_top, alpha=0.72)

        y_lo = plot_df['_log_be'].min() - 0.2
        y_hi = plot_df['_log_be'].max() + 0.2
        ax_top.set_ylim(y_lo, y_hi)
        _format_log10_axis(ax_top, y_lo, y_hi)

    apply_tufte_style(ax_top, grid_axis='y', offset_spines=False)
    ax_top.tick_params(axis='x', length=0)
    ax_top.set_xlabel('')
    ax_top.set_ylabel('Operations to break even (log scale)', labelpad=8)

    if title is None:
        title = f"Break-even Operations by {get_metric_display(x_col)}"
    ax_top.set_title(title, pad=12)

    # ===== Bottom subplot: muted stacked success / failure bars =====
    bar_w = 0.56
    ax_bar.bar(positions, harm_pcts, width=bar_w,
               color=_COLOR_BAR_BAD, edgecolor='white', linewidth=0.6,
               label='Harmful (never breaks even)')
    ax_bar.bar(positions, valid_pcts, width=bar_w, bottom=harm_pcts,
               color=_COLOR_BAR_GOOD, edgecolor='white', linewidth=0.6,
               label='Beneficial')

    for i, (vp, hp) in enumerate(zip(valid_pcts, harm_pcts)):
        if hp >= 12:
            ax_bar.text(i, hp / 2, f"{hp:.0f}%", ha='center', va='center',
                        fontsize=8.5, fontweight='semibold', color='white')
        if vp >= 12:
            ax_bar.text(i, hp + vp / 2, f"{vp:.0f}%", ha='center', va='center',
                        fontsize=8.5, fontweight='semibold', color='white')

    ax_bar.set_ylim(0, 100)
    ax_bar.set_yticks([0, 50, 100])
    ax_bar.set_yticklabels(['0%', '50%', '100%'])
    ax_bar.set_ylabel('Outcome', labelpad=8)
    ax_bar.set_xlabel('' if x_col == 'strategy' else get_metric_display(x_col), labelpad=8)
    apply_tufte_style(ax_bar, grid_axis=None, keep_left=False)
    ax_bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.24),
                  fontsize=8.5, ncol=2, frameon=False)

    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(cats, rotation=30, ha='right')

    _save_figure(output_path)


def binned_bar_chart(df, bin_col, value_col, output_path,
                     bins=None, labels=None,
                     title=None, baseline=None,
                     agg_func='median',
                     min_count=5,
                     figsize=(9.5, 5.2)):
    """Create Tufte-styled bar chart showing aggregated values per bin."""
    if bins is None:
        bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1000.0]
    if labels is None:
        labels = ['<0.5x', '0.5-1x', '1-1.5x', '1.5-2x', '2-3x', '3-5x', '>5x']

    plot_df = df.dropna(subset=[bin_col, value_col]).copy()
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return

    plot_df['bin'] = pd.cut(plot_df[bin_col], bins=bins, labels=labels)
    stats_df = plot_df.groupby('bin', observed=True)[value_col].agg([agg_func, 'count']).reset_index()
    stats_df = stats_df[stats_df['count'] >= min_count]

    if stats_df.empty:
        print(f"Skipping {output_path}: insufficient data per bin")
        return

    fig, ax = _setup_figure(figsize)
    sns.barplot(data=stats_df, x='bin', y=agg_func, palette="viridis",
                edgecolor='white', linewidth=0.6, saturation=0.82, ax=ax)

    for i, p in enumerate(ax.patches):
        if i < len(stats_df):
            ax.annotate(f"n={int(stats_df.iloc[i]['count'])}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=8.5, color='#555555',
                        xytext=(0, 4), textcoords='offset points')

    if baseline is not None:
        ax.axhline(baseline, color='#333333', linestyle='--', linewidth=0.85, alpha=0.65)

    apply_tufte_style(ax, grid_axis='y', keep_left=False)
    ax.set_xlabel(f"{get_metric_display(bin_col)} Bin", labelpad=8)
    ax.set_ylabel(f"{agg_func.title()} {get_metric_display(value_col)}", labelpad=8)

    if title is None:
        title = f"{agg_func.title()} {get_metric_display(value_col)} by {get_metric_display(bin_col)}"
    ax.set_title(title)

    _save_figure(output_path)


def binned_boxplot(df, bin_col, value_col, output_path,
                   bins=None, labels=None,
                   title=None, baseline=None,
                   min_count=5,
                   show_points=False,
                   figsize=(9.5, 4.2)):
    """Create Tufte-styled violin plot showing distribution of values per bin."""
    if bins is None:
        bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1000.0]
    if labels is None:
        labels = ['<0.5x', '0.5-1x', '1-1.5x', '1.5-2x', '2-3x', '3-5x', '>5x']

    plot_df = df.dropna(subset=[bin_col, value_col]).copy()
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return

    plot_df['bin'] = pd.cut(plot_df[bin_col], bins=bins, labels=labels)
    bin_counts = plot_df['bin'].value_counts()
    valid_bins = bin_counts[bin_counts >= min_count].index
    plot_df = plot_df[plot_df['bin'].isin(valid_bins)]

    if plot_df.empty:
        print(f"Skipping {output_path}: insufficient data per bin")
        return

    bin_order = [label for label in labels if label in valid_bins]
    fig, ax = _setup_figure(figsize)

    sns.violinplot(data=plot_df, x='bin', y=value_col, order=bin_order,
                   palette="viridis", cut=0, inner='box', linewidth=0.75,
                   saturation=0.82, ax=ax)
    _style_violin_bodies(ax, alpha=0.72)

    if show_points:
        sns.stripplot(data=plot_df, x='bin', y=value_col, order=bin_order,
                      color='#222222', alpha=0.28, jitter=0.2, size=2.4, ax=ax)

    y_span = plot_df[value_col].max() - plot_df[value_col].min()
    y_offset = 0.03 * y_span if y_span > 0 else 0.05
    for i, label in enumerate(bin_order):
        count = bin_counts[label]
        y_pos = plot_df[plot_df['bin'] == label][value_col].max() + y_offset
        ax.text(i, y_pos, f"n={int(count)}", ha='center', va='bottom',
                fontsize=8.5, color='#555555')

    if baseline is not None:
        ax.axhline(baseline, color='#333333', linestyle='--', linewidth=0.85, alpha=0.65)

    apply_tufte_style(ax, grid_axis='y', offset_spines=False)
    ax.set_xlabel(f"{get_metric_display(bin_col)} Bin", labelpad=8)
    ax.set_ylabel(f"{get_metric_display(value_col)}", labelpad=8)

    if title is None:
        title = f"{get_metric_display(value_col)} Distribution by {get_metric_display(bin_col)}"
    ax.set_title(title)

    plt.xticks(rotation=30, ha='right')
    _save_figure(output_path)


def cdf_plot(df, value_col, output_path,
             hue_col=None, hue_order=None,
             title=None, baseline=None,
             log_x=False,
             figsize=(9.0, 5.2)):
    """Create Tufte-styled CDF plot."""
    plot_df = df.dropna(subset=[value_col])
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return

    fig, ax = _setup_figure(figsize)

    if hue_col and hue_col in plot_df.columns:
        categories = hue_order if hue_order else sorted(plot_df[hue_col].unique())
        strategy_pal = get_strategy_palette(categories) if hue_col == 'strategy' else {}
        for cat in categories:
            subset = plot_df[plot_df[hue_col] == cat]
            values = subset[value_col].sort_values()
            cdf_y = np.arange(1, len(values) + 1) / len(values)
            color = strategy_pal.get(cat)
            ax.step(values, cdf_y, label=cat, where='post', linewidth=1.6,
                    **({"color": color} if color else {}))
        ax.legend(title=None, frameon=False, loc='lower right')
    else:
        values = plot_df[value_col].sort_values()
        cdf_y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, cdf_y, where='post', linewidth=1.6)

    if baseline is not None:
        ax.axvline(baseline, color='#555555', linestyle='--', linewidth=0.85, alpha=0.65)

    if log_x:
        ax.set_xscale('log')

    apply_tufte_style(ax, grid_axis='both', offset_spines=True)
    if log_x:
        format_log_axes(ax, which='x')

    ax.set_ylim(0, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(get_metric_display(value_col), labelpad=8)
    ax.set_ylabel('Cumulative fraction', labelpad=8)

    if title is None:
        title = f"CDF of {get_metric_display(value_col)}"
    ax.set_title(title)

    _save_figure(output_path)


def violin_plot(df, x_col, y_col, output_path,
                title=None, order=None,
                show_points=True,
                figsize=(11, 5.5)):
    """Create Tufte-styled violin plot with optional point overlay."""
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return

    fig, ax = _setup_figure(figsize)
    pal = get_strategy_palette(order) if x_col == 'strategy' else "Set2"

    sns.violinplot(data=plot_df, x=x_col, y=y_col, order=order,
                   palette=pal, inner="box", cut=0, linewidth=0.75,
                   saturation=0.85, ax=ax)
    _style_violin_bodies(ax, alpha=0.72)

    if show_points:
        sns.stripplot(data=plot_df, x=x_col, y=y_col, order=order,
                      color='#222222', alpha=0.25, jitter=0.2, size=2.4, ax=ax)

    apply_tufte_style(ax, grid_axis='y', offset_spines=False)
    ax.set_xlabel('' if x_col == 'strategy' else get_metric_display(x_col), labelpad=8)
    ax.set_ylabel(get_metric_display(y_col), labelpad=8)

    if title is None:
        title = f"{get_metric_display(y_col)} by {get_metric_display(x_col)}"
    ax.set_title(title)

    plt.xticks(rotation=30, ha='right')
    _save_figure(output_path)


def correlation_heatmap(df, cols, output_path,
                        title=None, method=None,
                        figsize=(9.5, 7.5)):
    """Create clean correlation heatmap for selected columns."""
    if method is None:
        method = get_correlation_method()
    available_cols = [c for c in cols if c in df.columns]
    if len(available_cols) < 2:
        print(f"Skipping {output_path}: need at least 2 columns")
        return

    corr = df[available_cols].corr(method=method)
    fig, ax = _setup_figure(figsize)

    hm = sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
        center=0, vmin=-1, vmax=1, ax=ax, square=True,
        linewidths=0.9, linecolor='white',
        cbar_kws={'shrink': 0.8, 'pad': 0.03},
        annot_kws={'size': 8.5},
    )
    cbar = hm.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    if title is None:
        title = f"Correlation Matrix ({method.title()})"
    ax.set_title(title, pad=12)

    _save_figure(output_path)


def pairwise_heatmap(win_frac_df, output_path, title=None, figsize=(9.5, 8.2)):
    """Create a clean pairwise win-fraction heatmap for reordering algorithms."""
    fig, ax = _setup_figure(figsize)

    hm = sns.heatmap(
        win_frac_df, annot=True, fmt='.0%',
        cmap='RdYlBu', center=0.5, vmin=0, vmax=1,
        ax=ax, square=True, linewidths=0.9, linecolor='white',
        cbar_kws={'label': 'Win fraction (row beats column)', 'shrink': 0.8, 'pad': 0.03},
        annot_kws={'size': 8.5},
    )
    cbar = hm.collections[0].colorbar
    if cbar is not None:
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    if title:
        ax.set_title(title, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    _save_figure(output_path)


# =============================================================================
# Profile Plot Utilities
# =============================================================================

def profile_perm_order(perm_values):
    """Canonical perm order for profile plots — Original drawn last (on top)."""
    seen = set()
    order = [p for p in list(PERMS)
             if p != 'None' and p in perm_values
             and not (p in seen or seen.add(p))]
    if 'None' in perm_values:
        order.append('None')
    return order


def draw_profile_curve(ax, taus_asc, n_matrices, perm, xlim_lo, xlim_hi,
                       higher_is_better):
    """Draw one perm's Dolan-Moré profile curve on *ax*."""
    n = len(taus_asc)
    color = get_perm_color(perm)
    label = get_perm_display(perm)
    ls = '--' if perm == 'None' else '-'
    lw = 1.9 if perm == 'None' else 1.45
    alpha = 0.85 if perm == 'None' else 0.92
    if perm == 'None':
        color = '#222222'

    if higher_is_better:
        taus = taus_asc[::-1]
        fracs = np.arange(1, n + 1) / n_matrices
        n_best = (taus >= 1.0 - 1e-9).sum()
        y_at_1 = n_best / n_matrices
        taus_rest = taus[n_best:]
        fracs_rest = fracs[n_best:]
        if len(taus_rest) > 0:
            ax.step(taus_rest, fracs_rest, where='pre',
                    label=label, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            ax.plot([xlim_hi, taus_rest[0]], [y_at_1, y_at_1],
                    color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            ax.plot([taus_rest[-1], xlim_lo], [fracs_rest[-1], fracs_rest[-1]],
                    color=color, linewidth=lw, linestyle=ls, alpha=alpha)
        else:
            ax.plot([xlim_hi, xlim_lo], [y_at_1, y_at_1],
                    label=label, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
    else:
        fracs = np.arange(1, n + 1) / n_matrices
        n_best = (taus_asc <= 1.0 + 1e-9).sum()
        y_at_1 = n_best / n_matrices
        taus_rest = taus_asc[n_best:]
        fracs_rest = fracs[n_best:]
        if len(taus_rest) > 0:
            ax.step(taus_rest, fracs_rest, where='pre',
                    label=label, color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            ax.plot([xlim_lo, taus_rest[0]], [y_at_1, y_at_1],
                    color=color, linewidth=lw, linestyle=ls, alpha=alpha)
            ax.plot([taus_rest[-1], xlim_hi], [fracs_rest[-1], fracs_rest[-1]],
                    color=color, linewidth=lw, linestyle=ls, alpha=alpha)
        else:
            ax.plot([xlim_lo, xlim_hi], [y_at_1, y_at_1],
                    label=label, color=color, linewidth=lw, linestyle=ls, alpha=alpha)


# =============================================================================
# Partial Correlation Utilities
# =============================================================================

def compute_partial_imp_matrix(df, n_cols, imp_metrics, kernel, method=None):
    """Compute partial correlation matrix for one kernel."""
    if method is None:
        method = get_correlation_method()

    df_k = df[(df['n_cols'] == n_cols)
              & (df['strategy'] != 'Original')
              & (df['kernel_id'] == kernel)]

    n = len(imp_metrics)
    mat = pd.DataFrame(np.full((n, n), np.nan),
                       index=imp_metrics, columns=imp_metrics)

    for i, m_target in enumerate(imp_metrics):
        if m_target not in df_k.columns:
            continue

        for j, m_control in enumerate(imp_metrics):
            if i == j or m_control not in df_k.columns:
                continue

            cols = [m_target, m_control, 'speedup']
            sub = df_k[cols].dropna()
            sub = sub[np.isfinite(sub).all(axis=1)]
            if len(sub) < 10:
                continue

            y = sub['speedup'].values
            x = sub[m_target].values
            z = sub[m_control].values
            z_aug = np.column_stack([z, np.ones(len(z))])
            try:
                coef_y = np.linalg.lstsq(z_aug, y, rcond=None)[0]
                coef_x = np.linalg.lstsq(z_aug, x, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            ry = y - z_aug @ coef_y
            rx = x - z_aug @ coef_x

            if np.std(ry) < 1e-12 or np.std(rx) < 1e-12:
                continue

            r, _ = compute_correlation(pd.Series(rx), pd.Series(ry), method)
            mat.iloc[i, j] = r

    return mat


# =============================================================================
# Utility Functions
# =============================================================================

def get_strategy_order(df):
    """Get standard order for strategies (Original first, then canonical)."""
    strategies = set(df['strategy'].unique())
    canonical = [p['display'] for p in PERMS.values() if p['display'] in strategies]
    extra = sorted(strategies - set(canonical) - {'Original'})
    result = canonical + extra
    if 'Original' in strategies:
        result = ['Original'] + result
    return result


def get_strategy_palette(strategies=None):
    """Return a {strategy_display_name: color} dict for seaborn palette kwarg."""
    base = {p['display']: p['color'] for p in PERMS.values()}
    base['Original'] = '#888888'
    if strategies is not None:
        return {s: base.get(s, '#333333') for s in strategies}
    return base


def get_density_columns(df):
    """Get list of block density columns (only block_density_N format)."""
    return [c for c in df.columns if c.startswith('block_density_') and c.split('_')[-1].isdigit()]


def get_improvement_columns(df):
    """Get list of improvement columns."""
    return [c for c in df.columns if c.endswith('_improvement')]


def safe_filename(name):
    """Convert string to safe filename."""
    return name.replace('/', '_').replace(' ', '_').replace('\\', '_')


# =============================================================================
# Publication-Ready Scatter Plots (Tufte-styled)
# =============================================================================

def _correlation_for_scatter(x_vals, y_vals, method=None,
                             log_x=False, log_y=False):
    """Compute correlation in linear or log10 space."""
    valid = x_vals.notna() & y_vals.notna() & np.isfinite(x_vals) & np.isfinite(y_vals)
    xp, yp = x_vals[valid], y_vals[valid]
    if log_x:
        pos = xp > 0
        xp, yp = xp[pos], yp[pos]
        xp = np.log10(xp)
    if log_y:
        pos = yp > 0
        xp, yp = xp[pos], yp[pos]
        yp = np.log10(yp)
    if len(xp) >= 2:
        r, _ = compute_correlation(xp, yp, method=method)
        return r
    return np.nan


def _loglog_slope(x_vals, y_vals, log_x=False, log_y=False, min_pts=5):
    """Return the slope of a linear fit in (optionally log-transformed) space."""
    xp = np.asarray(x_vals, dtype=float)
    yp = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(xp) & np.isfinite(yp)
    if log_x:
        mask &= xp > 0
    if log_y:
        mask &= yp > 0
    xp, yp = xp[mask], yp[mask]
    if log_x:
        xp = np.log10(xp)
    if log_y:
        yp = np.log10(yp)
    if len(xp) < min_pts:
        return np.nan
    return np.polyfit(xp, yp, 1)[0]


def scatter_publication(df, x_col, y_col, output_path,
                        hue_col=None, log_x=None, log_y=None,
                        show_correlation=True, label=None,
                        ylim=None, figsize=(3.6, 3.0)):
    """Publication-ready Tufte scatter plot sized for multi-panel layouts."""
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)

    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col in plot_df.columns:
        pal = get_strategy_palette() if hue_col == 'strategy' else "Set2"
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col,
                        alpha=0.65, ax=ax, palette=pal, s=14, edgecolor='none')
        ax.legend(title=None, frameon=False, fontsize=8)
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col,
                        alpha=0.65, ax=ax, s=14, edgecolor='none')

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    apply_tufte_style(ax, grid_axis='both', offset_spines=True)
    if log_x or log_y:
        format_log_axes(ax)

    ax.set_xlabel(get_metric_display(x_col), fontsize=11)
    ax.set_ylabel(get_metric_display(y_col), fontsize=11)
    ax.tick_params(axis='both', labelsize=9.5, which='major')

    if show_correlation:
        method = get_correlation_method()
        sym = correlation_display_symbol(method, log_x=log_x, log_y=log_y)
        corr_val = _correlation_for_scatter(plot_df[x_col], plot_df[y_col], method,
                                            log_x=log_x, log_y=log_y)
        slope = _loglog_slope(plot_df[x_col], plot_df[y_col],
                              log_x=log_x, log_y=log_y)
        ann = f"${sym}={corr_val:.2f}$"
        if np.isfinite(slope) and (log_x or log_y):
            ann += f"\n$\\alpha={slope:.2f}$"
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=9,
                color='#222222', va='top', ha='left')

    if label:
        ax.text(0.97, 0.03, label, transform=ax.transAxes, fontsize=9,
                ha='right', va='bottom', fontweight='semibold', color='#222222')

    if ylim is not None:
        ax.set_ylim(ylim)

    _save_figure(output_path)


def scatter_presentation(df, x_col, y_col, output_path,
                         log_x=None, log_y=None,
                         show_correlation=True, label=None,
                         baseline_y=1.0, baseline_x=None, trend_line=True,
                         ylim=None, xlim=None, quadrant_colors=False,
                         figsize=(7.0, 5.0)):
    """Presentation-ready Tufte scatter plot with muted quadrant coloring
    and optional trend line."""
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)

    fig, ax = plt.subplots(figsize=figsize)

    if quadrant_colors and baseline_x is not None and baseline_y is not None:
        xv = plot_df[x_col].values
        yv = plot_df[y_col].values
        colors = np.where(
            (xv >= baseline_x) & (yv >= baseline_y), _COLOR_GOOD,
            np.where(
                (xv < baseline_x) & (yv < baseline_y), _COLOR_BAD,
                _COLOR_NEUTRAL))
        ax.scatter(xv, yv, alpha=0.6, s=26, c=colors,
                   edgecolor='white', linewidth=0.35, zorder=3)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col],
                   alpha=0.6, s=26, color=PALETTE[0],
                   edgecolor='white', linewidth=0.35, zorder=3)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if baseline_y is not None:
        ax.axhline(baseline_y, color='#555555', linestyle='--',
                   linewidth=0.85, alpha=0.6, zorder=2)
    if baseline_x is not None:
        ax.axvline(baseline_x, color='#555555', linestyle='--',
                   linewidth=0.85, alpha=0.6, zorder=2)

    if trend_line and len(plot_df) >= 5:
        xv = plot_df[x_col].values.astype(float)
        yv = plot_df[y_col].values.astype(float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if log_x and log_y:
            pos = mask & (xv > 0) & (yv > 0)
            xfit, yfit = np.log10(xv[pos]), np.log10(yv[pos])
        elif log_x:
            pos = mask & (xv > 0)
            xfit, yfit = np.log10(xv[pos]), yv[pos]
        elif log_y:
            pos = mask & (yv > 0)
            xfit, yfit = xv[pos], np.log10(yv[pos])
        else:
            xfit, yfit = xv[mask], yv[mask]
        if len(xfit) >= 5:
            z = np.polyfit(xfit, yfit, 1)
            xs = np.linspace(xfit.min(), xfit.max(), 200)
            ys = np.polyval(z, xs)
            if log_x:
                xs = 10**xs
            if log_y:
                ys = 10**ys
            ax.plot(xs, ys, color='#333333', linewidth=1.2,
                    linestyle='-', alpha=0.5, zorder=2)

    apply_tufte_style(ax, grid_axis='both', offset_spines=True)
    if log_x or log_y:
        format_log_axes(ax)

    ax.set_xlabel(get_metric_display(x_col), fontsize=14, labelpad=8)
    ax.set_ylabel(get_metric_display(y_col), fontsize=14, labelpad=8)
    ax.tick_params(axis='both', labelsize=11.5, which='major')

    if show_correlation:
        method = get_correlation_method()
        sym = correlation_display_symbol(method, log_x=log_x, log_y=log_y)
        corr_val = _correlation_for_scatter(plot_df[x_col], plot_df[y_col], method,
                                            log_x=log_x, log_y=log_y)
        slope = _loglog_slope(plot_df[x_col], plot_df[y_col],
                              log_x=log_x, log_y=log_y)
        ann = f"${sym}={corr_val:.2f}$"
        if np.isfinite(slope) and (log_x or log_y):
            ann += f"\n$\\alpha={slope:.2f}$"
        ax.text(0.03, 0.97, ann, transform=ax.transAxes, fontsize=12,
                color='#222222', va='top', ha='left')

    if label:
        ax.text(0.97, 0.03, label, transform=ax.transAxes, fontsize=12,
                ha='right', va='bottom', fontstyle='italic', color='#333333')

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {Path(output_path).name}")


def grouped_scatter_publication(df, x_col, y_col, group_col, group_order,
                                output_path, group_labels=None,
                                hue_col=None, log_x=None, log_y=None,
                                show_correlation=True, ylim=None, xlim=None,
                                baseline_x=None, baseline_y=None,
                                quadrant_colors=False,
                                mark_failures=False,
                                figsize=(7.4, 4.8)):
    """2x3 grouped Tufte scatter with shared axes. One subplot per group."""
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)
    if group_labels is None:
        group_labels = {}

    nrows, ncols = 2, 3
    n_groups = min(len(group_order), nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    for idx in range(n_groups):
        group = group_order[idx]
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        df_g = plot_df[plot_df[group_col] == group]

        if len(df_g) < 2:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', fontsize=8, color='#777777')
        else:
            if quadrant_colors and baseline_x is not None and baseline_y is not None:
                xv = df_g[x_col].values
                yv = df_g[y_col].values
                colors = np.where(
                    (xv >= baseline_x) & (yv >= baseline_y), _COLOR_GOOD,
                    np.where(
                        (xv < baseline_x) & (yv < baseline_y), _COLOR_BAD,
                        _COLOR_NEUTRAL))
                ax.scatter(xv, yv, alpha=0.62, s=11, c=colors,
                           edgecolor='none', zorder=3)
            elif hue_col and hue_col in df_g.columns:
                pal = get_strategy_palette() if hue_col == 'strategy' else "Set2"
                sns.scatterplot(data=df_g, x=x_col, y=y_col, hue=hue_col,
                                alpha=0.62, ax=ax, palette=pal, s=11,
                                edgecolor='none', legend=(idx == 0))
            else:
                sns.scatterplot(data=df_g, x=x_col, y=y_col,
                                alpha=0.62, ax=ax, s=11, edgecolor='none')

            if show_correlation:
                method = get_correlation_method()
                sym = correlation_display_symbol(method, log_x=log_x, log_y=log_y)
                cr = _correlation_for_scatter(df_g[x_col], df_g[y_col], method,
                                              log_x=log_x, log_y=log_y)
                slope = _loglog_slope(df_g[x_col], df_g[y_col],
                                      log_x=log_x, log_y=log_y)
                ann = f"${sym}={cr:.2f}$"
                if np.isfinite(slope) and (log_x or log_y):
                    ann += f"\n$\\alpha={slope:.2f}$"
                ax.text(0.04, 0.96, ann,
                        transform=ax.transAxes, fontsize=7.5, color='#222222',
                        va='top', ha='left',
                        bbox=dict(boxstyle='square,pad=0.12', fc='white',
                                  ec='none', alpha=0.72))

            if mark_failures and baseline_x is not None and baseline_y is not None:
                fail_mask = (df_g[x_col] < baseline_x) | (df_g[y_col] < baseline_y)
                if fail_mask.any():
                    ax.scatter(df_g.loc[fail_mask, x_col],
                               df_g.loc[fail_mask, y_col],
                               marker='x', s=18, color=_COLOR_BAD, zorder=5,
                               linewidths=0.9, alpha=0.85)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        if baseline_y is not None:
            ax.axhline(baseline_y, color='#555555', linestyle='--',
                       linewidth=0.7, alpha=0.55, zorder=2)
        if baseline_x is not None:
            ax.axvline(baseline_x, color='#555555', linestyle='--',
                       linewidth=0.7, alpha=0.55, zorder=2)

        ax.set_title(group_labels.get(group, group), fontsize=9,
                     fontweight='semibold', pad=5, loc='left', color='#222222')

        ax.set_ylabel(get_metric_display(y_col) if c == 0 else '', fontsize=9.5)
        ax.set_xlabel('')

        apply_tufte_style(ax, grid_axis='both', offset_spines=False)
        ax.tick_params(axis='both', labelsize=8)

        if log_x or log_y:
            if log_x:
                format_log_axes(ax, which='x', dense=False)
            if log_y:
                format_log_axes(ax, which='y', dense=True)

    for idx in range(n_groups, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    for row in axes:
        for ax in row:
            if ax.get_visible():
                if ylim is not None:
                    ax.set_ylim(ylim)
                if xlim is not None:
                    ax.set_xlim(xlim)

    fig.supxlabel(get_metric_display(x_col), fontsize=9.5, y=0.01)
    fig.tight_layout(h_pad=1.1, w_pad=0.9)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {Path(output_path).name}")