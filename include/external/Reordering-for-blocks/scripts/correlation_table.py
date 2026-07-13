"""
Generate correlation tables between metrics and kernel performance (GFLOPS),
and median structural improvement ratio tables for reordering algorithms.
Outputs LaTeX tables.
"""

import pandas as pd
import numpy as np
import re
import sys
import warnings
from pathlib import Path
import argparse

# Allow importing plot_utils from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_utils as pu
from settings import (
    KERNEL_NAMES, ALL_METRICS, PERMS, BLOCK_SIZES,
    get_metric_display, get_metric_short, get_perm_display,
    enabled_metrics, block_density_metrics,
)

warnings.filterwarnings('ignore')


# =============================================================================
# Helpers
# =============================================================================

def _ordered_kernels(df, kernel_names):
    """Return kernel IDs in kernel_names order, falling back to alphabetical."""
    all_kernels = sorted(df['kernel_id'].unique())
    kernels = [k for k in kernel_names.keys() if k in all_kernels]
    return kernels if kernels else all_kernels


def _corr_method_tag():
    """Return a short tag for filenames, e.g. 'pearson', 'spearman'."""
    return pu.get_correlation_method()


def compute_correlations(df, n_cols, metrics, kernels=None, original_only=False):
    """Compute correlations between metrics and GFLOPS.

    The correlation method is read from ``filter_config.yaml``
    (``display.correlation_method``).

    Args:
        df: DataFrame with merged data
        n_cols: Filter to this n_cols value
        metrics: List of metric columns to compute correlations for
        kernels: List of kernels to include (None = all)
        original_only: If True, use only original (unreordered) matrices.

    Returns:
        DataFrame with correlations (rows=kernels, cols=metrics with _corr suffix)
    """
    df_nc = df[df['n_cols'] == n_cols]
    if original_only:
        df_nc = df_nc[df_nc['strategy'] == 'Original']

    if kernels is None:
        kernels = sorted(df_nc['kernel_id'].unique())

    method = pu.get_correlation_method()

    results = []
    for kernel in kernels:
        df_k = df_nc[df_nc['kernel_id'] == kernel]
        row_data = {'kernel': kernel}

        for metric in metrics:
            if metric not in df_k.columns:
                row_data[f'{metric}_corr'] = np.nan
                continue

            valid = df_k[[metric, 'gflops']].dropna()

            if len(valid) >= 10:
                r, _ = pu.compute_correlation(valid[metric], valid['gflops'], method)
                row_data[f'{metric}_corr'] = r
            else:
                row_data[f'{metric}_corr'] = np.nan

        results.append(row_data)

    return pd.DataFrame(results)


# =============================================================================
# LaTeX Generation
# =============================================================================

def _multiline_header(name):
    """Wrap a metric name into a \\shortstack with up to 3 lines for compact headers."""
    # Special case: names ending with a $...$ block (e.g. 'Block Density $32{\times}32$')
    m = re.match(r'^(.+?)\s+(\$.*\$)$', name)
    if m:
        prefix = m.group(1)
        suffix = m.group(2)
        prefix_words = prefix.split()
        if len(prefix_words) == 1:
            return r'\shortstack{' + prefix_words[0] + r'\\' + suffix + '}'
        else:
            return r'\shortstack{' + r'\\'.join(prefix_words) + r'\\' + suffix + '}'
    words = name.split()
    if len(words) <= 1:
        return name
    if len(words) == 2:
        return r'\shortstack{' + words[0] + r'\\' + words[1] + '}'
    # 3+ words: one word per line (up to ~3-4 lines)
    return r'\shortstack{' + r'\\'.join(words) + '}'


def correlation_to_latex(corr_df, metrics, kernel_names,
                         caption=None, label=None,
                         header_name_func=None):
    """Convert correlation DataFrame to LaTeX table string.

    Args:
        corr_df: DataFrame with correlations (must have 'kernel' column and
                 columns with _corr suffix)
        metrics: List of metric columns in desired order (base names without suffix)
        kernel_names: Dict mapping kernel IDs to display names
        caption: Table caption (optional)
        label: Table label for referencing (optional)
        header_name_func: Callable(metric_key) -> header string. Defaults
                          to ``get_metric_display``.

    Returns:
        LaTeX table string
    """
    if header_name_func is None:
        header_name_func = get_metric_display

    lines = []

    # Table header
    lines.append(r'\begin{table}[htb!]')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')

    # Column specification: first column left-aligned, rest centered
    col_spec = 'l' + 'c' * len(metrics)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Header row (multiline full names)
    header_cols = ['Kernel'] + [_multiline_header(header_name_func(m)) for m in metrics]
    lines.append(' & '.join(header_cols) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for _, row in corr_df.iterrows():
        kernel = row['kernel']
        kernel_display = kernel_names.get(kernel, kernel)

        # Find the maximum value in this row (for bolding)
        metric_values = {m: row.get(f'{m}_corr', np.nan) for m in metrics}
        valid_values = {m: v for m, v in metric_values.items() if not pd.isna(v)}
        max_metric = max(valid_values, key=lambda m: abs(valid_values[m])) if valid_values else None

        values = [kernel_display]
        for metric in metrics:
            val = row.get(f'{metric}_corr', np.nan)

            if pd.isna(val):
                values.append('--')
            elif metric == max_metric:
                values.append(r'\textbf{' + f'{val:.3f}' + '}')
            else:
                values.append(f'{val:.3f}')

        lines.append(' & '.join(values) + r' \\')

    # Table footer
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    if caption:
        lines.append(r'\caption{' + caption + '}')
    if label:
        lines.append(r'\label{' + label + '}')

    lines.append(r'\end{table}')

    return '\n'.join(lines)


def _build_metric_legend(metrics):
    """Build a LaTeX legend string expanding metric acronyms.

    Example output: 'RBW: Relative Bandwidth, RRS: Relative Row Spread, ...'
    """
    parts = []
    for m in metrics:
        abbr = get_metric_short(m)
        full = get_metric_display(m)
        parts.append(f"{abbr}: {full}")
    return ', '.join(parts) + '.'


def _generate_corr_table(corr_df, metrics, kernel_names,
                          output_dir, caption_tpl, label_prefix,
                          filename_prefix, n_cols_int,
                          header_name_func=None):
    """Generate a correlation LaTeX table for one n_cols value."""
    tag = _corr_method_tag()
    corr_display = pu.correlation_display_name()
    caption = caption_tpl.format(corr_display=corr_display)
    label = f"tab:{label_prefix}_{tag}_ncols_{n_cols_int}"

    latex = correlation_to_latex(
        corr_df, metrics, kernel_names,
        caption=caption, label=label,
        header_name_func=header_name_func)

    out_path = output_dir / f"{filename_prefix}_{tag}_ncols_{n_cols_int}.tex"
    with open(out_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {out_path}")


def generate_all_tables(df, output_dir, metrics=None, kernel_names=None,
                        original_only=False):
    """Generate LaTeX correlation tables for all n_cols values.

    Args:
        original_only: If True, use only original (unreordered) matrices.
    """
    if metrics is None:
        metrics = enabled_metrics()
    if kernel_names is None:
        kernel_names = KERNEL_NAMES

    suffix = '_original' if original_only else ''

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = _ordered_kernels(df, kernel_names)

    for n_cols in sorted(df['n_cols'].unique()):
        n_cols_int = int(n_cols)
        corr_df = compute_correlations(df, n_cols, metrics, kernels,
                                       original_only=original_only)

        if original_only:
            df_nc = df[(df['n_cols'] == n_cols) & (df['strategy'] == 'Original')]
            n_matrices = df_nc['matrix'].nunique()
            caption_tpl = (
                "{corr_display} correlation between metrics and SpMM GFLOPS "
                f"($n_{{{{cols}}}} = {n_cols_int}$). "
                f"Original (unreordered) matrices only ({n_matrices:,} matrices).")
        else:
            df_nc = df[df['n_cols'] == n_cols]
            n_configs = df_nc[['matrix', 'perm', 'perm_type']].drop_duplicates().shape[0]
            caption_tpl = (
                "{corr_display} correlation between metrics and SpMM GFLOPS "
                f"($n_{{{{cols}}}} = {n_cols_int}$). "
                f"All matrices and reorderings ({n_configs:,} configurations).")

        _generate_corr_table(
            corr_df, metrics, kernel_names,
            output_dir, caption_tpl,
            f'correlation{suffix}', f'correlation{suffix}',
            n_cols_int)


def generate_blocksize_tables(df, output_dir, kernel_names=None,
                              original_only=False):
    """Generate LaTeX block-density correlation tables for all n_cols.

    Args:
        original_only: If True, use only original (unreordered) matrices.
    """
    if kernel_names is None:
        kernel_names = KERNEL_NAMES

    suffix = '_original' if original_only else ''

    bd_metrics = block_density_metrics()
    def _bd_header(m):
        bs = m.split('_')[-1]
        return f'${bs}\\times{bs}$'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = _ordered_kernels(df, kernel_names)

    for n_cols in sorted(df['n_cols'].unique()):
        n_cols_int = int(n_cols)
        corr_df = compute_correlations(df, n_cols, bd_metrics, kernels,
                                       original_only=original_only)

        if original_only:
            df_nc = df[(df['n_cols'] == n_cols) & (df['strategy'] == 'Original')]
            n_matrices = df_nc['matrix'].nunique()
            caption_tpl = (
                "{corr_display} correlation between block density and SpMM GFLOPS "
                f"across block sizes ($n_{{{{cols}}}} = {n_cols_int}$). "
                f"Original (unreordered) matrices only ({n_matrices:,} matrices).")
        else:
            df_nc = df[df['n_cols'] == n_cols]
            n_configs = df_nc[['matrix', 'perm', 'perm_type']].drop_duplicates().shape[0]
            caption_tpl = (
                "{corr_display} correlation between block density and SpMM GFLOPS "
                f"across block sizes ($n_{{{{cols}}}} = {n_cols_int}$). "
                f"All matrices and reorderings ({n_configs:,} configurations).")

        _generate_corr_table(
            corr_df, bd_metrics, kernel_names,
            output_dir, caption_tpl,
            f'blocksize{suffix}', f'blocksize{suffix}',
            n_cols_int, header_name_func=_bd_header)


def generate_per_metric_tables(df, output_dir, metrics=None, kernel_names=None,
                               original_only=False):
    """Generate per-metric LaTeX tables: kernels (rows) x n_cols (columns).

    Args:
        original_only: If True, use only original (unreordered) matrices.
    """
    if metrics is None:
        metrics = enabled_metrics()
    if kernel_names is None:
        kernel_names = KERNEL_NAMES

    suffix = '_original' if original_only else ''

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_work = df[df['strategy'] == 'Original'] if original_only else df

    n_cols_values = sorted(df_work['n_cols'].unique())
    kernels = _ordered_kernels(df_work, kernel_names)
    method = pu.get_correlation_method()
    tag = _corr_method_tag()
    corr_display = pu.correlation_display_name()
    orig_note = ' (original matrices only)' if original_only else ''

    for mc in metrics:
        if mc not in df_work.columns:
            continue

        metric_display = get_metric_display(mc)
        metric_safe = pu.safe_filename(mc)

        rows = []
        for kernel in kernels:
            df_k = df_work[df_work['kernel_id'] == kernel]
            row_data = {'kernel': kernel}
            for n_cols in n_cols_values:
                df_kn = df_k[df_k['n_cols'] == n_cols]
                valid = df_kn[[mc, 'gflops']].dropna()
                if len(valid) < 5:
                    row_data[int(n_cols)] = np.nan
                    continue
                r, _ = pu.compute_correlation(valid[mc], valid['gflops'], method)
                row_data[int(n_cols)] = r
            rows.append(row_data)

        corr_df = pd.DataFrame(rows)
        n_cols_ints = [int(nc) for nc in n_cols_values]

        if corr_df[n_cols_ints].isna().all().all():
            continue

        lines = []
        lines.append(r'\begin{table}[htb!]')
        lines.append(r'\centering')
        lines.append(r'\footnotesize')
        col_spec = 'l' + 'c' * len(n_cols_ints)
        lines.append(r'\begin{tabular}{' + col_spec + '}')
        lines.append(r'\toprule')

        header_cols = ['Kernel'] + [f'$n_{{cols}} = {nc}$' for nc in n_cols_ints]
        lines.append(' & '.join(header_cols) + r' \\')
        lines.append(r'\midrule')

        for _, row in corr_df.iterrows():
            kernel_label = kernel_names.get(row['kernel'], row['kernel'])
            cells = [kernel_label]
            for nc in n_cols_ints:
                val = row[nc]
                if pd.isna(val):
                    cells.append('--')
                else:
                    cells.append(f'{val:.3f}')
            lines.append(' & '.join(cells) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(
            r"\caption{" + corr_display + " between " + metric_display +
            r' and SpMM GFLOPS across $n_{cols}$ values' + orig_note + '.}')
        lines.append(r'\label{tab:per_metric' + suffix + '_' + tag + '_' + metric_safe + '}')
        lines.append(r'\end{table}')

        out_path = output_dir / f"per_metric{suffix}_{tag}_{metric_safe}.tex"
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Saved: {out_path}")


# =============================================================================
# Median Structural Improvement Ratio Tables
# =============================================================================

def compute_improvement_ratios(df_analysis, metrics=None):
    """Compute per-(matrix, perm, perm_type) improvement ratios vs the original.

    For each matrix the original is the row with perm='None'.
    Improvement ratio:
        higher_is_better=True  -> reordered / original  (>1 means improved)
        higher_is_better=False -> original / reordered   (>1 means improved)

    Returns:
        DataFrame with columns: matrix, perm, perm_type, and one
        improvement ratio column per metric (named '<metric>_imp').
    """
    if metrics is None:
        metrics = enabled_metrics()

    # Only keep metrics that have a direction defined
    metrics = [m for m in metrics
               if m in df_analysis.columns and ALL_METRICS[m].get('higher_is_better') is not None]

    originals = df_analysis[df_analysis['perm'] == 'None'][['matrix'] + metrics].copy()
    originals = originals.groupby('matrix')[metrics].mean().reset_index()
    originals = originals.rename(columns={m: f'{m}_orig' for m in metrics})

    reordered = df_analysis[df_analysis['perm'] != 'None'].copy()
    reordered = reordered.merge(originals, on='matrix', how='left')

    for m in metrics:
        orig_col = f'{m}_orig'
        imp_col = f'{m}_imp'
        if ALL_METRICS[m]['higher_is_better']:
            reordered[imp_col] = reordered[m] / reordered[orig_col]
        else:
            reordered[imp_col] = reordered[orig_col] / reordered[m]

    keep_cols = ['matrix', 'perm', 'perm_type'] + [f'{m}_imp' for m in metrics]
    return reordered[keep_cols], metrics


def improvement_to_latex(median_df, metrics, caption=None, label=None):
    """Convert a median-improvement DataFrame to a LaTeX table.

    Args:
        median_df: DataFrame with 'perm' column and '<metric>_imp' value columns.
        metrics: ordered list of base metric names.
        caption, label: LaTeX caption / label.

    Returns:
        LaTeX table string.
    """
    lines = []
    lines.append(r'\begin{table}[htb!]')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')

    col_spec = 'l' + 'c' * len(metrics)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    header_cols = ['Algorithm'] + [_multiline_header(get_metric_display(m)) for m in metrics]
    lines.append(' & '.join(header_cols) + r' \\')
    lines.append(r'\midrule')

    # Pre-compute best algorithm (perm) per metric column
    best_perm_per_metric = {}
    for m in metrics:
        col = f'{m}_imp'
        valid = median_df[['perm', col]].dropna(subset=[col])
        if not valid.empty:
            best_perm_per_metric[m] = valid.loc[valid[col].idxmax(), 'perm']

    for _, row in median_df.iterrows():
        perm = row['perm']
        perm_display = get_perm_display(perm)

        cells = [perm_display]
        for m in metrics:
            val = row.get(f'{m}_imp', np.nan)
            if pd.isna(val):
                cells.append('--')
            elif best_perm_per_metric.get(m) == perm:
                cells.append(r'\textbf{' + f'{val:.2f}' + '}')
            else:
                cells.append(f'{val:.2f}')

        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    if caption:
        lines.append(r'\caption{' + caption + '}')
    if label:
        lines.append(r'\label{' + label + '}')

    lines.append(r'\end{table}')
    return '\n'.join(lines)


def _write_improvement_table(imp_df, imp_cols, used_metrics,
                              ordered_perms, caption, label, out_path):
    """Compute medians, sort by perm order, and write one improvement table."""
    median_df = imp_df.groupby('perm')[imp_cols].median().reset_index()
    perm_order = {p: i for i, p in enumerate(ordered_perms)}
    median_df['_order'] = median_df['perm'].map(perm_order).fillna(999)
    median_df = median_df.sort_values('_order').drop(columns='_order')

    latex = improvement_to_latex(median_df, used_metrics,
                                  caption=caption, label=label)
    with open(out_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {out_path}")


# =============================================================================
# Improvement ↔ Speedup Correlation Tables
# =============================================================================

def _enabled_improvement_metrics():
    """Return improvement column names matching the enabled correlation metrics."""
    base = [
        'bandwidth_improvement',
        'col_spread_improvement',
        'vertical_adjacency_improvement',
        'profile_improvement',
        'reuse_distance_improvement',
        'index_distance_improvement',
    ]
    # Mirror the block densities enabled in settings
    bd = [f'density_improvement_{bs}' for bs in BLOCK_SIZES
          if ALL_METRICS.get(f'block_density_{bs}', {}).get('enabled')]
    return base + bd


def _density_improvement_metrics():
    """Return density improvement columns for all block sizes."""
    return [f'density_improvement_{bs}' for bs in BLOCK_SIZES]


def compute_imp_correlations(df, n_cols, imp_metrics, kernels=None,
                             perm_type=None, method=None,
                             log_transform=False):
    """Compute correlations between improvement metrics and speedup.

    Only considers reordered rows (strategy != 'Original').

    If *method* is None the correlation method is read from config.
    If *perm_type* is given, only rows with that perm_type are used.
    If *log_transform* is True, both metric and speedup are log-transformed
    before computing the correlation (rows with non-positive values are dropped).
    """
    df_nc = df[(df['n_cols'] == n_cols) & (df['strategy'] != 'Original')]
    if perm_type is not None:
        df_nc = df_nc[df_nc['perm_type'] == perm_type]

    if kernels is None:
        kernels = sorted(df_nc['kernel_id'].unique())

    if method is None:
        method = pu.get_correlation_method()

    results = []
    for kernel in kernels:
        df_k = df_nc[df_nc['kernel_id'] == kernel]
        row_data = {'kernel': kernel}

        for metric in imp_metrics:
            if metric not in df_k.columns:
                row_data[f'{metric}_corr'] = np.nan
                continue

            valid = df_k[[metric, 'speedup']].dropna()
            valid = valid[np.isfinite(valid[metric]) & np.isfinite(valid['speedup'])]

            if log_transform:
                valid = valid[(valid[metric] > 0) & (valid['speedup'] > 0)]
                if len(valid) >= 10:
                    r, _ = pu.compute_correlation(
                        np.log(valid[metric]), np.log(valid['speedup']),
                        method)
                    row_data[f'{metric}_corr'] = r
                else:
                    row_data[f'{metric}_corr'] = np.nan
            elif len(valid) >= 10:
                r, _ = pu.compute_correlation(valid[metric], valid['speedup'], method)
                row_data[f'{metric}_corr'] = r
            else:
                row_data[f'{metric}_corr'] = np.nan

        results.append(row_data)

    return pd.DataFrame(results)


def generate_imp_correlation_tables(df, output_dir, kernel_names=None):
    """Generate LaTeX tables correlating improvement metrics with speedup."""
    if kernel_names is None:
        kernel_names = KERNEL_NAMES

    imp_metrics = _enabled_improvement_metrics()
    tag = _corr_method_tag()
    corr_display = pu.correlation_display_name()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = _ordered_kernels(df, kernel_names)

    for n_cols in sorted(df['n_cols'].unique()):
        n_cols_int = int(n_cols)
        df_nc = df[(df['n_cols'] == n_cols) & (df['strategy'] != 'Original')]
        n_configs = df_nc[['matrix', 'perm', 'perm_type']].drop_duplicates().shape[0]
        corr_df = compute_imp_correlations(df, n_cols, imp_metrics, kernels)

        caption = (
            f"{corr_display} correlation between structural improvement and speedup "
            f"($n_{{{{cols}}}} = {n_cols_int}$). "
            f"Only reordered configurations are included ({n_configs:,} total).")
        label = f"tab:imp_correlation_{tag}_ncols_{n_cols_int}"

        latex = correlation_to_latex(
            corr_df, imp_metrics, kernel_names,
            caption=caption, label=label)

        out_path = output_dir / f"imp_correlation_{tag}_ncols_{n_cols_int}.tex"
        with open(out_path, 'w') as f:
            f.write(latex)
        print(f"Saved: {out_path}")


def generate_imp_per_metric_tables(df, output_dir, kernel_names=None):
    """Generate per-improvement-metric LaTeX tables: kernels (rows) x n_cols (columns).

    Each table shows how the correlation between one improvement metric and
    speedup varies across n_cols values.
    """
    if kernel_names is None:
        kernel_names = KERNEL_NAMES

    imp_metrics = _enabled_improvement_metrics()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_work = df[df['strategy'] != 'Original']

    n_cols_values = sorted(df_work['n_cols'].unique())
    kernels = _ordered_kernels(df_work, kernel_names)
    method = pu.get_correlation_method()
    tag = _corr_method_tag()
    corr_display = pu.correlation_display_name()

    for mc in imp_metrics:
        if mc not in df_work.columns:
            continue

        metric_display = get_metric_display(mc)
        metric_safe = pu.safe_filename(mc)

        rows = []
        for kernel in kernels:
            df_k = df_work[df_work['kernel_id'] == kernel]
            row_data = {'kernel': kernel}
            for n_cols in n_cols_values:
                df_kn = df_k[df_k['n_cols'] == n_cols]
                valid = df_kn[[mc, 'speedup']].dropna()
                valid = valid[np.isfinite(valid[mc]) & np.isfinite(valid['speedup'])]
                if len(valid) < 10:
                    row_data[int(n_cols)] = np.nan
                    continue
                r, _ = pu.compute_correlation(valid[mc], valid['speedup'], method)
                row_data[int(n_cols)] = r
            rows.append(row_data)

        corr_df = pd.DataFrame(rows)
        n_cols_ints = [int(nc) for nc in n_cols_values]

        if corr_df[n_cols_ints].isna().all().all():
            continue

        lines = []
        lines.append(r'\begin{table}[htb!]')
        lines.append(r'\centering')
        lines.append(r'\footnotesize')
        col_spec = 'l' + 'c' * len(n_cols_ints)
        lines.append(r'\begin{tabular}{' + col_spec + '}')
        lines.append(r'\toprule')

        header_cols = ['Kernel'] + [f'$n_{{cols}} = {nc}$' for nc in n_cols_ints]
        lines.append(' & '.join(header_cols) + r' \\')
        lines.append(r'\midrule')

        for _, row in corr_df.iterrows():
            kernel_label = kernel_names.get(row['kernel'], row['kernel'])
            cells = [kernel_label]
            for nc in n_cols_ints:
                val = row[nc]
                if pd.isna(val):
                    cells.append('--')
                else:
                    cells.append(f'{val:.3f}')
            lines.append(' & '.join(cells) + r' \\')

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(
            r"\caption{" + corr_display + " between " + metric_display +
            r' and speedup across $n_{cols}$ values (reordered configurations only).}')
        lines.append(r'\label{tab:imp_per_metric_' + tag + '_' + metric_safe + '}')
        lines.append(r'\end{table}')

        out_path = output_dir / f"imp_per_metric_{tag}_{metric_safe}.tex"
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Saved: {out_path}")


def generate_imp_blocksize_tables(df, output_dir, kernel_names=None):
    """Generate LaTeX tables correlating density improvement with speedup across block sizes."""
    if kernel_names is None:
        kernel_names = KERNEL_NAMES

    bd_imp_metrics = _density_improvement_metrics()
    tag = _corr_method_tag()
    corr_display = pu.correlation_display_name()

    def _bd_imp_header(m):
        bs = m.split('_')[-1]
        return f'${bs}\\times{bs}$'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = _ordered_kernels(df, kernel_names)

    for n_cols in sorted(df['n_cols'].unique()):
        n_cols_int = int(n_cols)
        df_nc = df[(df['n_cols'] == n_cols) & (df['strategy'] != 'Original')]
        n_configs = df_nc[['matrix', 'perm', 'perm_type']].drop_duplicates().shape[0]
        corr_df = compute_imp_correlations(df, n_cols, bd_imp_metrics, kernels)

        caption = (
            f"{corr_display} correlation between block density improvement and speedup "
            f"across block sizes ($n_{{{{cols}}}} = {n_cols_int}$). "
            f"Only reordered configurations are included ({n_configs:,} total).")
        label = f"tab:imp_blocksize_{tag}_ncols_{n_cols_int}"

        latex = correlation_to_latex(
            corr_df, bd_imp_metrics, kernel_names,
            caption=caption, label=label,
            header_name_func=_bd_imp_header)

        out_path = output_dir / f"imp_blocksize_{tag}_ncols_{n_cols_int}.tex"
        with open(out_path, 'w') as f:
            f.write(latex)
        print(f"Saved: {out_path}")


def generate_improvement_tables(df_analysis, output_dir, metrics=None):
    """Generate LaTeX tables of median structural improvement ratios.

    Creates one table per perm_type plus a combined table.
    """
    if metrics is None:
        metrics = enabled_metrics()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imp_df, used_metrics = compute_improvement_ratios(df_analysis, metrics)
    if imp_df.empty:
        print("No improvement data to generate tables.")
        return

    # Determine perm order from PERMS (or alphabetical fallback)
    all_perms = sorted(imp_df['perm'].unique())
    ordered_perms = [p for p in PERMS if p in all_perms]
    ordered_perms += [p for p in all_perms if p not in ordered_perms]

    n_matrices = df_analysis[df_analysis['perm'] == 'None']['matrix'].nunique()
    imp_cols = [f'{m}_imp' for m in used_metrics]

    caption = (f"Median structural improvement ratio per reordering algorithm "
               f"({n_matrices} matrices). "
               f"Values $>1$ indicate improvement over the original ordering.")
    _write_improvement_table(
        imp_df, imp_cols, used_metrics, ordered_perms,
        caption=caption, label="tab:improvement",
        out_path=output_dir / "improvement.tex")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate correlation tables as LaTeX files"
    )
    # Random pipeline shortcut
    parser.add_argument(
        "--random", action="store_true",
        help="Use random-pipeline data (data_random paths from filter_config.yaml)"
    )
    # Perm-type pipeline selection
    parser.add_argument(
        "--row", action="store_true",
        help="Use ROW perm_type pipeline (output to plots_row/ or plots_random_row/)"
    )
    parser.add_argument(
        "--symmetric", action="store_true",
        help="Use SYMMETRIC perm_type pipeline (default)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for .tex files (default: plots/correlation_tables or plots_random/correlation_tables with --random)"
    )
    # Filter config (all filtering is driven by filter_config.yaml;
    # these flags *override* settings in the config file)
    parser.add_argument(
        "--filter-config", default=None,
        help="Path to filter_config.yaml (default: scripts/filter_config.yaml)"
    )
    parser.add_argument(
        "--operations", default=None,
        help="Override operations CSV path from config"
    )
    parser.add_argument(
        "--analysis", default=None,
        help="Override analysis CSV path from config"
    )
    parser.add_argument(
        "--matrices-list", default=None,
        help="Override matrices list path from config"
    )
    parser.add_argument(
        "--one-per-family", action="store_true", default=None,
        help="Override: enable one-per-family filter"
    )
    parser.add_argument(
        "--no-one-per-family", action="store_false", dest="one_per_family",
        help="Override: disable one-per-family filter"
    )
    parser.add_argument(
        "--blocksize-only",
        action="store_true",
        help="Generate only block size correlation tables"
    )
    parser.add_argument(
        "--improvement-only",
        action="store_true",
        help="Generate only median improvement ratio tables"
    )

    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing output directory before generating tables"
    )

    # Fine-grained section selection (used by run_plots.py wrapper)
    SECTION_CHOICES = ['correlations', 'blocksize', 'per-metric', 'improvement',
                       'imp-correlations', 'imp-blocksize', 'imp-per-metric',
                       'correlations-original', 'blocksize-original', 'per-metric-original']
    parser.add_argument(
        "--sections", nargs="+", choices=SECTION_CHOICES, default=None,
        help=(
            "Run only the listed table sections. "
            "Overrides --blocksize-only / --improvement-only. "
            f"Choices: {', '.join(SECTION_CHOICES)}"
        ),
    )
    return parser.parse_args()


def main():
    import shutil

    args = parse_args()

    # Resolve perm_type pipeline
    perm_type_filter = 'ROW' if args.row else 'SYMMETRIC'
    pipeline_key = ('random_' if args.random else '') + perm_type_filter.lower()

    # Apply --random defaults
    if args.output is None:
        data_label = 'random' if args.random else 'original'
        perm_label = 'row' if args.row else 'symmetric'
        args.output = f"plots/{data_label}_{perm_label}/correlation_tables"

    out_dir = Path(args.output)

    # --clean: remove and recreate output directory
    if args.clean and out_dir.exists():
        print(f"Cleaning output directory: {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load, filter, and add derived columns (single pipeline)
    # ------------------------------------------------------------------
    cli_overrides = {
        'operations_csv': args.operations,
        'analysis_csv': args.analysis,
        'matrices_list': args.matrices_list,
        'one_per_family': args.one_per_family,
        'random': args.random,
    }

    df, df_analysis, _cfg = pu.load_and_filter_data(
        config_path=args.filter_config,
        cli_overrides=cli_overrides,
    )

    # Split by perm_type — downstream code works on a single homogeneous DF
    pipeline_cfg = _cfg.get('pipelines', {}).get(pipeline_key, {})
    df, df_analysis = pu.split_by_perm_type(
        df, df_analysis, perm_type_filter, pipeline_cfg)

    # Add gflops, kernel_id, and relative metrics for correlation tables
    df = pu.add_base_metrics(df)
    df = pu.add_relative_metrics(df)
    # Add relative metrics to analysis df for improvement tables
    df_analysis = pu.add_relative_metrics(df_analysis)

    # Add speedup and improvement columns for imp-correlation tables
    df = pu.add_speedup(df)
    df = pu.add_improvement_columns(df)

    method = pu.get_correlation_method()
    print(f"\nCorrelation method: {method}")

    # ------------------------------------------------------------------
    # Generate tables (organized into subfolders)
    # ------------------------------------------------------------------
    print("\nGenerating LaTeX tables...")

    # --sections takes priority over legacy --*-only flags
    if args.sections is not None:
        sections = set(args.sections)
    elif args.improvement_only:
        sections = {'improvement'}
    elif args.blocksize_only:
        sections = {'blocksize'}
    else:
        sections = {'correlations', 'blocksize', 'per-metric', 'improvement',
                     'imp-correlations', 'imp-blocksize', 'imp-per-metric',
                     'correlations-original', 'blocksize-original', 'per-metric-original'}

    # All data (original + reordered)
    if 'correlations' in sections:
        generate_all_tables(df, out_dir / 'all')
    if 'blocksize' in sections:
        generate_blocksize_tables(df, out_dir / 'all')
    if 'per-metric' in sections:
        generate_per_metric_tables(df, out_dir / 'all')

    # Original only
    if 'correlations-original' in sections:
        generate_all_tables(df, out_dir / 'original', original_only=True)
    if 'blocksize-original' in sections:
        generate_blocksize_tables(df, out_dir / 'original', original_only=True)
    if 'per-metric-original' in sections:
        generate_per_metric_tables(df, out_dir / 'original', original_only=True)

    # Improvement tables
    if 'improvement' in sections:
        generate_improvement_tables(df_analysis, out_dir / 'improvement')

    # Improvement-correlation tables
    if 'imp-correlations' in sections:
        generate_imp_correlation_tables(df, out_dir / 'imp_correlations')
    if 'imp-blocksize' in sections:
        generate_imp_blocksize_tables(df, out_dir / 'imp_correlations')
    if 'imp-per-metric' in sections:
        generate_imp_per_metric_tables(df, out_dir / 'imp_correlations')

    print(f"\nAll tables saved to {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
