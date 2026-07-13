"""
Plot Script - Clean Architecture

Simple iteration-based plotting script that:
1. Loads and processes data once
2. Applies filters
3. Iterates over dimensions (n_cols, kernel) to generate plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plot_utils as pu
from settings import get_perm_display, KERNEL_NAMES, GROUPED_SCATTER_EXCLUDE, PERMS, ALL_METRICS, BLOCK_SIZES, get_metric_display, get_metric_color, get_metric_hatch
from correlation_table import (compute_imp_correlations,
                               _enabled_improvement_metrics,
                               _density_improvement_metrics, _corr_method_tag,
                               _ordered_kernels)

# Axis caps for ratio scatter plots (speedup, improvement)
RATIO_YLIM = (1 / 7, 7)
RATIO_XLIM = (0.05, 10)  # x-axis bounds for improvement ratios


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")

    # Random pipeline shortcut
    parser.add_argument("--random", action="store_true",
                        help="Use random-pipeline data (data_random paths from filter_config.yaml)")

    # Perm-type pipeline selection
    parser.add_argument("--row", action="store_true",
                        help="Use ROW perm_type pipeline (output to plots_row/ or plots_random_row/)")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use SYMMETRIC perm_type pipeline (default)")

    # Output
    parser.add_argument("--out", default=None, help="Output directory (default: plots or plots_random with --random)")

    # Filter config (all filtering is driven by filter_config.yaml;
    # these flags *override* settings in the config file)
    parser.add_argument("--filter-config", default=None,
                        help="Path to filter_config.yaml (default: scripts/filter_config.yaml)")
    parser.add_argument("--operations", default=None,
                        help="Override operations CSV path from config")
    parser.add_argument("--analysis", default=None,
                        help="Override analysis CSV path from config")
    parser.add_argument("--matrices-list", default=None,
                        help="Override matrices list path from config")
    parser.add_argument("--one-per-family", action="store_true", default=None,
                        help="Override: enable one-per-family filter")
    parser.add_argument("--no-one-per-family", action="store_false", dest="one_per_family",
                        help="Override: disable one-per-family filter")
    parser.add_argument("--include-rectangular", action="store_true", default=None,
                        help="Override: include rectangular matrices")
    parser.add_argument("--min-size", type=int, default=None,
                        help="Override: minimum matrix size (rows)")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Override: maximum matrix size (rows or cols)")
    
    # Plot selection
    parser.add_argument("--only-reorder-analysis", action="store_true")
    parser.add_argument("--only-kernels", action="store_true")
    parser.add_argument("--n-cols", type=int, default=None, help="Filter to specific n_cols")
    parser.add_argument("--kernel", type=str, default=None, help="Filter to specific kernel")
    
    # Fine-grained section selection (used by run_plots.py wrapper)
    SECTION_CHOICES = [
        'grouped-scatter',
        'breakeven',
        'reorder-analysis',
        'profiles',
        'speedup-profiles',
        'aggregate-improvement',
        'imp-correlation',
        'partial-correlation',
        'feature-importance',
    ]
    parser.add_argument(
        "--sections", nargs="+", choices=SECTION_CHOICES, default=None,
        help=(
            "Run only the listed plot sections. "
            "Overrides --only-kernels / --only-reorder-analysis. "
            f"Choices: {', '.join(SECTION_CHOICES)}"
        ),
    )

    # Parallelism
    parser.add_argument(
        "-j", "--jobs", type=int, default=None,
        help="Number of parallel workers for plot generation (default: min(cpu_count, 8), 1=sequential)",
    )

    return parser.parse_args()


def generate_grouped_scatter_plots(df, out_dir, args):
    """Grouped 2x3 scatter: improvement metric vs speedup (log-log) per n_cols."""
    print("\n=== Grouped Improvement vs Speedup Scatter ===")
    n_jobs = getattr(args, 'jobs', None)

    n_cols_values = sorted(df['n_cols'].unique())
    if args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"Warning: n_cols={args.n_cols} not found. Available: {n_cols_values}")
            return

    tasks = []

    for n_cols in n_cols_values:
        print(f"\n--- n_cols = {n_cols} ---")
        df_nc = df[df['n_cols'] == n_cols]

        kernels = sorted(df_nc['kernel_id'].unique())
        if args.kernel:
            kernels = [k for k in kernels if args.kernel.lower() in k.lower()]
            if not kernels:
                print(f"No kernels matching '{args.kernel}'")
                continue

        grouped_kernels = [k for k in kernels if k not in GROUPED_SCATTER_EXCLUDE]
        kernel_labels = {k: KERNEL_NAMES.get(k, k) for k in grouped_kernels}
        df_nc_reordered = df_nc[df_nc['strategy'] != 'Original']

        grouped_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_improvement_vs_speedup_loglog"
        grouped_dir.mkdir(parents=True, exist_ok=True)

        _kw = dict(ylim=RATIO_YLIM, xlim=RATIO_XLIM,
                    baseline_x=1.0, baseline_y=1.0,
                    quadrant_colors=True)

        for bs in BLOCK_SIZES:
            imp_col = f'density_improvement_{bs}'
            if imp_col in df_nc_reordered.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_dir / f"speedup_vs_density_imp_bs{bs}_loglog.png",
                    group_labels=kernel_labels, log_x=True, log_y=True,
                    **_kw)))

        for imp_col in ['bandwidth_improvement', 'row_spread_improvement',
                        'vertical_adjacency_improvement',
                        'reuse_distance_improvement',
                        'index_distance_improvement']:
            if imp_col in df_nc_reordered.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_dir / f"speedup_vs_{imp_col}_loglog.png",
                    group_labels=kernel_labels, log_x=True, log_y=True,
                    **_kw)))

    print(f"\n  Collected {len(tasks)} grouped scatter tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)


def generate_breakeven_plots(df, out_dir, reordering_csv='results/results_reordering.csv', args=None):
    """Compute and plot break-even number of operations for each reordering.

    For every (matrix, perm, kernel_id, n_cols) tuple the break-even count is:

        breakeven_n = time_reordering_ms / (time_baseline_ms - time_reordered_ms)

    Cases where reordering is harmful (denominator ≤ 0) are shown as '×'
    markers at a cap line and excluded from boxplot statistics.

    Outputs one boxplot per (kernel, n_cols) to
    ``plots/n_cols_{N}/{kernel}/breakeven/``.
    """
    print("\n=== Break-even Analysis ===")

    # --- Load reordering times ---
    reorder_csv_path = Path(reordering_csv)
    if not reorder_csv_path.exists():
        print(f"  Reordering CSV not found at {reorder_csv_path} — skipping break-even plots. "
              "Run  python scripts/parse_results.py  first to generate it.")
        return

    df_time = pd.read_csv(reorder_csv_path)
    if df_time.empty or 'time_reordering_ms' not in df_time.columns:
        print("  No reordering timing data available — skipping.")
        return

    # Keep only timing columns we need
    df_time = df_time[['matrix', 'perm', 'time_reordering_ms']].copy()
    # Average over duplicate runs for the same (matrix, perm)
    df_time = df_time.groupby(['matrix', 'perm'], as_index=False)['time_reordering_ms'].mean()
    print(f"  Loaded {len(df_time)} reordering timing entries")

    # --- Baseline operation times ---
    df_baseline = (
        df[df['strategy'] == 'Original']
        .groupby(['matrix', 'kernel_id', 'n_cols'])['time_operation_ms']
        .mean()
        .reset_index()
        .rename(columns={'time_operation_ms': 'time_baseline_ms'})
    )

    # --- Reordered rows only ---
    df_reord = df[df['strategy'] != 'Original'].copy()
    if df_reord.empty:
        print("  No reordered operation data — skipping.")
        return

    # Merge baseline time
    df_reord = df_reord.merge(df_baseline, on=['matrix', 'kernel_id', 'n_cols'], how='inner')

    # Merge reordering time
    df_reord = df_reord.merge(df_time, on=['matrix', 'perm'], how='inner')

    if df_reord.empty:
        print("  No overlapping data between operations and reordering timing — skipping.")
        return

    # --- Compute break-even ---
    df_reord['time_saved_ms'] = df_reord['time_baseline_ms'] - df_reord['time_operation_ms']
    df_reord['breakeven_n'] = np.where(
        df_reord['time_saved_ms'] > 0,
        df_reord['time_reordering_ms'] / df_reord['time_saved_ms'],
        np.nan,
    )
    df_reord['harmful'] = df_reord['time_saved_ms'] <= 0

    n_valid = int((~df_reord['harmful']).sum())
    n_harmful = int(df_reord['harmful'].sum())
    print(f"  Break-even computed: {n_valid} valid, {n_harmful} harmful (never breaks even)")

    # --- Apply CLI filters ---
    n_cols_values = sorted(df_reord['n_cols'].unique())
    if args is not None and args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"  Warning: n_cols={args.n_cols} not in data; available: {n_cols_values}")
            return

    kernel_filter = args.kernel if args is not None else None

    # --- Collect and generate plots ---
    n_jobs = getattr(args, 'jobs', None) if args is not None else None
    tasks = []
    for n_cols in n_cols_values:
        df_nc = df_reord[df_reord['n_cols'] == n_cols]

        kernels = sorted(df_nc['kernel_id'].unique())
        if kernel_filter:
            kernels = [k for k in kernels if kernel_filter.lower() in k.lower()]
            if not kernels:
                continue

        for kernel in kernels:
            df_k = df_nc[df_nc['kernel_id'] == kernel]
            kernel_safe = pu.safe_filename(kernel)
            breakeven_dir = out_dir / f"n_cols_{int(n_cols)}" / kernel_safe / "breakeven"
            breakeven_dir.mkdir(parents=True, exist_ok=True)

            strategies = sorted(df_k['strategy'].unique())
            palette = pu.get_strategy_palette(strategies)

            df_valid = df_k[~df_k['harmful']]
            df_harm = df_k[df_k['harmful']]

            tasks.append((pu.breakeven_boxplot, dict(
                df_valid=df_valid, df_harmful=df_harm,
                x_col='strategy', y_col='breakeven_n',
                output_path=breakeven_dir / f"breakeven.png",
                title=f"Break-even Operations\n{kernel}  (n_cols={int(n_cols)})",
                order=[s for s in strategies if s in df_valid['strategy'].unique()
                       or s in df_harm['strategy'].unique()],
                palette=palette)))

    print(f"  Collected {len(tasks)} break-even plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)
    print(f"  Break-even plots saved under {out_dir}/n_cols_*/*/breakeven/")


def generate_reorder_analysis_plots(df_analysis, out_dir, n_jobs=None):
    """Generate reordering analysis plots (independent of kernel performance)."""

    print("\n=== Reordering Analysis ===")

    reorder_dir = out_dir / "reorder_analysis"

    # Process analysis data
    df = df_analysis.copy()
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

    # Calculate improvements using the shared utility (extended mode includes
    # bandwidth_avg, max spreads, and blocks-per-row metrics)
    metrics_config = pu.build_metrics_config(df, include_extended=True)
    df = pu.compute_improvements(df, metrics_config)

    # Filter to reordered only
    df_reordered = df[df['strategy'] != 'Original']

    if df_reordered.empty:
        print("No reordered data for analysis plots")
        return

    strategies = sorted(df_reordered['strategy'].unique())

    # -----------------------------------------------------------------
    # All boxplot sections: (imp_col, title_template, subdir, filename_template)
    # -----------------------------------------------------------------
    boxplot_specs = []

    for bs in [4, 8, 16, 32, 64, 128]:
        boxplot_specs.append((
            f'density_improvement_{bs}',
            f"Density Improvement (BS {bs})",
            "density", f"density_improvement_bs{bs}.png"))

    # Collect all boxplot tasks
    tasks = []
    for imp_col, title_text, subdir, filename in boxplot_specs:
        if imp_col not in df_reordered.columns:
            continue
        output_dir = reorder_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((pu.boxplot_by_category, dict(
            df=df_reordered, x_col='strategy', y_col=imp_col,
            output_path=output_dir / filename,
            title=None,
            order=[s for s in strategies if s in df_reordered['strategy'].unique()],
            baseline=1.0, log_y=True)))

    print(f"  Collected {len(tasks)} reorder analysis plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)


def generate_profile_plots(df_analysis, out_dir):
    """Generate Dolan-Moré performance-profile plots.

    For each metric and solver (perm) s, compute the ratio relative to the
    best solver on each matrix m:
        higher_is_better:  r_{m,s} = val_{m,s} / max_{s'} val_{m,s'}  ∈ (0, 1]
        lower_is_better:   r_{m,s} = val_{m,s} / min_{s'} val_{m,s'}  ∈ [1, ∞)

    In both cases r = 1 means s is the best solver on that matrix.

    The profile plots:
      - higher_is_better: survival function P(ratio >= τ), x from 1 toward 0
      - lower_is_better:  CDF P(ratio <= τ), x from 1 toward ∞

    Higher curve = better solver in both cases.

    All permutations *and* the original ordering (None) are included as
    competing solvers.
    """
    # Same raw metrics as the correlation heatmaps (no bandwidth).
    profile_metrics = [
        'bandwidth_max',
        'locality_avg_col_spread',
        'locality_vertical_adjacency_ratio',
        'locality_profile',
        'access_dist_reuse_distance_mean',
        'access_dist_index_distance_mean',
    ]
    profile_metrics = [m for m in profile_metrics
                       if m in df_analysis.columns
                       and ALL_METRICS.get(m, {}).get('higher_is_better') is not None]
    if not profile_metrics:
        print("  No usable metrics — skipping profile plots.")
        return

    # Work directly with raw metric values (including perm='None').
    # The DataFrame is already filtered to a single perm_type by
    # split_by_perm_type(), so no per-perm_type grouping is needed.
    base_cols = ['matrix', 'perm', 'perm_type']
    group_df = df_analysis[base_cols + profile_metrics].copy()

    if group_df.empty:
        print("  No data — skipping profile plots.")
        return

    # Keep only matrices that have all perms present
    all_perms = group_df['perm'].unique()
    n_perms = len(all_perms)
    perm_counts = group_df.groupby('matrix')['perm'].nunique()
    complete_matrices = perm_counts[perm_counts == n_perms].index
    group_df = group_df[group_df['matrix'].isin(complete_matrices)]
    print(f"  Restricted to {len(complete_matrices)} matrices with all {n_perms} perms")

    if group_df.empty:
        print("  No complete matrices — skipping profile plots.")
        return

    # For each metric, compute Dolan-Moré performance ratios
    profile_data = {}  # metric -> (n_matrices, {perm -> sorted_ratios})
    for m in profile_metrics:
        sub = group_df[['matrix', 'perm', m]].dropna(subset=[m])
        # Filter out zero / negative values
        sub = sub[sub[m] > 0]
        if sub.empty:
            profile_data[m] = (0, {})
            continue

        # ratio = value / best:
        #   higher_is_better → ∈ (0, 1], lower_is_better → ∈ [1, ∞)
        hib = ALL_METRICS[m]['higher_is_better']
        if hib:
            best = sub.groupby('matrix')[m].max().rename('best')
        else:
            best = sub.groupby('matrix')[m].min().rename('best')
        sub = sub.merge(best, on='matrix')
        # relative value (value / best)
        sub['tau'] = sub[m] / sub['best']

        n_matrices = sub['matrix'].nunique()
        perm_ratios = {}
        for perm, perm_sub in sub.groupby('perm'):
            perm_ratios[perm] = np.sort(perm_sub['tau'].values)
        profile_data[m] = (n_matrices, perm_ratios)

    # Layout: 3 columns
    n_metrics = len(profile_metrics)
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    perm_order = pu.profile_perm_order(set(group_df['perm'].unique()))

    for idx, m in enumerate(profile_metrics):
        ax = axes[idx // ncols][idx % ncols]
        n_matrices, data = profile_data[m]
        if n_matrices == 0:
            continue
        hib = ALL_METRICS[m]['higher_is_better']

        # Compute xlim early so extensions can reach plot borders
        if hib:
            xlim_lo, xlim_hi = 2**-4, 1.05
        else:
            all_taus = np.concatenate([v for v in data.values()])
            max_tau = min(np.percentile(all_taus, 99) * 1.2, 2**6)
            xlim_lo, xlim_hi = 0.95, max(max_tau, 2)

        for perm in perm_order:
            if perm not in data or len(data[perm]) == 0:
                continue
            pu.draw_profile_curve(ax, data[perm], n_matrices, perm,
                                xlim_lo, xlim_hi, hib)

        ax.set_title(get_metric_display(m), fontsize=10)
        ax.set_xlabel(f"{get_metric_display(m)} relative to the best")
        ax.set_ylabel('Fraction of matrices')
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 1.05)
        if hib:
            ax.set_xlim(xlim_lo, xlim_hi)
            ax.invert_xaxis()
        else:
            ax.set_xlim(xlim_lo, xlim_hi)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend above the plots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(6, len(handles)), fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()

    prof_dir = out_dir / 'reorder_analysis' / 'profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(prof_dir / f'profiles.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved profiles.pdf/png")

    # --- Standalone block_density_16 profile plot ---
    # This uses its own data (not from profile_metrics above).
    bd_metric = 'block_density_16'
    if bd_metric in df_analysis.columns and ALL_METRICS.get(bd_metric, {}).get('higher_is_better') is not None:
        sub_bd = group_df[['matrix', 'perm']].copy()
        sub_bd[bd_metric] = df_analysis.set_index(['matrix', 'perm'])[bd_metric].reindex(
            pd.MultiIndex.from_frame(sub_bd[['matrix', 'perm']])).values
        sub_bd = sub_bd.dropna(subset=[bd_metric])
        sub_bd = sub_bd[sub_bd[bd_metric] > 0]
        if not sub_bd.empty:
            best_bd = sub_bd.groupby('matrix')[bd_metric].max().rename('best')
            sub_bd = sub_bd.merge(best_bd, on='matrix')
            sub_bd['tau'] = sub_bd[bd_metric] / sub_bd['best']
            n_matrices_bd = sub_bd['matrix'].nunique()
            data_bd = {}
            for perm, perm_sub in sub_bd.groupby('perm'):
                data_bd[perm] = np.sort(perm_sub['tau'].values)

            fig_bd, ax_bd = plt.subplots(figsize=(6, 4.5))
            xlim_lo_bd, xlim_hi_bd = 2**-4, 1.05
            for perm in perm_order:
                if perm not in data_bd or len(data_bd[perm]) == 0:
                    continue
                pu.draw_profile_curve(ax_bd, data_bd[perm], n_matrices_bd, perm,
                                    xlim_lo_bd, xlim_hi_bd, higher_is_better=True)
            ax_bd.set_xlabel(f'{get_metric_display(bd_metric)} relative to the best')
            ax_bd.set_ylabel('Fraction of matrices')
            ax_bd.set_xscale('log', base=2)
            ax_bd.set_ylim(0, 1.05)
            ax_bd.set_xlim(xlim_lo_bd, xlim_hi_bd)
            ax_bd.invert_xaxis()
            ax_bd.grid(True, alpha=0.3)
            handles_bd, labels_bd = ax_bd.get_legend_handles_labels()

            fig_bd.tight_layout()
            for ext in ('pdf', 'png'):
                fig_bd.savefig(prof_dir / f'profile_block_density_16.{ext}',
                               bbox_inches='tight', dpi=150)
            plt.close(fig_bd)
            print(f"  Saved profile_block_density_16.pdf/png")

            # --- Standalone legend ---
            ncol_leg = (len(handles_bd) + 1) // 2
            fig_leg = plt.figure(figsize=(6, 1.2))
            fig_leg.legend(handles_bd, labels_bd, loc='center',
                           ncol=ncol_leg, fontsize=9, frameon=False)
            for ext in ('pdf', 'png'):
                fig_leg.savefig(prof_dir / f'profile_legend.{ext}',
                                bbox_inches='tight', dpi=150)
            plt.close(fig_leg)
            print(f"  Saved profile_legend.pdf/png")


def generate_profile_aggregate_boxplot(df_analysis, out_dir):
    """Generate aggregated boxplot of improvement ratios across profile metrics.

    For each reordering algorithm and each of the 6 profile metrics, shows
    a boxplot of the improvement ratio (reordered / original, direction-aware)
    across all matrices. Algorithms on x-axis, 6 colored boxes per algorithm.
    """
    print("\n=== Aggregated Profile Improvement Boxplot ===")

    profile_metrics = [
        'bandwidth_max',
        'locality_avg_col_spread',
        'locality_vertical_adjacency_ratio',
        'locality_profile',
        'access_dist_reuse_distance_mean',
        'access_dist_index_distance_mean',
        'block_density_16',
    ]
    profile_metrics = [m for m in profile_metrics
                       if m in df_analysis.columns
                       and ALL_METRICS.get(m, {}).get('higher_is_better') is not None]
    if not profile_metrics:
        print("  No usable metrics — skipping aggregate boxplot.")
        return

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

    # Compute improvement ratios for each metric
    metrics_config = {
        'bandwidth_max': {'improvement_name': 'bandwidth_improvement', 'higher_is_better': False},
        'locality_avg_col_spread': {'improvement_name': 'col_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
        'locality_profile': {'improvement_name': 'profile_improvement', 'higher_is_better': False},
        'access_dist_reuse_distance_mean': {'improvement_name': 'reuse_distance_improvement', 'higher_is_better': False},
        'access_dist_index_distance_mean': {'improvement_name': 'index_distance_improvement', 'higher_is_better': False},
        'block_density_16': {'improvement_name': 'density_improvement_16', 'higher_is_better': True},
    }
    metrics_config = {k: v for k, v in metrics_config.items() if k in profile_metrics}
    df = pu.compute_improvements(df, metrics_config)

    # Exclude Original (improvement = 1 by definition)
    df = df[df['strategy'] != 'Original']

    # Melt into long form: one row per (matrix, strategy, metric)
    imp_cols = [v['improvement_name'] for v in metrics_config.values()]
    imp_display = {}
    for m, cfg in metrics_config.items():
        display = get_metric_display(m)
        # Shorten for readability
        for suffix in [' Improvement', ' Reduction']:
            display = display.replace(suffix, '')
        imp_display[cfg['improvement_name']] = display

    id_cols = ['matrix', 'strategy']
    df_long = df[id_cols + imp_cols].melt(
        id_vars=id_cols, value_vars=imp_cols,
        var_name='metric', value_name='improvement'
    )
    df_long['metric_display'] = df_long['metric'].map(imp_display)
    df_long = df_long.dropna(subset=['improvement'])
    # Remove inf / extreme outliers
    df_long = df_long[np.isfinite(df_long['improvement'])]

    if df_long.empty:
        print("  No improvement data — skipping aggregate boxplot.")
        return

    # Strategy order & palette (exclude Original) — same as aggregate speedup
    strat_order = [get_perm_display(p) for p in PERMS if p != 'None']
    strat_order = [s for s in strat_order if s in df_long['strategy'].unique()]
    strat_palette = pu.get_strategy_palette()
    strat_palette = {k: v for k, v in strat_palette.items() if k != 'Original'}

    metric_order = [imp_display[c] for c in imp_cols if c in imp_display]

    import seaborn as sns
    agg_dir = out_dir / 'reorder_analysis' / 'aggregate_improvement'
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Metrics on x-axis, reordering algorithms as hue (mirrors aggregate speedup by kernel)
    n_metrics = len(metric_order)
    fig, ax = plt.subplots(figsize=(max(14, n_metrics * 2.0), 6))

    sns.boxplot(
        data=df_long, x='metric_display', y='improvement', hue='strategy',
        order=metric_order, hue_order=strat_order, palette=strat_palette,
        ax=ax, fliersize=1.5, linewidth=0.8,
        whis=(5, 95), showfliers=False,
        medianprops=dict(color='red', linewidth=0.8),
    )

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('Improvement ratio')

    ax.set_yscale('log')
    p05 = df_long['improvement'].quantile(0.02)
    p95 = df_long['improvement'].quantile(0.98)
    ax.set_ylim(p05 / 1.3, p95 * 1.3)
    pu.format_log_axes(ax, which='y')

    ax.legend(title=None, bbox_to_anchor=(0.5, 1.02), loc='lower center',
              ncol=len(strat_order), fontsize=10, frameon=False)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(agg_dir / f'aggregate_improvement_boxplot.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved aggregate_improvement_boxplot.pdf/png")


def generate_speedup_aggregate_boxplot(df, out_dir, args=None):
    """Generate aggregated boxplot of kernel speedups by reordering algorithm.

    Kernels on x-axis, reordering algorithms as hue.
    One figure per n_cols value.
    """
    print("\n=== Aggregated Speedup Boxplot (by kernel) ===")

    if 'speedup' not in df.columns:
        print("  No speedup column — skipping aggregate speedup boxplot.")
        return

    import seaborn as sns

    # Exclude Original and cuSparse BSR (block-format outlier)
    df_reord = df[(df['strategy'] != 'Original')
                  & (df['kernel_id'] != 'CUSPARSE_SPMM_BSR_bs32')].copy()

    # Strategy order
    strat_order = [get_perm_display(p) for p in PERMS if p != 'None']
    strat_order = [s for s in strat_order if s in df_reord['strategy'].unique()]

    # Kernel order and display names
    kernels = sorted(df_reord['kernel_id'].unique())
    kernel_display = {k: KERNEL_NAMES.get(k, k) for k in kernels}
    kernel_names_ordered = [kernel_display[k] for k in kernels]

    agg_dir = out_dir / 'aggregate_speedup'
    agg_dir.mkdir(parents=True, exist_ok=True)
    strat_palette = pu.get_strategy_palette()
    # Remove Original from palette
    strat_palette = {k: v for k, v in strat_palette.items() if k != 'Original'}

    for nc in sorted(df_reord['n_cols'].unique()):
        df_nc = df_reord[df_reord['n_cols'] == nc].copy()
        df_nc['kernel_display'] = df_nc['kernel_id'].map(kernel_display)

        df_nc = df_nc.dropna(subset=['speedup'])
        df_nc = df_nc[np.isfinite(df_nc['speedup'])]
        if df_nc.empty:
            continue

        n_kernels = len(kernel_names_ordered)
        fig, ax = plt.subplots(figsize=(max(14, n_kernels * 2.0), 6))

        sns.boxplot(
            data=df_nc, x='kernel_display', y='speedup', hue='strategy',
            order=kernel_names_ordered, hue_order=strat_order,
            palette=strat_palette,
            ax=ax, fliersize=1.5, linewidth=0.8,
            whis=(5, 95), showfliers=False,
            medianprops=dict(color='red', linewidth=0.8),
        )

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('')
        ax.set_ylabel('Speedup')

        ax.set_yscale('log')
        p05 = df_nc['speedup'].quantile(0.02)
        p95 = df_nc['speedup'].quantile(0.98)
        ax.set_ylim(p05 / 1.3, p95 * 1.3)
        pu.format_log_axes(ax, which='y')

        ax.legend(title=None, bbox_to_anchor=(0.5, 1.02), loc='lower center',
                  ncol=len(strat_order), fontsize=10, frameon=False)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()

        for ext in ('pdf', 'png'):
            fig.savefig(agg_dir / f'aggregate_speedup_by_kernel_nc{int(nc)}.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved aggregate_speedup_by_kernel_nc{int(nc)}.pdf/png")


def generate_speedup_profile_plots(df, out_dir, args=None):
    """Generate Dolan-Moré performance-profile plots of kernel speedups.

    For each kernel and n_cols, compute the performance ratio of each
    strategy's speedup relative to the best strategy on each matrix:
        r_{m,s} = speedup_{m,s} / max_{s'} speedup_{m,s'}   ∈ (0, 1]

    Then plot the survival function P(ratio >= τ) vs τ.
    Higher curve = better strategy.  Original (speedup=1) is included.

    One grid figure per n_cols, with one subplot per kernel.
    """
    if 'speedup' not in df.columns:
        print("  No speedup column — skipping speedup profile plots.")
        return

    prof_dir = out_dir / 'speedup_profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)

    perm_order = pu.profile_perm_order(set(df['perm'].unique()))
    strategy_order = [get_perm_display(p) for p in perm_order]

    for nc in sorted(df['n_cols'].unique()):
        df_nc = df[df['n_cols'] == nc]
        kernel_ids = sorted(k for k in df_nc['kernel_id'].unique()
                            if k != 'ASPT_SPMM')
        if not kernel_ids:
            continue

        ncols_grid = min(3, len(kernel_ids))
        nrows_grid = (len(kernel_ids) + ncols_grid - 1) // ncols_grid
        fig, axes = plt.subplots(nrows_grid, ncols_grid,
                                 figsize=(5 * ncols_grid, 4 * nrows_grid),
                                 squeeze=False)

        for idx, kid in enumerate(kernel_ids):
            ax = axes[idx // ncols_grid][idx % ncols_grid]
            df_k = df_nc[df_nc['kernel_id'] == kid].copy()

            # Drop NaN / non-positive speedups
            df_k = df_k.dropna(subset=['speedup'])
            df_k = df_k[df_k['speedup'] > 0]

            if df_k.empty:
                ax.set_visible(False)
                continue

            # Keep only matrices with all strategies present
            all_strats = df_k['perm'].unique()
            n_strats = len(all_strats)
            strat_counts = df_k.groupby('matrix')['perm'].nunique()
            complete = strat_counts[strat_counts == n_strats].index
            df_k = df_k[df_k['matrix'].isin(complete)]

            if df_k.empty:
                ax.set_visible(False)
                continue

            # Compute ratio to best
            best = df_k.groupby('matrix')['speedup'].max().rename('best')
            df_k = df_k.merge(best, on='matrix')
            df_k['tau'] = df_k['speedup'] / df_k['best']

            n_matrices = df_k['matrix'].nunique()

            for perm in perm_order:
                sub = df_k[df_k['perm'] == perm]
                if sub.empty:
                    continue
                taus_asc = np.sort(sub['tau'].values)
                pu.draw_profile_curve(ax, taus_asc, n_matrices, perm,
                                    xlim_lo=-0.02, xlim_hi=1.05,
                                    higher_is_better=True)

            display_name = KERNEL_NAMES.get(kid, kid)
            ax.set_title(display_name, fontsize=10)
            ax.set_xlabel('Speedup relative to the best')
            ax.set_ylabel('Fraction of matrices')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-0.02, 1.05)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(len(kernel_ids), nrows_grid * ncols_grid):
            axes[idx // ncols_grid][idx % ncols_grid].set_visible(False)

        # Shared legend — collect from all subplots to catch all strategies
        all_handles, all_labels = [], []
        seen = set()
        for ax_row in axes:
            for ax in ax_row:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l not in seen:
                        seen.add(l)
                        all_handles.append(h)
                        all_labels.append(l)
        # Sort by canonical strategy order
        label_order = {s: i for i, s in enumerate(strategy_order)}
        pairs = sorted(zip(all_handles, all_labels),
                       key=lambda p: label_order.get(p[1], 999))
        if pairs:
            all_handles, all_labels = zip(*pairs)
        fig.legend(all_handles, all_labels, loc='upper center',
                   ncol=len(all_handles), fontsize=9,
                   bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout()

        for ext in ('pdf', 'png'):
            fig.savefig(prof_dir / f'speedup_profiles_nc{nc}.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved speedup_profiles_nc{nc}.pdf/png")


def generate_imp_correlation_plots(df, out_dir):
    """Generate grouped bar charts of improvement-metric correlations.

    Uses the correlation method from config (pearson/spearman/kendall).
    Generates both linear and log-log variants.
    One image per (scale, n_cols) combination.
    x-axis: kernels, y-axis: correlation value, one bar per metric.
    """
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    metric_order = [get_metric_display(m) for m in imp_metrics]
    n_metrics = len(metric_order)
    n_kernels = len(kernel_order)

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        scale_label = ' (log-log)' if log_transform else ''

        for n_cols in n_cols_values:
            corr_df = compute_imp_correlations(
                df, n_cols, imp_metrics, kernels,
                method=method, log_transform=log_transform)

            fig, ax = plt.subplots(figsize=(12, 6))

            bar_width = 0.7 / n_metrics
            x = np.arange(n_kernels)

            for mi, mc in enumerate(imp_metrics):
                col = f'{mc}_corr'
                display = get_metric_display(mc)
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
                        vals.append(0)
                    else:
                        vals.append(row.iloc[0][col])
                offset = (mi - (n_metrics - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       label=display, color=get_metric_color(mc),
                       edgecolor='black', linewidth=0.4)

            ax.axhline(0, color='grey', linestyle='--', alpha=0.5,
                       linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel('Correlation with Speedup')
            ax.legend(title='Metric', fontsize=10, title_fontsize=11)
            ax.grid(True, axis='y', alpha=0.3)

            fname = (f'imp_correlation_bars_{tag}{scale_suffix}'
                     f'_ncols_{int(n_cols)}.png')
            out_path = plot_dir / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  Saved: {out_path}")


def generate_imp_ncols_correlation_plots(df, out_dir):
    """Generate per-metric bar charts showing how improvement–speedup correlation varies with n_cols.

    One image per (metric, scale) combination.
    x-axis: kernels, grouped bars: one per n_cols value, y-axis: correlation.
    """
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    n_kernels = len(kernel_order)
    n_ncols = len(n_cols_values)

    # Teal/green shades with distinct hatches
    _teal_shades = ['#b2dfdb', '#4db6ac', '#00897b', '#004d40']
    _hatches = ['', '///', '...', 'xxx', '\\\\\\', '+++']
    ncols_colors = dict(zip(n_cols_values, _teal_shades[:n_ncols]))
    ncols_hatches = dict(zip(n_cols_values, _hatches[:n_ncols]))

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        scale_label = ' (log-log)' if log_transform else ''

        # Pre-compute correlations for all n_cols values
        corr_by_ncols = {}
        for n_cols in n_cols_values:
            corr_by_ncols[n_cols] = compute_imp_correlations(
                df, n_cols, imp_metrics, kernels,
                method=method, log_transform=log_transform)

        for mc in imp_metrics:
            col = f'{mc}_corr'
            metric_display = get_metric_display(mc)
            metric_safe = pu.safe_filename(mc)

            fig, ax = plt.subplots(figsize=(12, 6))

            bar_width = 0.7 / n_ncols
            x = np.arange(n_kernels)

            for ni, n_cols in enumerate(n_cols_values):
                corr_df = corr_by_ncols[n_cols]
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
                        vals.append(0)
                    else:
                        vals.append(row.iloc[0][col])
                offset = (ni - (n_ncols - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       label=f'n_cols={int(n_cols)}',
                       color=ncols_colors[n_cols],
                       hatch=ncols_hatches[n_cols],
                       edgecolor='black', linewidth=0.4)

            ax.axhline(0, color='grey', linestyle='--', alpha=0.5,
                       linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel(f'Correlation of {metric_display} with Speedup')
            ax.legend(title='$n_{cols}$', fontsize=10, title_fontsize=11)
            ax.grid(True, axis='y', alpha=0.3)

            fname = (f'imp_ncols_bars_{tag}{scale_suffix}_{metric_safe}.png')
            out_path = plot_dir / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  Saved: {out_path}")


def generate_imp_blocksize_correlation_plots(df, out_dir):
    """Generate per-kernel bar charts showing density-improvement–speedup correlation across block sizes.

    One image per (kernel, scale) combination.
    x-axis: block sizes, y-axis: correlation value.
    Mirrors the LaTeX tables from generate_imp_blocksize_tables in correlation_table.py.
    """
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    bd_imp_metrics = _density_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    n_kernels = len(kernel_order)
    n_block_sizes = len(bd_imp_metrics)
    block_labels = [f'{bs}' for bs in BLOCK_SIZES[:n_block_sizes]]

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        scale_label = ' (log-log)' if log_transform else ''

        for n_cols in n_cols_values:
            corr_df = compute_imp_correlations(
                df, n_cols, bd_imp_metrics, kernels,
                method=method, log_transform=log_transform)

            fig, ax = plt.subplots(figsize=(12, 6))

            bar_width = 0.7 / n_block_sizes
            x = np.arange(n_kernels)

            for bi, (metric, label) in enumerate(zip(bd_imp_metrics, block_labels)):
                col = f'{metric}_corr'
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
                        vals.append(0)
                    else:
                        vals.append(row.iloc[0][col])
                offset = (bi - (n_block_sizes - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       label=f'{label}×{label}',
                       color=get_metric_color(metric),
                       edgecolor='black', linewidth=0.4)

            ax.axhline(0, color='grey', linestyle='--', alpha=0.5,
                       linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel('Correlation of Block Density Improvement with Speedup')
            ax.legend(title='Block Size', fontsize=10, title_fontsize=11)
            ax.grid(True, axis='y', alpha=0.3)

            fname = (f'imp_blocksize_bars_{tag}{scale_suffix}'
                     f'_ncols_{int(n_cols)}.png')
            out_path = plot_dir / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  Saved: {out_path}")


def generate_partial_correlation_heatmaps(df, out_dir):
    """Generate partial-correlation heatmaps: one per (kernel, n_cols).

    Each heatmap is a metric×metric matrix where cell [i,j] shows
    r(speedup, metric_i | metric_j).  Diagonal = marginal correlation.
    Reveals which metrics are redundant vs. independently predictive.
    """
    import seaborn as sns
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    def _clean_label(m):
        s = get_metric_display(m)
        for suffix in [' Improvement', ' Reduction']:
            s = s.replace(suffix, '')
        return s
    display_labels = [_clean_label(m) for m in imp_metrics]

    plot_dir = Path(out_dir) / 'partial_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for n_cols in n_cols_values:
        n_cols_int = int(n_cols)

        for kernel in kernels:
            mat = pu.compute_partial_imp_matrix(
                df, n_cols, imp_metrics, kernel, method=method)

            if mat.isna().all().all():
                continue

            # Relabel for display
            mat.index = display_labels
            mat.columns = display_labels

            kernel_display = KERNEL_NAMES.get(kernel, kernel)
            n = len(imp_metrics)
            cell_size = max(0.9, min(1.2, 8 / n))
            figsize = (n * cell_size + 2.5, n * cell_size + 1.5)
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                mat.abs().astype(float), annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=0.7,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': f'|Partial {corr_display}|',
                          'shrink': 0.8},
                annot_kws={'size': 9},
            )

            ax.set_title(
                f'{kernel_display} — Partial Correlation with Speedup\n'
                f'|r(speedup, row | col)|   n_cols={n_cols_int}',
                fontsize=12, pad=12)
            ax.set_xlabel('Controlled metric')
            ax.set_ylabel('Target metric')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='y', rotation=0)

            kernel_safe = pu.safe_filename(kernel)
            fname = f'partial_corr_{tag}_{kernel_safe}_ncols_{n_cols_int}.png'
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(plot_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_dir / fname}")

        # Also generate a summary heatmap: mean |partial r| across kernels
        # to show which metrics are globally redundant
        all_mats = []
        for kernel in kernels:
            mat = pu.compute_partial_imp_matrix(
                df, n_cols, imp_metrics, kernel, method=method)
            if not mat.isna().all().all():
                all_mats.append(mat)

        if len(all_mats) >= 2:
            mean_mat = pd.concat(all_mats).groupby(level=0).mean()
            mean_mat = mean_mat.reindex(index=imp_metrics, columns=imp_metrics)
            mean_mat.index = display_labels
            mean_mat.columns = display_labels

            n = len(imp_metrics)
            cell_size = max(0.9, min(1.2, 8 / n))
            figsize = (n * cell_size + 2.5, n * cell_size + 1.5)
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                mean_mat.abs().astype(float), annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=0.7,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': f'Mean |Partial {corr_display}|',
                          'shrink': 0.8},
                annot_kws={'size': 9},
            )

            ax.set_title(
                f'Mean Partial Correlation with Speedup (all kernels)\n'
                f'|r(speedup, row | col)|   n_cols={n_cols_int}',
                fontsize=12, pad=12)
            ax.set_xlabel('Controlled metric')
            ax.set_ylabel('Target metric')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='y', rotation=0)

            fname = f'partial_corr_{tag}_mean_ncols_{n_cols_int}.png'
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(plot_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_dir / fname}")


def generate_feature_importance_plots(df, out_dir):
    """Random Forest feature importance: which structural improvement metrics
    best predict speedup, per (kernel, n_cols).

    Produces per n_cols:
      feature_importance/rf_importance_heatmap_ncols_{N}.png  — metric × kernel heatmap
      feature_importance/rf_importance_bars_ncols_{N}.png     — grouped bar chart
    """
    from sklearn.ensemble import RandomForestRegressor
    import seaborn as sns

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    kernel_displays = [KERNEL_NAMES.get(k, k) for k in kernels]
    n_cols_values = sorted(df['n_cols'].unique())
    n_metrics = len(imp_metrics)
    n_kernels = len(kernel_displays)

    def _clean(m):
        s = get_metric_display(m)
        for suffix in [' Improvement', ' Reduction']:
            s = s.replace(suffix, '')
        return s

    display_labels = [_clean(m) for m in imp_metrics]
    metric_order = [get_metric_display(m) for m in imp_metrics]

    plot_dir = Path(out_dir) / 'feature_importance'
    plot_dir.mkdir(parents=True, exist_ok=True)

    MIN_SAMPLES = 30

    def _fit_rf(sub, n_trees=300):
        X = sub[imp_metrics].values
        y = sub['speedup'].values
        rf = RandomForestRegressor(n_estimators=n_trees, random_state=42,
                                   min_samples_leaf=5, n_jobs=-1)
        rf.fit(X, y)
        return rf.feature_importances_

    for n_cols in n_cols_values:
        nc_int = int(n_cols)
        df_nc = df[df['n_cols'] == n_cols]
        imp_mat = pd.DataFrame(np.nan, index=display_labels,
                               columns=kernel_displays)

        for kernel, kdisp in zip(kernels, kernel_displays):
            sub = df_nc[(df_nc['kernel_id'] == kernel) &
                        (df_nc['strategy'] != 'Original')
                        ][imp_metrics + ['speedup']].dropna()
            sub = sub[np.isfinite(sub).all(axis=1)]
            if len(sub) < MIN_SAMPLES:
                print(f"  Skipping {kdisp} n_cols={nc_int}: {len(sub)} samples")
                continue
            imp_mat[kdisp] = _fit_rf(sub)

        valid = imp_mat.dropna(axis=1, how='all')
        if valid.empty:
            continue

        # ---- Heatmap ------------------------------------------------- #
        fig, ax = plt.subplots(
            figsize=(max(5, valid.shape[1] * 1.3 + 2),
                     max(4, valid.shape[0] * 0.5 + 1.5)))
        sns.heatmap(valid.astype(float), annot=True, fmt='.3f',
                    cmap='Blues', vmin=0,
                    ax=ax, linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Feature Importance', 'shrink': 0.8},
                    annot_kws={'size': 8})
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        fname = plot_dir / f'rf_importance_heatmap_ncols_{nc_int}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")

        # ---- Grouped bar chart --------------------------------------- #
        valid_kernels = list(valid.columns)
        n_valid = len(valid_kernels)
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.7 / n_metrics
        x = np.arange(n_valid)

        for mi, (mc, dlabel, mdisplay) in enumerate(
                zip(imp_metrics, display_labels, metric_order)):
            vals = [valid.loc[dlabel, k] if dlabel in valid.index else 0.0
                    for k in valid_kernels]
            offset = (mi - (n_metrics - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width * 0.9,
                   label=mdisplay, color=get_metric_color(mc),
                   hatch=get_metric_hatch(mc),
                   edgecolor='black', linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_kernels, rotation=30, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('Feature Importance')
        ax.legend(title='Metric', fontsize=10, title_fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        fname = plot_dir / f'rf_importance_bars_ncols_{nc_int}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")


def _should_run(section: str, args) -> bool:
    """Decide whether *section* should run given CLI flags.

    Priority: --sections (fine-grained) > --only-* (coarse) > default (all).
    """
    if args.sections is not None:
        return section in args.sections
    # Legacy coarse flags
    kernel_sections = {'grouped-scatter', 'breakeven', 'speedup-profiles'}
    reorder_sections = {'reorder-analysis', 'profiles'}
    if args.only_kernels:
        return section in kernel_sections
    if args.only_reorder_analysis:
        return section in reorder_sections
    return True  # no filter → run everything


def main():
    args = parse_args()

    # Resolve perm_type pipeline
    perm_type_filter = 'ROW' if args.row else 'SYMMETRIC'
    pipeline_key = ('random_' if args.random else '') + perm_type_filter.lower()

    # Apply --random defaults (before any other processing)
    if args.out is None:
        data_label = 'random' if args.random else 'original'
        perm_label = 'row' if args.row else 'symmetric'
        args.out = f"plots/{data_label}_{perm_label}"

    # Validate mutually exclusive options (only matters for legacy flags)
    if args.sections is None and args.only_reorder_analysis and args.only_kernels:
        print("Error: --only-reorder-analysis and --only-kernels are mutually exclusive.")
        return

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set professional publication style
    pu.set_professional_style()
    
    # -----------------------------------------------------------------
    # 1. Load & Filter Data (single pipeline from filter_config.yaml)
    # -----------------------------------------------------------------
    cli_overrides = {
        'operations_csv': args.operations,
        'analysis_csv': args.analysis,
        'matrices_list': args.matrices_list,
        'one_per_family': args.one_per_family,
        'min_size': args.min_size,
        'max_size': args.max_size,
        'random': args.random,
    }
    # --include-rectangular means square_only=False
    if args.include_rectangular is not None:
        cli_overrides['square_only'] = not args.include_rectangular

    df, df_analysis, _cfg = pu.load_and_filter_data(
        config_path=args.filter_config,
        cli_overrides=cli_overrides,
    )

    # Split by perm_type — downstream code works on a single homogeneous DF
    pipeline_cfg = _cfg.get('pipelines', {}).get(pipeline_key, {})
    df, df_analysis = pu.split_by_perm_type(
        df, df_analysis, perm_type_filter, pipeline_cfg)

    # -----------------------------------------------------------------
    # 2. Process Data (add all derived columns)
    # -----------------------------------------------------------------
    has_ops = not df.empty
    if has_ops:
        print("\nProcessing data...")
        df = pu.prepare_full_dataframe(df)
        print(f"Unique kernels: {df['kernel_id'].unique().tolist()}")
        print(f"Unique n_cols: {sorted(df['n_cols'].unique())}")
    else:
        print("\nNo operations data — kernel/breakeven/original-scatter plots will be skipped.")

    # Helper for reordering CSV path (shared by breakeven & timing)
    def _reordering_csv():
        return _cfg.get('data', {}).get('reordering_csv',
                                        'results/results_reordering.csv')

    # -----------------------------------------------------------------
    # 3. Generate Plots  (each guarded by _should_run)
    # -----------------------------------------------------------------

    if has_ops and _should_run('grouped-scatter', args):
        print("\n" + "="*60)
        print("Generating grouped improvement vs speedup scatter plots...")
        print("="*60)
        generate_grouped_scatter_plots(df, out_dir, args)

    if has_ops and _should_run('breakeven', args):
        print("\n" + "="*60)
        print("Generating break-even analysis plots...")
        print("="*60)
        generate_breakeven_plots(df, out_dir, reordering_csv=_reordering_csv(),
                                 args=args)

    if _should_run('reorder-analysis', args):
        print("\n" + "="*60)
        print("Generating reordering analysis plots...")
        print("="*60)
        generate_reorder_analysis_plots(df_analysis, out_dir, n_jobs=args.jobs)
        
    if _should_run('profiles', args):
        print("\n" + "="*60)
        print("Generating performance profile plots...")
        print("="*60)
        generate_profile_plots(df_analysis, out_dir)

    if _should_run('aggregate-improvement', args):
        print("\n" + "="*60)
        print("Generating aggregate improvement boxplot...")
        print("="*60)
        generate_profile_aggregate_boxplot(df_analysis, out_dir)
        if has_ops:
            generate_speedup_aggregate_boxplot(df, out_dir, args)

    if has_ops and _should_run('speedup-profiles', args):
        print("\n" + "="*60)
        print("Generating speedup performance profile plots...")
        print("="*60)
        generate_speedup_profile_plots(df, out_dir, args)

    if has_ops and _should_run('imp-correlation', args):
        print("\n" + "="*60)
        print("Generating improvement-correlation bar charts...")
        print("="*60)
        generate_imp_correlation_plots(df, out_dir)
        generate_imp_ncols_correlation_plots(df, out_dir)
        generate_imp_blocksize_correlation_plots(df, out_dir)

    if has_ops and _should_run('partial-correlation', args):
        print("\n" + "="*60)
        print("Generating partial-correlation heatmaps...")
        print("="*60)
        generate_partial_correlation_heatmaps(df, out_dir)

    if has_ops and _should_run('feature-importance', args):
        print("\n" + "="*60)
        print("Generating RF feature importance plots...")
        print("="*60)
        generate_feature_importance_plots(df, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
