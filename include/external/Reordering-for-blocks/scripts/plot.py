"""
Plot Script - Clean Architecture (Tufte-styled)

Simple iteration-based plotting script that:
1. Loads and processes data once
2. Applies filters
3. Iterates over dimensions (n_cols, kernel) to generate plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd
import numpy as np
import plot_utils as pu
from multiprocessing import cpu_count
from settings import (
    get_perm_display, KERNEL_NAMES, GROUPED_SCATTER_EXCLUDE, PERMS,
    ALL_METRICS, BLOCK_SIZES, get_metric_display, get_metric_color,
    get_metric_hatch,
)
from correlation_table import (
    compute_imp_correlations,
    _enabled_improvement_metrics,
    _density_improvement_metrics,
    _corr_method_tag,
    _ordered_kernels,
)

# Axis caps for ratio scatter plots (speedup, improvement)
RATIO_YLIM = (1 / 7, 7)
RATIO_XLIM = (0.05, 10)  # x-axis bounds for improvement ratios


# ---------------------------------------------------------------------------
# Tufte Styling Helpers
# ---------------------------------------------------------------------------

def set_tufte_rcparams():
    """Apply clean, high data-ink-ratio matplotlib defaults on top of plot_utils."""
    plt.rcParams.update({
        'axes.edgecolor': '#444444',
        'axes.linewidth': 0.7,
        'axes.labelcolor': '#222222',
        'axes.titlecolor': '#111111',
        'axes.titlepad': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': '#444444',
        'ytick.color': '#444444',
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'grid.color': '#888888',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.2,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
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


def _format_log10_ratio_axis(ax, log_lo, log_hi):
    """Format a log10-transformed ratio y-axis with intuitive linear ticks (e.g. 0.5x, 1x, 2x)."""
    candidates = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    ticks = [t for t in candidates if log_lo <= np.log10(t) <= log_hi]
    if len(ticks) < 3:
        # Fallback to powers of 10
        p_lo, p_hi = int(np.floor(log_lo)), int(np.ceil(log_hi))
        ticks = [10 ** p for p in range(p_lo, p_hi + 1)]
    ax.set_yticks([np.log10(t) for t in ticks])
    ax.set_yticklabels([f'{t:g}×' for t in ticks])


def _draw_tufte_violin(ax, data, x, y_log, hue, order, hue_order, palette):
    """Draw clipped, log-space violins with embedded median/IQR and muted fills."""
    import seaborn as sns
    sns.violinplot(
        data=data, x=x, y=y_log, hue=hue,
        order=order, hue_order=hue_order, palette=palette,
        ax=ax, cut=0, inner='box', linewidth=0.65, saturation=0.85,
    )
    for coll in ax.collections:
        if isinstance(coll, PolyCollection):
            coll.set_alpha(0.72)
            coll.set_edgecolor('white')
            coll.set_linewidth(0.5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument("--asymmetric", action="store_true",
                        help="Use ASYMMETRIC perm_type pipeline (two-sided perms, "
                             "e.g. CLUB_jaccard; output to plots_asymmetric/ or "
                             "plots_random_asymmetric/)")

    # Output
    parser.add_argument("--out", default=None, help="Output directory (default: plots or plots_random with --random)")

    # Filter config
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

    # Fine-grained section selection
    SECTION_CHOICES = [
        'grouped-scatter',
        'breakeven',
        'reorder-analysis',
        'pareto',
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
        "-j", "--jobs", type=int, default=min(8, cpu_count()),
        help="Number of parallel workers for plot generation (default: min(cpu_count, 8), 1=sequential)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plot Generators
# ---------------------------------------------------------------------------

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
                    mark_failures=True,
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
    """Compute and plot break-even number of operations for each reordering."""
    print("\n=== Break-even Analysis ===")

    reorder_csv_path = Path(reordering_csv)
    if not reorder_csv_path.exists():
        print(f"  Reordering CSV not found at {reorder_csv_path} — skipping break-even plots. "
              "Run  python scripts/parse_results.py  first to generate it.")
        return

    df_time = pd.read_csv(reorder_csv_path)
    if df_time.empty or 'time_reordering_ms' not in df_time.columns:
        print("  No reordering timing data available — skipping.")
        return

    df_time = df_time[['matrix', 'perm', 'time_reordering_ms']].copy()
    df_time = df_time.groupby(['matrix', 'perm'], as_index=False)['time_reordering_ms'].mean()
    print(f"  Loaded {len(df_time)} reordering timing entries")

    df_baseline = (
        df[df['strategy'] == 'Original']
        .groupby(['matrix', 'kernel_id', 'n_cols'])['time_operation_ms']
        .mean()
        .reset_index()
        .rename(columns={'time_operation_ms': 'time_baseline_ms'})
    )

    df_reord = df[df['strategy'] != 'Original'].copy()
    if df_reord.empty:
        print("  No reordered operation data — skipping.")
        return

    df_reord = df_reord.merge(df_baseline, on=['matrix', 'kernel_id', 'n_cols'], how='inner')
    df_reord = df_reord.merge(df_time, on=['matrix', 'perm'], how='inner')

    if df_reord.empty:
        print("  No overlapping data between operations and reordering timing — skipping.")
        return

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

    n_cols_values = sorted(df_reord['n_cols'].unique())
    if args is not None and args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"  Warning: n_cols={args.n_cols} not in data; available: {n_cols_values}")
            return

    kernel_filter = args.kernel if args is not None else None
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
                output_path=breakeven_dir / "breakeven.png",
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

    df = df_analysis.copy()
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

    metrics_config = pu.build_metrics_config(df, include_extended=True)
    df = pu.compute_improvements(df, metrics_config)
    df_reordered = df[df['strategy'] != 'Original']

    if df_reordered.empty:
        print("No reordered data for analysis plots")
        return

    strategies = sorted(df_reordered['strategy'].unique())

    boxplot_specs = [
        (f'density_improvement_{bs}', f"Density Improvement (BS {bs})",
         "density", f"density_improvement_bs{bs}.png")
        for bs in [4, 8, 16, 32, 64, 128]
    ]

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


def generate_pareto_plots(df_analysis, out_dir, reordering_csv='results/results_reordering.csv', args=None):
    """Pareto frontier: geometric mean density improvement vs reordering time."""
    print("\n=== Pareto Frontier (quality vs time) ===")

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)
    df['matrix'] = df['matrix'].astype(str)

    bd_cols = [c for c in df.columns if c.startswith('block_density_')
               and c.split('_')[-1].isdigit()]
    if not bd_cols:
        print("  No block_density_* columns — skipping pareto plots.")
        return

    reorder_csv_path = Path(reordering_csv)
    if reorder_csv_path.exists():
        df_time = pd.read_csv(reorder_csv_path)
        if df_time.empty or 'time_reordering_ms' not in df_time.columns:
            print("  No reordering timing data available — skipping.")
            return
        df_time = df_time[['matrix', 'perm', 'time_reordering_ms']].copy()
        df_time['perm'] = df_time['perm'].astype(str)
        df_time['matrix'] = df_time['matrix'].astype(str)
        df_time = df_time.groupby(['matrix', 'perm'], as_index=False)['time_reordering_ms'].mean()
    elif 'time_reordering_ms' in df.columns:
        df_time = df[['matrix', 'perm', 'time_reordering_ms']].copy()
        df_time = df_time.groupby(['matrix', 'perm'], as_index=False)['time_reordering_ms'].mean()
    else:
        print(f"  No timing at {reorder_csv_path} and no timing column — skipping.")
        return

    all_perms = sorted(df['perm'].unique())
    if 'None' not in all_perms:
        print("  No 'None' baseline in analysis data — skipping.")
        return

    has_base = df[df['perm'] == 'None']['matrix'].unique()
    n_matrices = len(has_base)
    df = df[df['matrix'].isin(has_base)]
    df_time = df_time[df_time['matrix'].isin(has_base)]
    if df.empty:
        return

    base = (df[df['perm'] == 'None']
            .set_index('matrix')[bd_cols]
            .rename(columns={c: c + '_base' for c in bd_cols}))
    df_reord = df[df['perm'] != 'None'].merge(
        base, left_on='matrix', right_index=True, how='inner')
    if df_reord.empty:
        return

    ratios = np.column_stack([
        (df_reord[c] / df_reord[c + '_base'].replace(0, np.nan)).to_numpy()
        for c in bd_cols
    ])
    valid_ratios = np.where((ratios > 0) & np.isfinite(ratios), ratios, np.nan)
    with np.errstate(invalid='ignore'):
        geomean_imp = np.exp(np.nanmean(np.log(valid_ratios), axis=1))

    df_reord = df_reord.assign(mean_density_imp=geomean_imp)
    df_reord = df_reord.dropna(subset=['mean_density_imp'])
    df_reord = df_reord[np.isfinite(df_reord['mean_density_imp'])]
    df_time = df_time[(df_time['perm'] != 'None') & (df_time['time_reordering_ms'] > 0)]

    imp_stats = (df_reord.groupby('perm')
                 .agg(imp_med=('mean_density_imp', 'median'),
                      imp_q1=('mean_density_imp', lambda s: s.quantile(0.25)),
                      imp_q3=('mean_density_imp', lambda s: s.quantile(0.75)),
                      n_imp=('mean_density_imp', 'size'))
                 .reset_index())
    time_stats = (df_time.groupby('perm')
                  .agg(t_med=('time_reordering_ms', 'median'),
                       t_q1=('time_reordering_ms', lambda s: s.quantile(0.25)),
                       t_q3=('time_reordering_ms', lambda s: s.quantile(0.75)),
                       n_t=('time_reordering_ms', 'size'))
                  .reset_index())

    stats = imp_stats.merge(time_stats, on='perm', how='inner')
    stats = stats[(stats['t_med'] > 0) & np.isfinite(stats['imp_med'])]
    if stats.empty:
        return

    stats['strategy'] = stats['perm'].apply(get_perm_display)
    order = pu.profile_perm_order(set(stats['perm'].unique()))
    stats['pos'] = stats['perm'].apply(lambda p: order.index(p) if p in order else 999)
    stats = stats.sort_values('pos').reset_index(drop=True)

    by_time = stats.sort_values(['t_med', 'imp_med'], ascending=[True, False])
    frontrunners, best_y = set(), -np.inf
    for _, row in by_time.iterrows():
        if row['imp_med'] > best_y + 1e-12:
            frontrunners.add(row['perm'])
            best_y = row['imp_med']

    stats['front'] = stats['perm'].isin(frontrunners)
    stats['coverage'] = np.minimum(stats['n_imp'], stats['n_t']) / float(n_matrices)
    front = stats[stats['front']].sort_values('t_med')

    _pareto_scatter(stats, front, out_dir, n_matrices, show_iqr=False)
    _pareto_scatter(stats, front, out_dir, n_matrices, show_iqr=True)


def _pareto_scatter(stats, front, out_dir, n_matrices, show_iqr=False):
    """Draw a Tufte-inspired Pareto scatter: direct labels, crisp hierarchy, zero chartjunk."""
    pu.set_professional_style()
    set_tufte_rcparams()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    palette = pu.get_strategy_palette(list(stats['strategy'].unique()))

    # Compute generous log bounds so direct labels breathe
    x_col_lo = 't_q1' if show_iqr else 't_med'
    x_col_hi = 't_q3' if show_iqr else 't_med'
    y_col_lo = 'imp_q1' if show_iqr else 'imp_med'
    y_col_hi = 'imp_q3' if show_iqr else 'imp_med'

    x_min = stats[x_col_lo].min() * 0.65
    x_max = stats[x_col_hi].max() * 2.2
    y_min = min(stats[y_col_lo].min(), 1.0) * 0.82
    y_max = stats[y_col_hi].max() * 1.25

    # 1. Subtle 1x baseline with direct right-margin label (no legend needed)
    ax.axhline(1.0, color='#999999', linestyle=':', linewidth=0.9, zorder=1)
    ax.text(x_max * 0.96, 1.0, 'Baseline (1×)', fontsize=8, color='#777777',
            va='bottom', ha='right', style='italic', zorder=2)

    # 2. Clean stepped Pareto frontier (no broken log-scale fill_between)
    if len(front) >= 1:
        xs = list(front['t_med']) + [x_max * 0.92]
        ys = list(front['imp_med']) + [front['imp_med'].iloc[-1]]
        ax.step(xs, ys, where='post', color='#333333', linestyle='--',
                linewidth=1.0, alpha=0.65, zorder=2)
        ax.text(xs[-1], ys[-1], 'Pareto frontier', fontsize=8, color='#444444',
                va='bottom', ha='right', style='italic', zorder=2)

    # 3. Optional capless Tufte IQR crosshairs
    if show_iqr:
        for _, row in stats.iterrows():
            c = palette.get(row['strategy'], '#444444')
            alpha_iqr = 0.45 if row['front'] else 0.22
            ax.errorbar(
                row['t_med'], row['imp_med'],
                xerr=[[max(0.0, row['t_med'] - row['t_q1'])],
                      [max(0.0, row['t_q3'] - row['t_med'])]],
                yerr=[[max(0.0, row['imp_med'] - row['imp_q1'])],
                      [max(0.0, row['imp_q3'] - row['imp_med'])]],
                fmt='none', ecolor=c, elinewidth=0.9, capsize=0,
                alpha=alpha_iqr, zorder=2,
            )

    # 4. Points with clear foreground (Pareto) vs background (dominated) hierarchy
    texts = []
    for _, row in stats.iterrows():
        c = palette.get(row['strategy'], '#333333')
        is_front = bool(row['front'])

        ax.scatter(
            row['t_med'], row['imp_med'],
            s=95 if is_front else 42,
            color=c,
            edgecolors='white' if is_front else 'none',
            linewidths=1.3 if is_front else 0.0,
            alpha=0.95 if is_front else 0.50,
            zorder=5 if is_front else 3,
        )

        n_min = min(int(row['n_imp']), int(row['n_t']))
        label_str = row['strategy']
        if n_min < 0.95 * n_matrices:
            label_str += f" ({n_min}/{n_matrices})"

        txt = ax.text(
            row['t_med'], row['imp_med'], f"  {label_str}",
            fontsize=8.5 if is_front else 7.8,
            fontweight='semibold' if is_front else 'normal',
            color='#111111' if is_front else '#666666',
            va='center', ha='left', zorder=6 if is_front else 4,
        )
        texts.append(txt)

    # 5. Scales, subtle Tufte axes, and minimal directional cue
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    pu.format_log_axes(ax, which='y')
    apply_tufte_style(ax, grid_axis='both', offset_spines=True)

    ax.text(0.01, 0.99, '↖  Faster & higher quality',
            transform=ax.transAxes, fontsize=8.5, color='#555555',
            style='italic', va='top', ha='left')

    ax.set_xlabel('Median Reordering Time (ms, log scale)', fontsize=10, labelpad=8)
    ax.set_ylabel('Median Density Improvement (GeoMean BS 4–128)', fontsize=10, labelpad=8)

    # 6. Auto-repel overlapping labels cleanly
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.5, alpha=0.7),
            expand_points=(1.35, 1.35),
        )
    except ImportError:
        pass

    fig.tight_layout()
    pareto_dir = out_dir / 'pareto'
    pareto_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_iqr' if show_iqr else ''
    for ext in ('pdf', 'png'):
        fig.savefig(pareto_dir / f'pareto_density_vs_time{suffix}.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved pareto_density_vs_time{suffix}.pdf/png")


def generate_profile_plots(df_analysis, out_dir):
    """Generate Tufte-styled Dolan-Moré performance-profile plots."""
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

    base_cols = ['matrix', 'perm', 'perm_type']
    group_df = df_analysis[base_cols + profile_metrics].copy()
    if group_df.empty:
        print("  No data — skipping profile plots.")
        return

    n_matrices_total = group_df['matrix'].nunique()
    print(f"  Using all {n_matrices_total} matrices (pairwise-complete per metric)")

    profile_data = {}
    for m in profile_metrics:
        sub = group_df[['matrix', 'perm', m]].dropna(subset=[m])
        sub = sub[sub[m] > 0]
        if sub.empty:
            profile_data[m] = (0, {})
            continue

        hib = ALL_METRICS[m]['higher_is_better']
        best = (sub.groupby('matrix')[m].max() if hib else sub.groupby('matrix')[m].min()).rename('best')
        sub = sub.merge(best, on='matrix')
        sub['tau'] = sub[m] / sub['best']

        n_matrices = sub['matrix'].nunique()
        perm_ratios = {perm: np.sort(perm_sub['tau'].values)
                       for perm, perm_sub in sub.groupby('perm')}
        profile_data[m] = (n_matrices, perm_ratios)

    n_metrics = len(profile_metrics)
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows),
                             squeeze=False)

    perm_order = pu.profile_perm_order(set(group_df['perm'].unique()))

    for idx, m in enumerate(profile_metrics):
        ax = axes[idx // ncols][idx % ncols]
        n_matrices, data = profile_data[m]
        if n_matrices == 0:
            ax.set_visible(False)
            continue
        hib = ALL_METRICS[m]['higher_is_better']

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

        ax.set_title(get_metric_display(m), fontsize=10, fontweight='semibold', pad=8)
        ax.set_xlabel('Ratio to best', fontsize=9)
        ax.set_ylabel('Fraction of matrices' if (idx % ncols == 0) else '', fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(xlim_lo, xlim_hi)
        if hib:
            ax.invert_xaxis()
        apply_tufte_style(ax, grid_axis='both', offset_spines=True)

    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(6, len(handles)), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout(h_pad=1.6, w_pad=1.4)

    prof_dir = out_dir / 'reorder_analysis' / 'profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(prof_dir / f'profiles.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved profiles.pdf/png")

    # --- Standalone block_density_16 profile plot ---
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
            data_bd = {perm: np.sort(perm_sub['tau'].values)
                       for perm, perm_sub in sub_bd.groupby('perm')}

            fig_bd, ax_bd = plt.subplots(figsize=(5.8, 4.2))
            xlim_lo_bd, xlim_hi_bd = 2**-4, 1.05
            for perm in perm_order:
                if perm not in data_bd or len(data_bd[perm]) == 0:
                    continue
                pu.draw_profile_curve(ax_bd, data_bd[perm], n_matrices_bd, perm,
                                      xlim_lo_bd, xlim_hi_bd, higher_is_better=True)
            ax_bd.set_xlabel(f'{get_metric_display(bd_metric)} relative to best', fontsize=9.5)
            ax_bd.set_ylabel('Fraction of matrices', fontsize=9.5)
            ax_bd.set_xscale('log', base=2)
            ax_bd.set_ylim(0, 1.03)
            ax_bd.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_bd.set_xlim(xlim_lo_bd, xlim_hi_bd)
            ax_bd.invert_xaxis()
            apply_tufte_style(ax_bd, grid_axis='both', offset_spines=True)
            handles_bd, labels_bd = ax_bd.get_legend_handles_labels()

            fig_bd.tight_layout()
            for ext in ('pdf', 'png'):
                fig_bd.savefig(prof_dir / f'profile_block_density_16.{ext}',
                               bbox_inches='tight', dpi=300)
            plt.close(fig_bd)
            print("  Saved profile_block_density_16.pdf/png")

            ncol_leg = (len(handles_bd) + 1) // 2
            fig_leg = plt.figure(figsize=(6, 1.0))
            fig_leg.legend(handles_bd, labels_bd, loc='center',
                           ncol=ncol_leg, fontsize=9, frameon=False)
            for ext in ('pdf', 'png'):
                fig_leg.savefig(prof_dir / f'profile_legend.{ext}',
                                bbox_inches='tight', dpi=300)
            plt.close(fig_leg)
            print("  Saved profile_legend.pdf/png")


def generate_profile_aggregate_boxplot(df_analysis, out_dir):
    """Generate Tufte-styled log-space violin plot of improvement ratios across profile metrics."""
    print("\n=== Aggregated Profile Improvement Violin Plot ===")

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
        print("  No usable metrics — skipping aggregate violin plot.")
        return

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

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
    df = df[df['strategy'] != 'Original']

    imp_cols = [v['improvement_name'] for v in metrics_config.values()]
    imp_display = {}
    for m, cfg in metrics_config.items():
        display = get_metric_display(m)
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
    df_long = df_long[np.isfinite(df_long['improvement']) & (df_long['improvement'] > 0)]

    if df_long.empty:
        print("  No improvement data — skipping aggregate violin plot.")
        return

    # Compute KDE in log10 space so violin density is not distorted by log scaling
    df_long['log_imp'] = np.log10(df_long['improvement'])

    strat_order = [get_perm_display(p) for p in PERMS if p != 'None']
    strat_order = [s for s in strat_order if s in df_long['strategy'].unique()]
    strat_palette = {k: v for k, v in pu.get_strategy_palette().items() if k != 'Original'}
    metric_order = [imp_display[c] for c in imp_cols if c in imp_display]

    agg_dir = out_dir / 'reorder_analysis' / 'aggregate_improvement'
    agg_dir.mkdir(parents=True, exist_ok=True)

    n_metrics = len(metric_order)
    fig, ax = plt.subplots(figsize=(max(13, n_metrics * 1.9), 5.6))

    _draw_tufte_violin(
        ax, df_long, x='metric_display', y_log='log_imp', hue='strategy',
        order=metric_order, hue_order=strat_order, palette=strat_palette,
    )

    ax.axhline(y=0.0, color='#333333', linestyle='--', linewidth=0.8, alpha=0.65, zorder=2)

    p02 = df_long['log_imp'].quantile(0.02)
    p98 = df_long['log_imp'].quantile(0.98)
    pad = max(0.12, (p98 - p02) * 0.08)
    y_lo, y_hi = p02 - pad, p98 + pad
    ax.set_ylim(y_lo, y_hi)
    _format_log10_ratio_axis(ax, y_lo, y_hi)

    ax.set_xlabel('')
    ax.set_ylabel('Improvement ratio (log scale)', fontsize=10, labelpad=8)
    ax.tick_params(axis='x', rotation=25)
    plt.setp(ax.get_xticklabels(), ha='right')
    apply_tufte_style(ax, grid_axis='y', offset_spines=False)

    ax.legend(title=None, bbox_to_anchor=(0.5, 1.03), loc='lower center',
              ncol=min(len(strat_order), 7), fontsize=9.5, frameon=False)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(agg_dir / f'aggregate_improvement_boxplot.{ext}',
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved aggregate_improvement_boxplot.pdf/png (violin)")


def generate_speedup_aggregate_boxplot(df, out_dir, args=None):
    """Generate Tufte-styled log-space violin plot of kernel speedups by reordering algorithm."""
    print("\n=== Aggregated Speedup Violin Plot (by kernel) ===")

    if 'speedup' not in df.columns:
        print("  No speedup column — skipping aggregate speedup violin plot.")
        return

    df_reord = df[(df['strategy'] != 'Original')
                  & (df['kernel_id'] != 'CUSPARSE_SPMM_BSR_bs32')].copy()

    strat_order = [get_perm_display(p) for p in PERMS if p != 'None']
    strat_order = [s for s in strat_order if s in df_reord['strategy'].unique()]

    kernels = sorted(df_reord['kernel_id'].unique())
    kernel_display = {k: KERNEL_NAMES.get(k, k) for k in kernels}
    kernel_names_ordered = [kernel_display[k] for k in kernels]

    agg_dir = out_dir / 'aggregate_speedup'
    agg_dir.mkdir(parents=True, exist_ok=True)
    strat_palette = {k: v for k, v in pu.get_strategy_palette().items() if k != 'Original'}

    for nc in sorted(df_reord['n_cols'].unique()):
        df_nc = df_reord[df_reord['n_cols'] == nc].copy()
        df_nc['kernel_display'] = df_nc['kernel_id'].map(kernel_display)
        df_nc = df_nc.dropna(subset=['speedup'])
        df_nc = df_nc[np.isfinite(df_nc['speedup']) & (df_nc['speedup'] > 0)]
        if df_nc.empty:
            continue

        df_nc['log_speedup'] = np.log10(df_nc['speedup'])

        n_kernels = len(kernel_names_ordered)
        fig, ax = plt.subplots(figsize=(max(13, n_kernels * 1.9), 5.6))

        _draw_tufte_violin(
            ax, df_nc, x='kernel_display', y_log='log_speedup', hue='strategy',
            order=kernel_names_ordered, hue_order=strat_order, palette=strat_palette,
        )

        ax.axhline(y=0.0, color='#333333', linestyle='--', linewidth=0.8, alpha=0.65, zorder=2)

        p02 = df_nc['log_speedup'].quantile(0.02)
        p98 = df_nc['log_speedup'].quantile(0.98)
        pad = max(0.08, (p98 - p02) * 0.08)
        y_lo, y_hi = p02 - pad, p98 + pad
        ax.set_ylim(y_lo, y_hi)
        _format_log10_ratio_axis(ax, y_lo, y_hi)

        ax.set_xlabel('')
        ax.set_ylabel('Speedup (log scale)', fontsize=10, labelpad=8)
        ax.tick_params(axis='x', rotation=25)
        plt.setp(ax.get_xticklabels(), ha='right')
        apply_tufte_style(ax, grid_axis='y', offset_spines=False)

        ax.legend(title=None, bbox_to_anchor=(0.5, 1.03), loc='lower center',
                  ncol=min(len(strat_order), 7), fontsize=9.5, frameon=False)

        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(agg_dir / f'aggregate_speedup_by_kernel_nc{int(nc)}.{ext}',
                        bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"  Saved aggregate_speedup_by_kernel_nc{int(nc)}.pdf/png (violin)")


def generate_speedup_profile_plots(df, out_dir, args=None):
    """Generate Tufte-styled Dolan-Moré performance-profile plots of kernel speedups."""
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
                                 figsize=(4.8 * ncols_grid, 3.6 * nrows_grid),
                                 squeeze=False)

        for idx, kid in enumerate(kernel_ids):
            ax = axes[idx // ncols_grid][idx % ncols_grid]
            df_k = df_nc[df_nc['kernel_id'] == kid].copy()
            df_k = df_k.dropna(subset=['speedup'])
            df_k = df_k[df_k['speedup'] > 0]

            if df_k.empty:
                ax.set_visible(False)
                continue

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

            ax.set_title(KERNEL_NAMES.get(kid, kid), fontsize=10, fontweight='semibold', pad=8)
            ax.set_xlabel('Speedup relative to best', fontsize=9)
            ax.set_ylabel('Fraction of matrices' if (idx % ncols_grid == 0) else '', fontsize=9)
            ax.set_ylim(0, 1.03)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlim(-0.02, 1.05)
            ax.invert_xaxis()
            apply_tufte_style(ax, grid_axis='both', offset_spines=True)

        for idx in range(len(kernel_ids), nrows_grid * ncols_grid):
            axes[idx // ncols_grid][idx % ncols_grid].set_visible(False)

        all_handles, all_labels = [], []
        seen = set()
        for ax_row in axes:
            for ax in ax_row:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l not in seen:
                        seen.add(l)
                        all_handles.append(h)
                        all_labels.append(l)

        label_order = {s: i for i, s in enumerate(strategy_order)}
        pairs = sorted(zip(all_handles, all_labels),
                       key=lambda p: label_order.get(p[1], 999))
        if pairs:
            all_handles, all_labels = zip(*pairs)

        fig.legend(all_handles, all_labels, loc='upper center',
                   ncol=min(len(all_handles), 7), fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 1.03))

        fig.tight_layout(h_pad=1.6, w_pad=1.4)

        for ext in ('pdf', 'png'):
            fig.savefig(prof_dir / f'speedup_profiles_nc{nc}.{ext}',
                        bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"  Saved speedup_profiles_nc{nc}.pdf/png")


def generate_imp_correlation_plots(df, out_dir):
    """Generate Tufte-styled grouped bar charts of improvement-metric correlations."""
    method = pu.get_correlation_method()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    n_metrics = len(imp_metrics)
    n_kernels = len(kernel_order)

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''

        for n_cols in n_cols_values:
            corr_df = compute_imp_correlations(
                df, n_cols, imp_metrics, kernels,
                method=method, log_transform=log_transform)

            fig, ax = plt.subplots(figsize=(11.5, 5.2))
            bar_width = 0.72 / n_metrics
            x = np.arange(n_kernels)

            for mi, mc in enumerate(imp_metrics):
                col = f'{mc}_corr'
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    val = np.nan if row.empty else row.iloc[0].get(col, np.nan)
                    vals.append(0.0 if pd.isna(val) else val)
                offset = (mi - (n_metrics - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.92,
                       label=get_metric_display(mc),
                       color=get_metric_color(mc),
                       edgecolor='white', linewidth=0.6)

            ax.axhline(0, color='#333333', linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=25, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel('Correlation with Speedup', fontsize=10, labelpad=8)
            apply_tufte_style(ax, grid_axis='y', keep_left=False)
            ax.legend(title=None, bbox_to_anchor=(0.5, 1.02), loc='lower center',
                      ncol=min(n_metrics, 4), fontsize=9, frameon=False)

            fname = f'imp_correlation_bars_{tag}{scale_suffix}_ncols_{int(n_cols)}.png'
            out_path = plot_dir / fname
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def generate_imp_ncols_correlation_plots(df, out_dir):
    """Generate per-metric Tufte-styled bar charts across n_cols."""
    method = pu.get_correlation_method()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    n_kernels = len(kernel_order)
    n_ncols = len(n_cols_values)

    _teal_shades = ['#b2dfdb', '#4db6ac', '#00897b', '#004d40']
    _hatches = ['', '///', '...', 'xxx', '\\\\\\', '+++']
    ncols_colors = dict(zip(n_cols_values, _teal_shades[:n_ncols]))
    ncols_hatches = dict(zip(n_cols_values, _hatches[:n_ncols]))

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        corr_by_ncols = {
            n_cols: compute_imp_correlations(
                df, n_cols, imp_metrics, kernels,
                method=method, log_transform=log_transform)
            for n_cols in n_cols_values
        }

        for mc in imp_metrics:
            col = f'{mc}_corr'
            metric_display = get_metric_display(mc)
            metric_safe = pu.safe_filename(mc)

            fig, ax = plt.subplots(figsize=(11.5, 5.2))
            bar_width = 0.72 / n_ncols
            x = np.arange(n_kernels)

            for ni, n_cols in enumerate(n_cols_values):
                corr_df = corr_by_ncols[n_cols]
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    val = np.nan if row.empty else row.iloc[0].get(col, np.nan)
                    vals.append(0.0 if pd.isna(val) else val)
                offset = (ni - (n_ncols - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.92,
                       label=f'$n_{{cols}}={int(n_cols)}$',
                       color=ncols_colors[n_cols],
                       hatch=ncols_hatches[n_cols],
                       edgecolor='#333333' if ncols_hatches[n_cols] else 'white',
                       linewidth=0.45)

            ax.axhline(0, color='#333333', linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=25, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel(f'Correlation of {metric_display} with Speedup', fontsize=10, labelpad=8)
            apply_tufte_style(ax, grid_axis='y', keep_left=False)
            ax.legend(title=None, bbox_to_anchor=(0.5, 1.02), loc='lower center',
                      ncol=n_ncols, fontsize=9.5, frameon=False)

            fname = f'imp_ncols_bars_{tag}{scale_suffix}_{metric_safe}.png'
            out_path = plot_dir / fname
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def generate_imp_blocksize_correlation_plots(df, out_dir):
    """Generate Tufte-styled bar charts of density-improvement–speedup correlation across block sizes."""
    method = pu.get_correlation_method()
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

        for n_cols in n_cols_values:
            corr_df = compute_imp_correlations(
                df, n_cols, bd_imp_metrics, kernels,
                method=method, log_transform=log_transform)

            fig, ax = plt.subplots(figsize=(11.5, 5.2))
            bar_width = 0.72 / n_block_sizes
            x = np.arange(n_kernels)

            for bi, (metric, label) in enumerate(zip(bd_imp_metrics, block_labels)):
                col = f'{metric}_corr'
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    val = np.nan if row.empty else row.iloc[0].get(col, np.nan)
                    vals.append(0.0 if pd.isna(val) else val)
                offset = (bi - (n_block_sizes - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.92,
                       label=f'{label}×{label}',
                       color=get_metric_color(metric),
                       edgecolor='white', linewidth=0.6)

            ax.axhline(0, color='#333333', linewidth=0.8, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=25, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel('Correlation of Block Density Improvement with Speedup', fontsize=10, labelpad=8)
            apply_tufte_style(ax, grid_axis='y', keep_left=False)
            ax.legend(title='Block size', bbox_to_anchor=(0.5, 1.02), loc='lower center',
                      ncol=n_block_sizes, fontsize=9, title_fontsize=9.5, frameon=False)

            fname = f'imp_blocksize_bars_{tag}{scale_suffix}_ncols_{int(n_cols)}.png'
            out_path = plot_dir / fname
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def generate_partial_correlation_heatmaps(df, out_dir):
    """Generate clean, minimal partial-correlation heatmaps: one per (kernel, n_cols)."""
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

    def _draw_heatmap(mat, title, cbar_label, out_path):
        n = len(imp_metrics)
        cell_size = max(0.85, min(1.15, 7.5 / n))
        fig, ax = plt.subplots(figsize=(n * cell_size + 2.2, n * cell_size + 1.4))

        hm = sns.heatmap(
            mat.abs().astype(float), annot=True, fmt='.2f',
            cmap='YlOrRd', vmin=0, vmax=0.7,
            ax=ax, linewidths=1.0, linecolor='white', square=True,
            cbar_kws={'label': cbar_label, 'shrink': 0.78, 'pad': 0.03},
            annot_kws={'size': 8.5},
        )
        cbar = hm.collections[0].colorbar
        if cbar is not None:
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0, labelsize=8.5)

        ax.set_title(title, fontsize=11, pad=12, color='#111111')
        ax.set_xlabel('Controlled metric', fontsize=9.5, labelpad=8)
        ax.set_ylabel('Target metric', fontsize=9.5, labelpad=8)
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")

    for n_cols in n_cols_values:
        n_cols_int = int(n_cols)
        all_mats = []

        for kernel in kernels:
            mat = pu.compute_partial_imp_matrix(
                df, n_cols, imp_metrics, kernel, method=method)
            if mat.isna().all().all():
                continue
            all_mats.append(mat.copy())

            mat.index = display_labels
            mat.columns = display_labels
            kernel_display = KERNEL_NAMES.get(kernel, kernel)
            kernel_safe = pu.safe_filename(kernel)
            _draw_heatmap(
                mat,
                title=f'{kernel_display} — |r(speedup, row | col)|  (n_cols={n_cols_int})',
                cbar_label=f'|Partial {corr_display}|',
                out_path=plot_dir / f'partial_corr_{tag}_{kernel_safe}_ncols_{n_cols_int}.png',
            )

        if len(all_mats) >= 2:
            mean_mat = pd.concat(all_mats).groupby(level=0).mean()
            mean_mat = mean_mat.reindex(index=imp_metrics, columns=imp_metrics)
            mean_mat.index = display_labels
            mean_mat.columns = display_labels
            _draw_heatmap(
                mean_mat,
                title=f'Mean Partial Correlation (all kernels) — |r(speedup, row | col)|  (n_cols={n_cols_int})',
                cbar_label=f'Mean |Partial {corr_display}|',
                out_path=plot_dir / f'partial_corr_{tag}_mean_ncols_{n_cols_int}.png',
            )


def generate_feature_importance_plots(df, out_dir):
    """Random Forest feature importance: clean heatmap + Tufte grouped bars."""
    from sklearn.ensemble import RandomForestRegressor
    import seaborn as sns

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    kernel_displays = [KERNEL_NAMES.get(k, k) for k in kernels]
    n_cols_values = sorted(df['n_cols'].unique())
    n_metrics = len(imp_metrics)

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
        imp_mat = pd.DataFrame(np.nan, index=display_labels, columns=kernel_displays)

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

        # ---- Heatmap ----
        fig, ax = plt.subplots(
            figsize=(max(5, valid.shape[1] * 1.25 + 2),
                     max(4, valid.shape[0] * 0.55 + 1.5)))
        hm = sns.heatmap(
            valid.astype(float), annot=True, fmt='.3f',
            cmap='Blues', vmin=0, ax=ax,
            linewidths=1.0, linecolor='white',
            cbar_kws={'label': 'Feature Importance', 'shrink': 0.8, 'pad': 0.03},
            annot_kws={'size': 8.5},
        )
        cbar = hm.collections[0].colorbar
        if cbar is not None:
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0, labelsize=8.5)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(length=0)
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        fig.tight_layout()
        fname = plot_dir / f'rf_importance_heatmap_ncols_{nc_int}.png'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fname}")

        # ---- Grouped bar chart ----
        valid_kernels = list(valid.columns)
        n_valid = len(valid_kernels)
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        bar_width = 0.72 / n_metrics
        x = np.arange(n_valid)

        for mi, (mc, dlabel, mdisplay) in enumerate(
                zip(imp_metrics, display_labels, metric_order)):
            vals = [valid.loc[dlabel, k] if dlabel in valid.index else 0.0
                    for k in valid_kernels]
            offset = (mi - (n_metrics - 1) / 2) * bar_width
            hatch = get_metric_hatch(mc)
            ax.bar(x + offset, vals, bar_width * 0.92,
                   label=mdisplay, color=get_metric_color(mc),
                   hatch=hatch,
                   edgecolor='#333333' if hatch else 'white',
                   linewidth=0.45)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_kernels, rotation=25, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('Feature Importance', fontsize=10, labelpad=8)
        apply_tufte_style(ax, grid_axis='y', keep_left=False)
        ax.legend(title=None, bbox_to_anchor=(0.5, 1.02), loc='lower center',
                  ncol=min(n_metrics, 4), fontsize=9, frameon=False)
        fig.tight_layout()
        fname = plot_dir / f'rf_importance_bars_ncols_{nc_int}.png'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _should_run(section: str, args) -> bool:
    if args.sections is not None:
        return section in args.sections
    kernel_sections = {'grouped-scatter', 'breakeven', 'speedup-profiles'}
    reorder_sections = {'reorder-analysis', 'profiles', 'pareto'}
    if args.only_kernels:
        return section in kernel_sections
    if args.only_reorder_analysis:
        return section in reorder_sections
    return True


def main():
    args = parse_args()

    perm_type_filter = 'ROW' if args.row else ('ASYMMETRIC' if args.asymmetric else 'SYMMETRIC')
    pipeline_key = ('random_' if args.random else '') + perm_type_filter.lower()

    if args.out is None:
        data_label = 'random' if args.random else 'original'
        perm_label = perm_type_filter.lower()
        args.out = f"plots/{data_label}_{perm_label}"

    if args.sections is None and args.only_reorder_analysis and args.only_kernels:
        print("Error: --only-reorder-analysis and --only-kernels are mutually exclusive.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pu.set_professional_style()
    set_tufte_rcparams()

    cli_overrides = {
        'operations_csv': args.operations,
        'analysis_csv': args.analysis,
        'matrices_list': args.matrices_list,
        'one_per_family': args.one_per_family,
        'min_size': args.min_size,
        'max_size': args.max_size,
        'random': args.random,
    }
    if args.include_rectangular is not None:
        cli_overrides['square_only'] = not args.include_rectangular

    df, df_analysis, _cfg = pu.load_and_filter_data(
        config_path=args.filter_config,
        cli_overrides=cli_overrides,
    )

    pipeline_cfg = _cfg.get('pipelines', {}).get(pipeline_key, {})
    df, df_analysis = pu.split_by_perm_type(
        df, df_analysis, perm_type_filter, pipeline_cfg)

    has_ops = not df.empty
    if has_ops:
        print("\nProcessing data...")
        df = pu.prepare_full_dataframe(df)
        print(f"Unique kernels: {df['kernel_id'].unique().tolist()}")
        print(f"Unique n_cols: {sorted(df['n_cols'].unique())}")
    else:
        print("\nNo operations data — kernel/breakeven/original-scatter plots will be skipped.")

    def _reordering_csv():
        return _cfg.get('data', {}).get('reordering_csv',
                                        'results/results_reordering.csv')

    if has_ops and _should_run('grouped-scatter', args):
        print("\n" + "=" * 60)
        print("Generating grouped improvement vs speedup scatter plots...")
        print("=" * 60)
        generate_grouped_scatter_plots(df, out_dir, args)

    if has_ops and _should_run('breakeven', args):
        print("\n" + "=" * 60)
        print("Generating break-even analysis plots...")
        print("=" * 60)
        generate_breakeven_plots(df, out_dir, reordering_csv=_reordering_csv(),
                                 args=args)

    if _should_run('reorder-analysis', args):
        print("\n" + "=" * 60)
        print("Generating reordering analysis plots...")
        print("=" * 60)
        generate_reorder_analysis_plots(df_analysis, out_dir, n_jobs=args.jobs)

    if _should_run('profiles', args):
        print("\n" + "=" * 60)
        print("Generating performance profile plots...")
        print("=" * 60)
        generate_profile_plots(df_analysis, out_dir)

    if _should_run('pareto', args):
        print("\n" + "=" * 60)
        print("Generating Pareto density-vs-time plots...")
        print("=" * 60)
        generate_pareto_plots(df_analysis, out_dir, reordering_csv=_reordering_csv(),
                              args=args)

    if _should_run('aggregate-improvement', args):
        print("\n" + "=" * 60)
        print("Generating aggregate improvement violin plots...")
        print("=" * 60)
        generate_profile_aggregate_boxplot(df_analysis, out_dir)
        if has_ops:
            generate_speedup_aggregate_boxplot(df, out_dir, args)

    if has_ops and _should_run('speedup-profiles', args):
        print("\n" + "=" * 60)
        print("Generating speedup performance profile plots...")
        print("=" * 60)
        generate_speedup_profile_plots(df, out_dir, args)

    if has_ops and _should_run('imp-correlation', args):
        print("\n" + "=" * 60)
        print("Generating improvement-correlation bar charts...")
        print("=" * 60)
        generate_imp_correlation_plots(df, out_dir)
        generate_imp_ncols_correlation_plots(df, out_dir)
        generate_imp_blocksize_correlation_plots(df, out_dir)

    if has_ops and _should_run('partial-correlation', args):
        print("\n" + "=" * 60)
        print("Generating partial-correlation heatmaps...")
        print("=" * 60)
        generate_partial_correlation_heatmaps(df, out_dir)

    if has_ops and _should_run('feature-importance', args):
        print("\n" + "=" * 60)
        print("Generating RF feature importance plots...")
        print("=" * 60)
        generate_feature_importance_plots(df, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()