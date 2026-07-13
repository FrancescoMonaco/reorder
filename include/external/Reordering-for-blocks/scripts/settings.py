"""
Shared settings and display-name dictionaries for plots and tables."""

# =============================================================================
# Professional Plot Style - Palette
# =============================================================================

# Colorblind-friendly palette for categorical data (Tol's muted scheme)
PALETTE = [
    '#332288', '#88CCEE', '#44AA99', '#117733',
    '#999933', '#DDCC77', '#CC6677', '#882255',
    '#AA4499', '#DDDDDD', '#4477AA', '#000000',
]

# =============================================================================
# Metric Configuration – single source of truth
# =============================================================================
# Every metric column lives here.  Each entry can carry:
#   display          – full human-readable name (plot labels, LaTeX captions)
#   short            – compact abbreviation for table column headers
#                      (defaults to *display* when omitted)
#   log_scale        – whether to use log scale on plot axes (default False)
#   enabled          – include in default correlation tables (default False)
#   higher_is_better – direction of improvement; True / False / None (N/A)
#
# Add or remove metrics here; every other module derives its look-ups from
# this single dict.
# =============================================================================

BLOCK_SIZES = [4, 8, 16, 32, 64, 128]

ALL_METRICS = {
    # ── Performance ──────────────────────────────────────────────────────
    'gflops':  {'display': 'GFLOPS',  'log_scale': True},
    'speedup': {'display': 'Speedup'},

    # ── Bandwidth ────────────────────────────────────────────────────────
    'bandwidth_max': {
        'display': 'Bandwidth', 'short': 'BW',
        'log_scale': True, 'higher_is_better': False,
    },
    'bandwidth_avg': {
        'display': 'Average Bandwidth', 'short': 'ABW',
        'enabled': True, 'higher_is_better': False,
    },
    'rel_bandwidth': {
        'display': 'Relative Bandwidth', 'short': 'RBW',
        'log_scale': True, 'enabled': True, 'higher_is_better': False,
    },
    'bandwidth_improvement': {
        'display': 'Bandwidth Reduction', 'short': 'BWR',
        'color': '#6B9AC4', 'hatch': '',
    },
    'bandwidth_avg_improvement': {
        'display': 'Avg Bandwidth Reduction', 'short': 'ABWR',
    },

    # ── Row spread / locality ────────────────────────────────────────────
    'locality_avg_row_spread': {
        'display': 'Average Row Spread', 'short': 'ARS',
        'higher_is_better': False,
    },
    'locality_max_row_spread': {
        'display': 'Maximum Row Spread', 'short': 'MRS',
        'higher_is_better': False,
    },
    'rel_row_spread': {
        'display': 'Relative Row Spread', 'short': 'RRS',
        'log_scale': True, 'enabled': True, 'higher_is_better': False,
    },
    'row_spread_improvement':  {'display': 'Row Spread Reduction', 'short': 'RSR'},
    'col_spread_improvement':  {
        'display': 'Col Spread Reduction', 'short': 'CSR',
        'color': '#D4A574', 'hatch': '///',
    },

    # ── Profile improvement ──────────────────────────────────────────────
    'profile_improvement': {
        'display': 'Profile Reduction', 'short': 'ProfR',
        'color': '#C47A7A', 'hatch': 'xxx',
    },

    # ── Column spread ────────────────────────────────────────────────────
    'locality_avg_col_spread': {
        'display': 'Average Column Spread', 'short': 'ACS',
        'enabled': True, 'higher_is_better': False,
    },
    'locality_max_col_spread': {
        'display': 'Maximum Column Spread', 'short': 'MCS',
        'higher_is_better': False,
    },

    # ── Vertical adjacency ───────────────────────────────────────────────
    'locality_consecutive_vertical_pairs': {
        'display': 'Consecutive Vertical Pairs', 'short': 'CVP',
        'higher_is_better': True,
    },
    'locality_vertical_adjacency_ratio': {
        'display': 'Vertical Adjacency Ratio', 'short': 'VAR',
        'enabled': True, 'higher_is_better': True,
    },
    'vertical_adjacency_improvement': {
        'display': 'Vertical Adjacency Improvement', 'short': 'VAI',
        'color': '#7DB58A', 'hatch': '...',
    },

    # ── Access distance improvements ──────────────────────────────────
    'reuse_distance_improvement': {
        'display': 'Reuse Distance Reduction', 'short': 'RDR',
        'color': '#E8A838', 'hatch': '\\\\',
    },
    'reuse_distance_median_improvement': {
        'display': 'Median Reuse Dist. Reduction', 'short': 'MdRDR',
    },
    'index_distance_improvement': {
        'display': 'Index Distance Reduction', 'short': 'IDR',
        'color': '#E05858', 'hatch': '+++',
    },
    'index_distance_median_improvement': {
        'display': 'Median Index Dist. Reduction', 'short': 'MdIDR',
    },

    # ── NNZ distribution ─────────────────────────────────────────────────
    'locality_avg_nnz_per_row': {
        'display': 'Average NNZ per Row', 'short': 'ANR',
    },
    'locality_max_nnz_per_row': {
        'display': 'Maximum NNZ per Row', 'short': 'MNR',
    },
    'locality_num_empty_rows': {
        'display': 'Number of Empty Rows', 'short': 'NER',
    },
    'locality_num_empty_cols': {
        'display': 'Number of Empty Columns', 'short': 'NEC',
    },

    # ── Profile ──────────────────────────────────────────────────────────
    'locality_profile': {
        'display': 'Profile', 'short': 'Prof',
        'log_scale': True, 'enabled': True, 'higher_is_better': False,
    },

    # ── Access distance (reuse distance — temporal locality) ─────────────
    'access_dist_reuse_distance_mean': {
        'display': 'Reuse Distance', 'short': 'RD',
        'log_scale': True, 'enabled': True, 'higher_is_better': False,
    },
    'access_dist_reuse_distance_median': {
        'display': 'Median Reuse Distance', 'short': 'MdRD',
        'log_scale': True, 'higher_is_better': False,
    },
    'access_dist_reuse_distance_max': {
        'display': 'Max Reuse Distance', 'short': 'XRD',
        'log_scale': True, 'higher_is_better': False,
    },
    'access_dist_reuse_distance_std': {
        'display': 'Reuse Distance Std', 'short': 'SRD',
        'higher_is_better': False,
    },

    # ── Access distance (index distance — spatial locality) ───────────
    'access_dist_index_distance_mean': {
        'display': 'Index Distance', 'short': 'ID',
        'log_scale': True, 'enabled': True, 'higher_is_better': False,
    },
    'access_dist_index_distance_median': {
        'display': 'Median Index Distance', 'short': 'MdID',
        'log_scale': True, 'higher_is_better': False,
    },
    'access_dist_index_distance_max': {
        'display': 'Max Index Distance', 'short': 'XID',
        'log_scale': True, 'higher_is_better': False,
    },
    'access_dist_index_distance_std': {
        'display': 'Index Distance Std', 'short': 'SID',
        'higher_is_better': False,
    },

    # ── Relative access distance (normalized by cols) ─────────────────
    'rel_reuse_distance': {
        'display': 'Relative Reuse Distance', 'short': 'RRD',
        'log_scale': True, 'higher_is_better': False,
    },
    'rel_index_distance': {
        'display': 'Relative Index Distance', 'short': 'RID',
        'log_scale': True, 'higher_is_better': False,
    },

    # ── Overall density ──────────────────────────────────────────────────
    'density': {
        'display': 'Density', 'short': 'Dens',
        'log_scale': True,
    },
}

# ── Block-size-dependent metrics (generated from BLOCK_SIZES) ────────────
_BLOCK_DENSITY_ENABLED = {16}  # which block densities show in tables

# Color/hatch sequences for density-improvement metrics (one per block size)
_DENSITY_IMPROVEMENT_COLORS  = ['#e8c8e8', '#d4a0d4', '#b06cb0', '#8c3f8c', '#6a1b6a', '#3d003d']
_DENSITY_IMPROVEMENT_HATCHES = ['#', '#', '#', '#', '#', '#']

for _i, _bs in enumerate(BLOCK_SIZES):
    ALL_METRICS[f'block_density_{_bs}'] = {
        'display': f'Block Density ${_bs}{{\\times}}{_bs}$',
        'short':   f'BD{_bs}',
        'log_scale': True,
        'enabled': _bs in _BLOCK_DENSITY_ENABLED,
        'higher_is_better': True,
    }
    ALL_METRICS[f'density_improvement_{_bs}'] = {
        'display': f'Density Improvement ${_bs}{{\\times}}{_bs}$',
        'short': f'DI{_bs}',
        'color': _DENSITY_IMPROVEMENT_COLORS[_i],
        'hatch': _DENSITY_IMPROVEMENT_HATCHES[_i],
    }
    ALL_METRICS[f'avg_blocks_per_row_improvement_{_bs}'] = {
        'display': f'Avg Blocks/Row Reduction ${_bs}{{\\times}}{_bs}$',
        'short': f'ABRR{_bs}',
    }
    for _pfx, _nm in [('avg', 'Avg'), ('max', 'Max')]:
        ALL_METRICS[f'{_pfx}_blocks_per_row_{_bs}'] = {
            'display': f'{_nm} Blocks/Row ${_bs}{{\\times}}{_bs}$',
            'short':   f'{"A" if _pfx == "avg" else "M"}BR{_bs}',
            'enabled': False,
            'higher_is_better': False,
        }


# ─────────────────────────────────────────────────────────────────────────
# Metric helpers – use these instead of reaching into ALL_METRICS manually
# ─────────────────────────────────────────────────────────────────────────

def get_metric_display(col: str) -> str:
    """Full human-readable name for *col*, or a formatted fallback."""
    m = ALL_METRICS.get(col)
    if m:
        return m['display']
    return col.replace('_', ' ').title()


def get_metric_short(col: str) -> str:
    """Short abbreviation for *col* (falls back to display name)."""
    m = ALL_METRICS.get(col)
    if m:
        return m.get('short', m['display'])
    return col


def use_log_scale(col: str) -> bool:
    """Whether *col* should default to a log axis."""
    return ALL_METRICS.get(col, {}).get('log_scale', False)


def enabled_metrics() -> list[str]:
    """Metric keys with ``enabled=True`` (default correlation-table set)."""
    return [k for k, v in ALL_METRICS.items() if v.get('enabled')]


def block_density_metrics() -> list[str]:
    """Ordered list of ``block_density_<bs>`` keys."""
    return [f'block_density_{bs}' for bs in BLOCK_SIZES]


def get_metric_color(col: str) -> str:
    """Bar/box color for metric *col*. Falls back to muted grey."""
    return ALL_METRICS.get(col, {}).get('color', '#9B9B9B')


def get_metric_hatch(col: str) -> str:
    """Hatch pattern for metric *col*. Falls back to no hatch."""
    return ALL_METRICS.get(col, {}).get('hatch', '')


# =============================================================================
# Reordering Algorithms – single source of truth
# =============================================================================

PERMS = {
    'None':            {'display': 'Original',  'color': '#333333'},
    'SB_rcm':          {'display': 'RCM',      'color': PALETTE[0]},
    'SB_amd':          {'display': 'AMD',       'color': PALETTE[1]},
    'SB_rabbit':       {'display': 'Rabbit',   'color': PALETTE[2]},
    'GROOT_reorder':   {'display': 'GROOT',    'color': PALETTE[3]},
    'SB_gray':         {'display': 'Gray',     'color': PALETTE[4]},
    'SB_metis':        {'display': 'Metis',    'color': PALETTE[5]},
    'SB_patoh':        {'display': 'PaToH',    'color': PALETTE[6]},
    'SB_slashburn':    {'display': 'SlashBurn', 'color': PALETTE[7]},
    'SPARTA_reorder':  {'display': 'SPARTA',   'color': PALETTE[8]},
    'TCA_reorder':     {'display': 'DTC-LSH',  'color': PALETTE[9]},
    'SB_degree':       {'display': 'Degree',    'color': PALETTE[10]},
    'random1D':        {'display': 'Random',   'color': '#000000'},
}


# ─────────────────────────────────────────────────────────────────────────
# Perm helpers
# ─────────────────────────────────────────────────────────────────────────

def get_perm_display(perm: str) -> str:
    """Display name for a raw perm id, or the id itself as fallback."""
    p = PERMS.get(perm)
    return p['display'] if p else perm


def get_perm_color(perm: str) -> str:
    """Color for a raw perm id, or dark grey as fallback."""
    p = PERMS.get(perm)
    return p['color'] if p else '#333333'


# =============================================================================
# Kernel Display Names
# =============================================================================

KERNEL_NAMES = {
    'ACCSPMM_SPMM': 'AccSpMM',
    'ASPT_SPMM': 'ASPT',
    'BLEST_BFS': 'BLEST-BFS',
    'CUSPARSE_SPMM_BSR_bs32': 'cuSparse BSR',
    'CUSPARSE_SPMM_CSR': 'cuSparse CSR',
    'DTC_SPMM': 'DTC-Spmm',
    'FLASHSPARSE_SPMM': 'FlashSparse',
    'SMAT_SPMM_bs32': 'SMAT',
}

# Kernels excluded from the grouped 2x3 scatter plots
GROUPED_SCATTER_EXCLUDE = {'CUSPARSE_SPMM_BSR_bs32'}




