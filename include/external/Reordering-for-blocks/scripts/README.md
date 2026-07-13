# scripts/

Utilities for parsing experiment results, generating plots, and monitoring jobs.

## Files

### parse_results.py

Aggregates experiment outputs from SbatchMan job archives into CSV files.

**Usage:**
```bash
python scripts/parse_results.py
```

**Outputs:**
- `results/results_analysis.csv` - Matrix structural metrics (~17K rows)
- `results/results_operations.csv` - SpMM benchmark timings (~137K rows)

**Key Functions:**
- `parse_analysis_result()` - Parses JSON output from matrix analysis
- `parse_operation_result()` - Parses timing from `<Timer>[label] X.XXX ms` format
- Reads job outputs from `SbatchMan/experiments/*/jobs/*/output.txt`

### plot.py

Generates correlation and performance plots from parsed results.

**Usage:**
```bash
python scripts/plot.py [--one-per-family] [--row] [--random] [--n-cols N] [--kernel KERNEL]
```

**Arguments:**
- `--one-per-family`: Select one representative matrix per SuiteSparse family
- `--row` / `--symmetric` (default): Select ROW or SYMMETRIC perm_type pipeline
- `--random`: Use random-pipeline data
- `--n-cols`: Filter by dense matrix width (32, 256, 1024)
- `--kernel`: Filter by specific kernel name

**Outputs to:** `plots/` (default), `plots_row/` (with `--row`), `plots_random/` (with `--random`), `plots_random_row/` (with `--random --row`)

Break-even analysis plots (minimum SpMM operations for reordering to pay for itself) are generated per kernel under `plots/n_cols_{N}/{kernel}/breakeven/`. Cases where reordering is harmful are shown as × markers at a cap line.

### plot_utils.py

Shared plotting utilities and style configurations.

### job_check.py

Monitors SLURM job status and generates reports.

**Usage:**
```bash
python scripts/job_check.py
```

**Outputs to:** `reports/job_status_{timestamp}.txt`

### check_missing.py

Identifies missing or failed experiment runs.

## Plotting Rules

- **Boxplot whiskers at 5th/95th percentiles**: All boxplots (including binned speedups and break-even plots) must use whiskers at the 5th and 95th percentiles (`whis=(5, 95)` for seaborn, `whis=(5, 95)` for matplotlib) with outliers hidden (`showfliers=False`). Do not use the default 1.5×IQR whiskers.

## Data Flow

```
SbatchMan/experiments/*/jobs/*/output.txt
            │
            ▼
    parse_results.py
            │
            ▼
    results/*.csv
            │
            ▼
       plot.py (--symmetric default, --row for ROW pipeline)
            │
            ├──> plots/n_cols_{N}/{kernel}/breakeven/   (break-even boxplots)
            ├──> plots/n_cols_{N}/{kernel}/speedup/      (speedup boxplots)
            ├──> plots/reorder_analysis/                 (structural analysis)
            └──> (or plots_row/, plots_random/, plots_random_row/)
```
