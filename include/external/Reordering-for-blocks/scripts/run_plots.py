#!/usr/bin/env python3
"""
Interactive wrapper around plot.py, correlation_table.py, and spy_plots.py.

Run without arguments for an interactive menu, or pass flags to select
specific tasks directly:

    python run_plots.py                     # interactive menu
    python run_plots.py --plots 1 3 5       # run specific plot groups
    python run_plots.py --corr --spy        # run correlation tables + spy plots
    python run_plots.py --all               # everything
"""

import argparse
import sys
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent

# ── Plot groups (plot.py --sections <key>) ───────────────────────────────────
# Each entry: (label, section key passed to plot.py --sections)
PLOT_GROUPS = [
    ("Grouped improvement scatter",    "grouped-scatter"),
    ("Break-even analysis plots",      "breakeven"),
    ("Reorder analysis plots",         "reorder-analysis"),
    ("Performance profile plots",      "profiles"),
    ("Speedup profile plots",          "speedup-profiles"),
    ("Aggregate improvement plots",    "aggregate-improvement"),
    ("Improvement correlation plots",  "imp-correlation"),
    ("Partial correlation heatmaps",   "partial-correlation"),
    ("Feature importance plots",       "feature-importance"),
    ("ALL plot.py plots",              None),          # None → no --sections flag
]

# ── Correlation table groups (correlation_table.py --sections <key>) ─────────
CORR_GROUPS = [
    ("Metric-kernel correlation tables (all)",        "correlations"),
    ("Block-size correlation tables (all)",           "blocksize"),
    ("Per-metric correlation tables (all)",           "per-metric"),
    ("Metric-kernel correlation tables (original)",   "correlations-original"),
    ("Block-size correlation tables (original)",      "blocksize-original"),
    ("Per-metric correlation tables (original)",      "per-metric-original"),
    ("Median improvement tables",                     "improvement"),
    ("Improvement-speedup correlation tables",        "imp-correlations"),
    ("Improvement-speedup block-size tables",         "imp-blocksize"),
    ("ALL correlation tables",                        None),
]

# ── Spy plot (spy_plots.py) ──────────────────────────────────────────────────
# Spy plots require --matrices-dir and --perms-dir; user will be prompted.


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(text: str) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}")


def _run(cmd: list[str], label: str) -> None:
    """Print and execute a command."""
    _header(label)
    print(f"  > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\n[!] Command exited with code {result.returncode}")


def _prompt(prompt_text: str, default: str = "") -> str:
    """Prompt with an optional default."""
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt_text}{suffix}: ").strip()
    return val if val else default


def _pick_multiple(options: list[str], header: str) -> list[int]:
    """Display numbered options and return list of chosen indices (0-based)."""
    print(f"\n{header}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"  0. Skip")
    raw = input("Choice (comma-separated, e.g. 1,3 or 0 to skip): ").strip()
    if raw == "0" or not raw:
        return []
    indices = []
    for part in raw.replace(" ", "").split(","):
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(options):
                indices.append(idx)
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_plot(choices: list[int], extra_args: list[str] | None = None) -> None:
    """Run selected plot.py groups in a single invocation."""
    extra = extra_args or []
    # Collect section keys; None means "all" → no --sections flag
    sections = []
    run_all = False
    for idx in choices:
        label, section = PLOT_GROUPS[idx]
        if section is None:
            run_all = True
            break
        sections.append(section)

    cmd = [sys.executable, str(SCRIPTS_DIR / "plot.py")]
    if not run_all and sections:
        cmd += ["--sections"] + sections
    cmd += extra

    summary = ", ".join(PLOT_GROUPS[i][0] for i in choices)
    _run(cmd, f"plot.py  [{summary}]")


def run_corr(choices: list[int], extra_args: list[str] | None = None) -> None:
    """Run selected correlation_table.py groups in a single invocation."""
    extra = extra_args or []
    sections = []
    run_all = False
    for idx in choices:
        label, section = CORR_GROUPS[idx]
        if section is None:
            run_all = True
            break
        sections.append(section)

    cmd = [sys.executable, str(SCRIPTS_DIR / "correlation_table.py")]
    if not run_all and sections:
        cmd += ["--sections"] + sections
    cmd += extra

    summary = ", ".join(CORR_GROUPS[i][0] for i in choices)
    _run(cmd, f"correlation_table.py  [{summary}]")


def run_spy(extra_args: list[str] | None = None) -> None:
    """Run spy_plots.py, prompting for required dirs if not provided."""
    extra = extra_args or []

    # Check if required dirs were passed via extra args
    has_matrices_dir = any(a.startswith("--matrices-dir") for a in extra)
    has_perms_dir = any(a.startswith("--perms-dir") for a in extra)

    if not has_matrices_dir:
        d = _prompt("Matrices directory", "datasets")
        extra += ["--matrices-dir", d]
    if not has_perms_dir:
        d = _prompt("Permutations directory", "perms")
        extra += ["--perms-dir", d]

    cmd = [sys.executable, str(SCRIPTS_DIR / "spy_plots.py")] + extra
    _run(cmd, "Spy plots")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────────────────────────

def interactive() -> None:
    """Full interactive flow."""
    _header("Plot Pipeline – Interactive Mode")

    print("\nWhich modules do you want to run?")
    print("  1. plot.py          (performance / reorder plots)")
    print("  2. correlation_table.py (LaTeX correlation tables)")
    print("  3. spy_plots.py     (spy matrix visualisations)")
    print("  0. Quit")
    modules_raw = input("Modules (comma-separated, e.g. 1,2): ").strip()
    if modules_raw == "0" or not modules_raw:
        print("Nothing selected – exiting.")
        return

    modules = [int(m) for m in modules_raw.replace(" ", "").split(",") if m.isdigit()]

    # Shared extra args the user may want to forward
    print("\nExtra CLI args forwarded to ALL scripts (blank for none):")
    print("  e.g. --filter-config my_filter.yaml --n-cols 256")
    extra_raw = input("> ").strip()
    extra_args = extra_raw.split() if extra_raw else []

    # ── plot.py ──────────────────────────────────────────────────────────
    if 1 in modules:
        labels = [g[0] for g in PLOT_GROUPS]
        picks = _pick_multiple(labels, "plot.py – select plot groups:")
        if picks:
            run_plot(picks, extra_args)

    # ── correlation_table.py ─────────────────────────────────────────────
    if 2 in modules:
        labels = [g[0] for g in CORR_GROUPS]
        picks = _pick_multiple(labels, "correlation_table.py – select table groups:")
        if picks:
            run_corr(picks, extra_args)

    # ── spy_plots.py ─────────────────────────────────────────────────────
    if 3 in modules:
        run_spy(extra_args)

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive wrapper around plot.py, correlation_table.py, spy_plots.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--plots", nargs="*", type=int, metavar="N",
        help=(
            "Run specific plot.py groups by number (1-indexed). "
            f"Options: {', '.join(f'{i+1}={g[0]}' for i,g in enumerate(PLOT_GROUPS))}"
        ),
    )
    parser.add_argument("--corr", nargs="*", type=int, metavar="N",
        help="Run correlation_table.py groups by number (or no number = all)")
    parser.add_argument("--spy", action="store_true", help="Run spy_plots.py")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--random", action="store_true",
        help="Use random-pipeline data (forwarded as --random to all scripts)")
    parser.add_argument("--row", action="store_true",
        help="Use ROW perm_type pipeline (forwarded as --row to all scripts)")
    parser.add_argument("--symmetric", action="store_true",
        help="Use SYMMETRIC perm_type pipeline (default, forwarded as --symmetric)")

    # Pass-through args (everything after --)
    parser.add_argument("extra", nargs=argparse.REMAINDER,
        help="Extra args forwarded to underlying scripts (put after --)")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()

    # Strip leading "--" from remainder args
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Forward --random and --row/--symmetric to all underlying scripts
    if args.random:
        extra = ["--random"] + extra
    if args.row:
        extra = ["--row"] + extra
    elif args.symmetric:
        extra = ["--symmetric"] + extra

    non_interactive = args.all or args.plots is not None or args.corr is not None or args.spy

    if not non_interactive:
        interactive()
        return

    # ── Non-interactive mode ─────────────────────────────────────────────
    if args.all:
        run_plot([len(PLOT_GROUPS) - 1], extra)          # "ALL plot.py plots"
        run_corr([len(CORR_GROUPS) - 1], extra)          # "ALL correlation tables"
        run_spy(extra)
        return

    if args.plots is not None:
        indices = [n - 1 for n in args.plots if 1 <= n <= len(PLOT_GROUPS)] if args.plots else [len(PLOT_GROUPS) - 1]
        run_plot(indices, extra)

    if args.corr is not None:
        indices = [n - 1 for n in args.corr if 1 <= n <= len(CORR_GROUPS)] if args.corr else [len(CORR_GROUPS) - 1]
        run_corr(indices, extra)

    if args.spy:
        run_spy(extra)


if __name__ == "__main__":
    main()
