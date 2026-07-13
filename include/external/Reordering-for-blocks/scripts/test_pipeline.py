#!/usr/bin/env python3
from __future__ import annotations
"""
Isolated pipeline tester — run the full reordering + SpMM pipeline on a single matrix.

Usage:
    python3 scripts/test_pipeline.py <matrix_path>
    python3 scripts/test_pipeline.py <matrix_path> --check-only
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PERM_ALGORITHMS = [
    "GROOT_reorder",
    "random1D",
    "random2D",
    "SB_degree",
    "SB_gray",
    "SB_rcm",
    "SB_metis",
    "SB_amd",
    "SB_rabbit",
    "SB_patoh",
    "SB_slashburn",
    "SPARTA_reorder",
    "ACCORDER_reorder",
    "TCA_reorder",
]

KERNELS = {
    "ASPT":          "python3 operators/aspt_spmm.py",
    "CUSPARSE_CSR":  "python3 operators/cusparse_csr_spmm.py",
    "CUSPARSE_BSR":  "python3 operators/cusparse_bsr_spmm.py",
    "SMAT":          "python3 operators/smat_spmm.py",
    "DTC":           "python3 operators/dtc_spmm.py",
    "FlashSparse":   "python3 operators/flashsparse_spmm.py",
    "AccSpMM":       "python3 operators/accspmm_spmm.py",
    "BLEST_BFS":     "python3 operators/blest_bfs.py",
}

BLOCK_SIZE_KERNELS = {"CUSPARSE_BSR", "SMAT"}  # kernels needing --blocksize 32
BFS_KERNELS = {"BLEST_BFS"}  # kernels that use --n-sources instead of --n-cols

SLURM_COMMON = (
    "--partition=short --time=00:02:00 --cpus-per-task=1 "
    "--ntasks=1 --nodes=1 --account=flavio.vella"
)

N_COLS = 32
BLOCK_SIZE = 32
N_SOURCES = 64
POLL_INTERVAL = 10  # seconds

# ── Timing regex patterns (same as parse_results.py) ─────────────────────────

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
TIMER_PATTERN = re.compile(r"<Timer>\[(.*?)\]\s+([0-9.]+)\s+ms")
GROOT_TIMER_PATTERN = re.compile(r"\[KNN_MST_DFS\]\s+Reordering time \(ms\):\s+([0-9.]+)")
SPARTA_TIMER_PATTERN = re.compile(r"^timer:\s+([0-9.eE+\-]+)", re.MULTILINE)

TIMER_LABEL_ALIASES = {"spmm": "operation"}


# ── Preprocess snippets ───────────────────────────────────────────────────────

DEFAULT_PREPROCESS = """\
source ~/.venv/bin/activate
module load CUDA/
module load GCC/13.3.0
"""

DTC_PREPROCESS = f"source {PROJECT_ROOT}/operators/dtc_preprocess.sh"
FLASH_PREPROCESS = f"source {PROJECT_ROOT}/operators/flashsparse_preprocess.sh"
CONDA_INIT = """\
if [ -f /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh ]; then
    source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -n "$CONDA_EXE" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found" >&2; exit 1
fi"""

GROOT_PREPROCESS = f"""\
{CONDA_INIT}
conda activate GROOT"""

TCA_PREPROCESS = f"""\
{CONDA_INIT}
conda activate DTC-LSH
module load CUDA/
module load GCC/13.3.0"""


BLEST_PREPROCESS = f"""\
source ~/.venv/bin/activate
module load CUDA/
module load GCC/13.3.0"""

ACCORDER_PREPROCESS = f"""\
source {PROJECT_ROOT}/MtxPerm/ACCORDER/accorder_preprocess.sh"""


def kernel_preprocess(kernel: str) -> str:
    if kernel == "DTC":
        return DTC_PREPROCESS
    if kernel == "FlashSparse":
        return FLASH_PREPROCESS
    if kernel == "BLEST_BFS":
        return BLEST_PREPROCESS
    return DEFAULT_PREPROCESS


# ── Permutation commands ──────────────────────────────────────────────────────

def perm_command(algo: str, mtx: str, perm_path: str) -> str:
    cmds = {
        "GROOT_reorder":   f"python3 MtxPerm/GROOT/reorder.py {mtx} {perm_path}",
        "random1D":        f"python3 MtxPerm/RANDOM/GB_random_permutation_1d.py {mtx} {perm_path}",
        "random2D":        f"python3 MtxPerm/RANDOM/GB_random_permutation_2d.py {mtx} {perm_path}",
        "SB_degree":       f"./MtxPerm/SPARSEBASE/build/degree_perm {mtx} {perm_path}",
        "SB_gray":         f"./MtxPerm/SPARSEBASE/build/gray_perm {mtx} {perm_path}",
        "SB_rcm":          f"./MtxPerm/SPARSEBASE/build/rcm_perm {mtx} {perm_path}",
        "SB_metis":        f"./MtxPerm/SPARSEBASE/build/metis_part_perm {mtx} {perm_path} --nparts 128",
        "SB_amd":          f"./MtxPerm/SPARSEBASE/build/amd_perm {mtx} {perm_path}",
        "SB_rabbit":       f"./MtxPerm/SPARSEBASE/build/rabbit_perm {mtx} {perm_path}",
        "SB_patoh":        f"./MtxPerm/SPARSEBASE/build/patoh_perm {mtx} {perm_path} --nparts 128",
        "SB_slashburn":    f"./MtxPerm/SPARSEBASE/build/slashburn_perm {mtx} {perm_path}",
        "SPARTA_reorder":  f"python3 MtxPerm/SPARTA/reorder.py {mtx} {perm_path} --block-size 32 --tau 0.5",
        "ACCORDER_reorder": f"./MtxPerm/ACCORDER/build/accorder_perm {mtx} {perm_path}",
        "TCA_reorder":     f"python3 MtxPerm/DTC-LSH/reorder.py {mtx} {perm_path}",
    }
    return cmds[algo]


def perm_preprocess(algo: str) -> str:
    if algo == "GROOT_reorder":
        return GROOT_PREPROCESS
    if algo == "TCA_reorder":
        return TCA_PREPROCESS
    if algo == "ACCORDER_reorder":
        return ACCORDER_PREPROCESS
    return "source ~/.venv/bin/activate"


# ── Script generation ─────────────────────────────────────────────────────────

def write_job_script(path: Path, preprocess: str, command: str) -> None:
    """Write a SLURM batch script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"""\
#!/bin/bash
set -euo pipefail
cd {PROJECT_ROOT}
{preprocess}
{command}
"""
    path.write_text(script)
    path.chmod(0o755)


def submit_job(script: Path, log: Path, gpu: bool = False) -> str:
    """Submit a job via sbatch and return the job ID."""
    gpu_flag = "--gres=gpu:1" if gpu else "--gres=gpu:0"
    cmd = f"sbatch {SLURM_COMMON} {gpu_flag} --output={log} --error={log} {script}"
    result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def wait_for_jobs(job_ids: list[str], phase_name: str) -> None:
    """Poll sacct until all jobs are finished."""
    if not job_ids:
        return
    id_str = ",".join(job_ids)
    print(f"  Waiting for {len(job_ids)} {phase_name} jobs ...", flush=True)
    while True:
        time.sleep(POLL_INTERVAL)
        result = subprocess.run(
            ["sacct", "-j", id_str, "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True,
        )
        states = [s.strip() for s in result.stdout.strip().splitlines() if s.strip()]
        if not states:
            continue
        # Filter out sub-steps (e.g. "batch" steps) — keep only main job states
        active = [s for s in states if s not in ("COMPLETED", "FAILED", "TIMEOUT",
                                                   "CANCELLED", "OUT_OF_MEMORY",
                                                   "NODE_FAIL")]
        if all(s in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED",
                      "OUT_OF_MEMORY", "NODE_FAIL", "") for s in states):
            break
    print(f"  All {phase_name} jobs finished.", flush=True)


# ── SpMM command builder ─────────────────────────────────────────────────────

def spmm_command(kernel: str, mtx: str, perm_path: str | None = None,
                 perm_type: str | None = None) -> str:
    base = KERNELS[kernel]
    parts = [base, mtx]
    if kernel in BFS_KERNELS:
        parts.append(f"--n-sources {N_SOURCES}")
    else:
        if kernel in BLOCK_SIZE_KERNELS:
            parts.append(f"--blocksize {BLOCK_SIZE}")
        parts.append(f"--n-cols {N_COLS}")
    if perm_path and perm_type:
        parts.append(f"--perm {perm_path} --perm-type {perm_type}")
    return " ".join(parts)


# ── Log parsing (mirrors parse_results.py) ───────────────────────────────────

def _clean(text: str) -> str:
    """Strip ANSI escape codes."""
    if '\x1b' in text or '\x1B' in text:
        return ANSI_ESCAPE.sub('', text)
    return text


def parse_spmm_timers(log_path: Path) -> dict[str, float]:
    """Extract all <Timer>[label] values from an SpMM log. Returns {label: ms}."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return {}
    content = _clean(log_path.read_text())
    timers = {}
    for m in TIMER_PATTERN.finditer(content):
        label = TIMER_LABEL_ALIASES.get(m.group(1).lower(), m.group(1).lower())
        timers[label] = float(m.group(2))
    return timers


def parse_perm_timers(algo: str, log_path: Path) -> dict[str, float]:
    """Extract reordering/loading timers from a perm log. Returns {label: ms}."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return {}
    content = _clean(log_path.read_text())
    timers = {}

    # Standard <Timer> format (SparseBase, random)
    for m in TIMER_PATTERN.finditer(content):
        timers[m.group(1).lower()] = float(m.group(2))

    # GROOT format
    if "reordering" not in timers:
        groot_m = GROOT_TIMER_PATTERN.search(content)
        if groot_m:
            timers["reordering"] = float(groot_m.group(1))

    # SPARTA format (microseconds → ms)
    if "reordering" not in timers:
        sparta_m = SPARTA_TIMER_PATTERN.search(content)
        if sparta_m:
            timers["reordering"] = float(sparta_m.group(1)) / 1000.0

    return timers


# ── Result checking ──────────────────────────────────────────────────────────

def validate_permutation(perm_file: Path, expected_n: int) -> tuple[bool, str]:
    """Check that a .perm file contains a proper permutation of 1..N.
    Returns (is_valid, error_message)."""
    try:
        text = perm_file.read_text().strip()
    except Exception as e:
        return False, f"read error: {e}"

    lines = text.splitlines()
    # Each line is a permutation (1 line = symmetric/row, 2 lines = asymmetric)
    for i, line in enumerate(lines):
        vals = line.split()
        if len(vals) != expected_n:
            return False, f"line {i+1}: expected {expected_n} values, got {len(vals)}"
        try:
            ints = [int(v) for v in vals]
        except ValueError:
            return False, f"line {i+1}: non-integer values"
        if min(ints) < 1 or max(ints) > expected_n:
            return False, f"line {i+1}: values out of range [1, {expected_n}]"
        if len(set(ints)) != expected_n:
            return False, f"line {i+1}: {expected_n - len(set(ints))} duplicate values"
    return True, ""


def check_perm_result(algo: str, run_dir: Path, mtx_name: str,
                      job_ids: dict[str, str],
                      mtx_dim: int | None = None) -> str:
    perm_file = run_dir / "perms" / algo / f"{mtx_name}.perm"
    if not perm_file.exists() or perm_file.stat().st_size == 0:
        return check_slurm_state(job_ids.get(f"perm_{algo}"), "FAIL")
    if mtx_dim is not None:
        valid, err = validate_permutation(perm_file, mtx_dim)
        if not valid:
            return f"BAD_PERM"
    return "PASS"


def check_spmm_result(log_path: Path, job_id: str | None) -> str:
    if not log_path.exists():
        return "PENDING" if job_id is None else "FAIL"
    if log_path.stat().st_size == 0:
        return check_slurm_state(job_id, "FAIL")
    content = log_path.read_text()
    if "<Timer>" in content:
        return "PASS"
    return check_slurm_state(job_id, "FAIL")


def check_slurm_state(job_id: str | None, default: str = "FAIL") -> str:
    if job_id is None:
        return default
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=10,
        )
        states = [s.strip() for s in result.stdout.strip().splitlines() if s.strip()]
        if any("TIMEOUT" in s for s in states):
            return "TIMEOUT"
        if any("CANCELLED" in s for s in states):
            return "CANCELLED"
        if any("OUT_OF_MEMORY" in s for s in states):
            return "OOM"
    except Exception:
        pass
    return default


def _fmt_ms(val: float | None) -> str:
    """Format a millisecond value for display, or '-' if missing."""
    if val is None:
        return "-"
    if val >= 1000:
        return f"{val:.0f}"
    if val >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


# ── Matrix validation ─────────────────────────────────────────────────────────

def validate_matrix(mtx_path: Path) -> int:
    """Verify the matrix is square and symmetric (Matrix Market header check).
    Returns the matrix dimension (number of rows/cols)."""
    with open(mtx_path) as f:
        # First line: %%MatrixMarket matrix coordinate <type> <symmetry>
        banner = f.readline().strip()
        if not banner.startswith("%%MatrixMarket"):
            print(f"ERROR: {mtx_path} is not a valid Matrix Market file")
            sys.exit(1)

        parts = banner.split()
        if len(parts) < 5:
            print(f"ERROR: Malformed MatrixMarket banner: {banner}")
            sys.exit(1)

        symmetry = parts[4].lower()
        if symmetry not in ("symmetric", "skew-symmetric", "hermitian"):
            print(f"ERROR: Matrix is not symmetric (symmetry={parts[4]})")
            print("       This test pipeline requires a square symmetric matrix.")
            sys.exit(1)

        # Skip comment lines, read size line
        for line in f:
            if not line.startswith("%"):
                size_parts = line.strip().split()
                rows, cols = int(size_parts[0]), int(size_parts[1])
                if rows != cols:
                    print(f"ERROR: Matrix is not square ({rows} x {cols})")
                    sys.exit(1)
                print(f"Matrix validated: {rows}x{cols}, {symmetry}, nnz={size_parts[2]}")
                return rows

    print(f"ERROR: Could not find size line in {mtx_path}")
    sys.exit(1)


# ── Main logic ───────────────────────────────────────────────────────────────

def find_or_create_run_dir(mtx_name: str, check_only: bool) -> Path:
    """Find the most recent run dir (for --check-only) or create a new one."""
    test_runs = PROJECT_ROOT / "test_runs"
    if check_only:
        # Find the most recent matching directory
        candidates = sorted(test_runs.glob(f"{mtx_name}_*"), reverse=True)
        if not candidates:
            print(f"ERROR: No existing run found for {mtx_name} in test_runs/")
            sys.exit(1)
        return candidates[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = test_runs / f"{mtx_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_job_ids(run_dir: Path) -> dict[str, str]:
    """Load saved job IDs from a previous run."""
    id_file = run_dir / "job_ids.txt"
    ids = {}
    if id_file.exists():
        for line in id_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                ids[k.strip()] = v.strip()
    return ids


def save_job_ids(run_dir: Path, job_ids: dict[str, str]) -> None:
    id_file = run_dir / "job_ids.txt"
    with open(id_file, "w") as f:
        for k, v in job_ids.items():
            f.write(f"{k}={v}\n")


def run_pipeline(mtx_path: Path, check_only: bool,
                 only_perms: list[str] | None = None) -> None:
    mtx_abs = mtx_path.resolve()
    mtx_name = mtx_abs.stem

    if not mtx_abs.exists():
        print(f"ERROR: Matrix file not found: {mtx_abs}")
        sys.exit(1)

    mtx_dim = validate_matrix(mtx_abs)

    run_dir = find_or_create_run_dir(mtx_name, check_only)
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    perms_dir = run_dir / "perms"

    print(f"Matrix:  {mtx_abs}")
    print(f"Run dir: {run_dir}")
    print()

    if check_only:
        job_ids = load_job_ids(run_dir)
        print_summary(run_dir, mtx_name, job_ids, mtx_dim)
        return

    job_ids: dict[str, str] = {}  # key -> slurm job id

    algos_to_run = only_perms if only_perms else PERM_ALGORITHMS

    # ── Phase 1: Permutations ─────────────────────────────────────────────
    print(f"=== Phase 1: Permutation Generation ({len(algos_to_run)} CPU jobs) ===")
    phase1_ids = []
    for algo in algos_to_run:
        perm_out = perms_dir / algo / f"{mtx_name}.perm"
        perm_out.parent.mkdir(parents=True, exist_ok=True)
        script = jobs_dir / f"perm_{algo}.sh"
        log = jobs_dir / f"perm_{algo}.log"
        write_job_script(script, perm_preprocess(algo),
                         perm_command(algo, str(mtx_abs), str(perm_out)))
        jid = submit_job(script, log, gpu=(algo == "TCA_reorder"))
        key = f"perm_{algo}"
        job_ids[key] = jid
        phase1_ids.append(jid)
        print(f"  Submitted {algo}: job {jid}")

    wait_for_jobs(phase1_ids, "permutation")
    print()

    if only_perms:
        # Print perm-only summary and exit early
        save_job_ids(run_dir, job_ids)
        print(f"=== PERMUTATION RESULTS (--only-perms) ===")
        print(f"  {'Algorithm':<20s} {'Status':<10s} {'Reorder (ms)':>14s} {'Load (ms)':>12s}")
        print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*12}")
        for algo in algos_to_run:
            status = check_perm_result(algo, run_dir, mtx_name, job_ids, mtx_dim)
            log = jobs_dir / f"perm_{algo}.log"
            timers = parse_perm_timers(algo, log)
            t_reorder = _fmt_ms(timers.get("reordering"))
            t_load = _fmt_ms(timers.get("loading"))
            print(f"  {algo:<20s} {status:<10s} {t_reorder:>14s} {t_load:>12s}")
        print()
        return

    # ── Phase 2: SpMM no reorder ──────────────────────────────────────────
    print("=== Phase 2: SpMM NO_REORDER (6 GPU jobs) ===")
    phase2_ids = []
    for kernel in KERNELS:
        script = jobs_dir / f"{kernel}_NO_REORDER.sh"
        log = jobs_dir / f"{kernel}_NO_REORDER.log"
        cmd = spmm_command(kernel, str(mtx_abs))
        write_job_script(script, kernel_preprocess(kernel), cmd)
        jid = submit_job(script, log, gpu=True)
        key = f"{kernel}_NO_REORDER"
        job_ids[key] = jid
        phase2_ids.append(jid)
        print(f"  Submitted {kernel}: job {jid}")

    wait_for_jobs(phase2_ids, "NO_REORDER")
    print()

    # ── Phase 3: SpMM ROW reorder ─────────────────────────────────────────
    n_reorder_jobs = len(KERNELS) * len(PERM_ALGORITHMS)
    print(f"=== Phase 3: SpMM ROW reorder ({n_reorder_jobs} GPU jobs) ===")
    phase3_ids = []
    for kernel in KERNELS:
        for algo in PERM_ALGORITHMS:
            perm_path = perms_dir / algo / f"{mtx_name}.perm"
            script = jobs_dir / f"{kernel}_ROW_{algo}.sh"
            log = jobs_dir / f"{kernel}_ROW_{algo}.log"
            cmd = spmm_command(kernel, str(mtx_abs),
                               str(perm_path), "ROW")
            write_job_script(script, kernel_preprocess(kernel), cmd)
            jid = submit_job(script, log, gpu=True)
            key = f"{kernel}_ROW_{algo}"
            job_ids[key] = jid
            phase3_ids.append(jid)
    print(f"  Submitted {len(phase3_ids)} ROW jobs")

    wait_for_jobs(phase3_ids, "ROW")
    print()

    # ── Phase 4: SpMM SYMMETRIC reorder ───────────────────────────────────
    print(f"=== Phase 4: SpMM SYMMETRIC reorder ({n_reorder_jobs} GPU jobs) ===")
    phase4_ids = []
    for kernel in KERNELS:
        for algo in PERM_ALGORITHMS:
            perm_path = perms_dir / algo / f"{mtx_name}.perm"
            script = jobs_dir / f"{kernel}_SYMMETRIC_{algo}.sh"
            log = jobs_dir / f"{kernel}_SYMMETRIC_{algo}.log"
            cmd = spmm_command(kernel, str(mtx_abs),
                               str(perm_path), "SYMMETRIC")
            write_job_script(script, kernel_preprocess(kernel), cmd)
            jid = submit_job(script, log, gpu=True)
            key = f"{kernel}_SYMMETRIC_{algo}"
            job_ids[key] = jid
            phase4_ids.append(jid)
    print(f"  Submitted {len(phase4_ids)} SYMMETRIC jobs")

    wait_for_jobs(phase4_ids, "SYMMETRIC")
    print()

    save_job_ids(run_dir, job_ids)
    print_summary(run_dir, mtx_name, job_ids, mtx_dim)


# ── Summary printer ──────────────────────────────────────────────────────────

def print_summary(run_dir: Path, mtx_name: str, job_ids: dict[str, str],
                   mtx_dim: int | None = None) -> None:
    jobs_dir = run_dir / "jobs"
    kernel_names = list(KERNELS.keys())
    short_names = {
        "ASPT": "ASPT", "CUSPARSE_CSR": "CSR", "CUSPARSE_BSR": "BSR",
        "SMAT": "SMAT", "DTC": "DTC", "FlashSparse": "FLASH",
        "AccSpMM": "ACCSPMM", "BLEST_BFS": "BLEST",
    }
    COL_W = 10  # column width for grid tables

    # ── Permutation results with timings ──────────────────────────────────
    print("=== PERMUTATION RESULTS ===")
    print(f"  {'Algorithm':<20s} {'Status':<10s} {'Reorder (ms)':>14s} {'Load (ms)':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*12}")
    for algo in PERM_ALGORITHMS:
        status = check_perm_result(algo, run_dir, mtx_name, job_ids, mtx_dim)
        log = jobs_dir / f"perm_{algo}.log"
        timers = parse_perm_timers(algo, log)
        t_reorder = _fmt_ms(timers.get("reordering"))
        t_load = _fmt_ms(timers.get("loading"))
        print(f"  {algo:<20s} {status:<10s} {t_reorder:>14s} {t_load:>12s}")
    print()

    # ── SpMM NO_REORDER with timings ──────────────────────────────────────
    print("=== SpMM RESULTS (NO_REORDER) ===")
    print(f"  {'Kernel':<20s} {'Status':<10s} {'Op (ms)':>10s} {'Perm (ms)':>10s} {'Load (ms)':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for kernel in kernel_names:
        log = jobs_dir / f"{kernel}_NO_REORDER.log"
        jid = job_ids.get(f"{kernel}_NO_REORDER")
        status = check_spmm_result(log, jid)
        timers = parse_spmm_timers(log)
        t_op = _fmt_ms(timers.get("operation"))
        t_perm = _fmt_ms(timers.get("permutation"))
        t_load = _fmt_ms(timers.get("loading"))
        print(f"  {kernel:<20s} {status:<10s} {t_op:>10s} {t_perm:>10s} {t_load:>10s}")
    print()

    # ── SpMM ROW — status grid ────────────────────────────────────────────
    header = f"  {'':20s}" + "".join(f"{short_names[k]:>{COL_W}s}" for k in kernel_names)
    print("=== SpMM RESULTS (ROW) ===")
    print(header)
    for algo in PERM_ALGORITHMS:
        row = f"  {algo:20s}"
        for kernel in kernel_names:
            log = jobs_dir / f"{kernel}_ROW_{algo}.log"
            jid = job_ids.get(f"{kernel}_ROW_{algo}")
            status = check_spmm_result(log, jid)
            row += f"{status:>{COL_W}s}"
        print(row)
    print()

    # ── SpMM ROW — operation time grid (ms) ───────────────────────────────
    print("=== SpMM TIMINGS (ROW) — operation time in ms ===")
    print(header)
    for algo in PERM_ALGORITHMS:
        row = f"  {algo:20s}"
        for kernel in kernel_names:
            log = jobs_dir / f"{kernel}_ROW_{algo}.log"
            timers = parse_spmm_timers(log)
            row += f"{_fmt_ms(timers.get('operation')):>{COL_W}s}"
        print(row)
    print()

    # ── SpMM SYMMETRIC — status grid ──────────────────────────────────────
    print("=== SpMM RESULTS (SYMMETRIC) ===")
    print(header)
    for algo in PERM_ALGORITHMS:
        row = f"  {algo:20s}"
        for kernel in kernel_names:
            log = jobs_dir / f"{kernel}_SYMMETRIC_{algo}.log"
            jid = job_ids.get(f"{kernel}_SYMMETRIC_{algo}")
            status = check_spmm_result(log, jid)
            row += f"{status:>{COL_W}s}"
        print(row)
    print()

    # ── SpMM SYMMETRIC — operation time grid (ms) ─────────────────────────
    print("=== SpMM TIMINGS (SYMMETRIC) — operation time in ms ===")
    print(header)
    for algo in PERM_ALGORITHMS:
        row = f"  {algo:20s}"
        for kernel in kernel_names:
            log = jobs_dir / f"{kernel}_SYMMETRIC_{algo}.log"
            timers = parse_spmm_timers(log)
            row += f"{_fmt_ms(timers.get('operation')):>{COL_W}s}"
        print(row)
    print()

    # ── Totals ────────────────────────────────────────────────────────────
    n_perms = len(PERM_ALGORITHMS)
    n_kernels = len(KERNELS)
    total = n_perms + n_kernels + 2 * (n_kernels * n_perms)
    pass_count = 0
    fail_count = 0
    other_count = 0

    for algo in PERM_ALGORITHMS:
        s = check_perm_result(algo, run_dir, mtx_name, job_ids, mtx_dim)
        if s == "PASS": pass_count += 1
        elif s == "FAIL": fail_count += 1
        else: other_count += 1

    for kernel in kernel_names:
        log = jobs_dir / f"{kernel}_NO_REORDER.log"
        s = check_spmm_result(log, job_ids.get(f"{kernel}_NO_REORDER"))
        if s == "PASS": pass_count += 1
        elif s == "FAIL": fail_count += 1
        else: other_count += 1

    for perm_type in ("ROW", "SYMMETRIC"):
        for kernel in kernel_names:
            for algo in PERM_ALGORITHMS:
                log = jobs_dir / f"{kernel}_{perm_type}_{algo}.log"
                s = check_spmm_result(log, job_ids.get(f"{kernel}_{perm_type}_{algo}"))
                if s == "PASS": pass_count += 1
                elif s == "FAIL": fail_count += 1
                else: other_count += 1

    print(f"TOTAL: {pass_count} PASS / {fail_count} FAIL / {other_count} OTHER  (out of {total})")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the full reordering + SpMM pipeline on a single matrix."
    )
    parser.add_argument("matrix", type=Path, help="Path to .mtx matrix file")
    parser.add_argument("--check-only", action="store_true",
                        help="Re-read logs from the most recent run, don't launch jobs")
    parser.add_argument("--only-perms", nargs="+", metavar="ALGO",
                        help="Only run these perm algorithms (skip SpMM phases)")
    args = parser.parse_args()

    if args.only_perms:
        # Validate algorithm names
        for a in args.only_perms:
            if a not in PERM_ALGORITHMS:
                parser.error(f"Unknown algorithm '{a}'. Choose from: {', '.join(PERM_ALGORITHMS)}")

    run_pipeline(args.matrix, args.check_only, only_perms=args.only_perms)


if __name__ == "__main__":
    main()
