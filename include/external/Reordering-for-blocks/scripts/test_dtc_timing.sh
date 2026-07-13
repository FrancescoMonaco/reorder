#!/bin/bash
#SBATCH --job-name=dtc_timing_test
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=flavio.vella
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/paolo.syloslabini/ReorderingSurvey-2026/test_dtc_timing.out

BASEDIR=/home/paolo.syloslabini/ReorderingSurvey-2026

# Run from a temp directory so DTC writes its CSV there (isolated from other jobs)
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

# Setup DTC environment
source "$BASEDIR/operators/dtc_preprocess.sh"

MTX="$BASEDIR/datasets/all-matrices/SuiteSparse_10000_1000000_10000/PowerSystem/power197k/power197k.mtx"

echo "=== Running from: $TMPDIR ==="
echo ""

echo "=== Our wrapper output ==="
python3 "$BASEDIR/operators/dtc_spmm.py" "$MTX" --n-cols 256 --n-iterations 5

echo ""
echo "=== DTC internal CSV ==="
cat DTCSpMM_exe_time_and_throughput.csv 2>/dev/null || echo "(no CSV produced)"

# Cleanup
rm -rf "$TMPDIR"
