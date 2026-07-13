#!/bin/bash
#SBATCH --job-name=test_tca
#SBATCH --output=test_tca.out
#SBATCH --error=test_tca.out
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=flavio.vella

set -euo pipefail
cd /home/paolo.syloslabini/ReorderingSurvey-2026

PYTHON=/home/paolo.syloslabini/.conda/envs/DTC-LSH/bin/python
export PATH=/home/paolo.syloslabini/.conda/envs/DTC-LSH/bin:$PATH

module load CUDA/
module load GCC/13.3.0

MTX=datasets/large-matrices/SuiteSparse_1000000_100000000_10000/Zaoui/kkt_power/kkt_power.mtx
PERM=/tmp/test_tca_kkt_power.perm

echo "=== Test TCA reorder on kkt_power (2M rows) ==="
echo "Python: $PYTHON"
$PYTHON --version
echo ""

$PYTHON MtxPerm/DTC-LSH/reorder.py "$MTX" "$PERM" --thres 16

echo ""
echo "=== Verify .perm output ==="
if [ -f "$PERM" ]; then
    echo "Perm file exists: $PERM"
    wc -l "$PERM"
    head -c 200 "$PERM"
    echo ""
    echo "Number of elements: $(wc -w < "$PERM")"
else
    echo "ERROR: Perm file not created!"
    exit 1
fi

echo ""
echo "=== DONE ==="
