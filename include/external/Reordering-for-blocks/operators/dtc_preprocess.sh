#!/bin/bash
module purge
module load CUDA/
module load GCC/13.3.0
source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate DTCSpMM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/paolo.syloslabini/ReorderingSurvey-2026/operators/DTC-SpMM/third_party/sputnik/build/sputnik
