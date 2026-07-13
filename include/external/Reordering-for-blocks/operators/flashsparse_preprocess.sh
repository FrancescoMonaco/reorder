#!/bin/bash
module purge
module load CUDA/12.1.1
source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate FlashSparse
