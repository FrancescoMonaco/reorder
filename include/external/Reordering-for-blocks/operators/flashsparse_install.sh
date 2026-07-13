#!/bin/bash
# Install FlashSparse CUDA kernels
# Run from the project root directory

set -e

# Load required modules
module purge
module load CUDA/12.1.1
# Note: Using system GCC 11.5.0, not module GCC (which causes assembler issues)

# Activate conda environment
source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate FlashSparse

# Set CUDA architecture (A100 = 8.0)
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

# Navigate to FlashSparse source
cd "$(dirname "$0")/FlashSparse/FlashSparse"

# Clean previous builds
rm -rf build dist *.egg-info
rm -rf Block_gpu/build Block/build SpMM/build SDDMM/build

# Build and install using pip (no build isolation to use conda's torch)
pip install --no-build-isolation -e .

echo "FlashSparse installation complete!"
echo "Test with: python -c 'import torch; import FS_Block_gpu; import FS_SpMM; print(\"OK\")'"
