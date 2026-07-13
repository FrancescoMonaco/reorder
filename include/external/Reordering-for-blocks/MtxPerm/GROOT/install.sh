#!/bin/bash
#
# Install script for Groot (Graph-Centric Row Reordering) - Python Version
# Clones the official Groot-EuroSys25 repository and creates a conda environment.
#
# Reference:
#     Chen et al., "Groot: Graph-Centric Row Reordering with Tree for
#     Sparse Matrix Multiplications on Tensor Cores", EuroSys 2025
#
# Repository: https://github.com/yuang-chen/Groot-EuroSys25
#
# Requirements:
#   - Conda (miniconda/anaconda)
#
# Usage:
#   ./install.sh           # Create/update conda environment
#   ./install.sh --clean   # Remove and recreate environment
#

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/Groot-EuroSys25"
ENV_NAME="GROOT"

# Parse arguments
CLEAN=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean) CLEAN=true ;;
    esac
    shift
done

echo "=========================================="
echo "Groot Installation Script (Python Version)"
echo "=========================================="
echo "Target directory: ${REPO_DIR}"
echo "Conda environment: ${ENV_NAME}"
echo ""

# Initialize conda
if [ -f /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh ]; then
    source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda. Please install miniconda/anaconda."
    exit 1
fi

# Step 1: Clone repository if not present
echo "[1/3] Checking repository..."
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Groot repository..."
    git clone https://github.com/yuang-chen/Groot-EuroSys25.git "$REPO_DIR"
    echo "Cloned Groot"
else
    echo "Repository already cloned at $REPO_DIR"
fi

# Step 2: Create/update conda environment
echo ""
echo "[2/3] Setting up conda environment..."

if [ "$CLEAN" = true ]; then
    echo "Removing existing environment..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
    conda activate "$ENV_NAME"
    pip install --upgrade pynndescent scipy numpy
else
    echo "Creating new environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.11 -y
    conda activate "$ENV_NAME"
    pip install pynndescent scipy numpy
fi

# Step 3: Verify installation
echo ""
echo "[3/3] Verifying installation..."

cd "$SCRIPT_DIR"
python -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from pynndescent import NNDescent
from sparse import sparse_jaccard
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
print('All dependencies verified successfully')
"

echo ""
echo "=========================================="
echo "Groot Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Usage:"
echo "  python3 MtxPerm/GROOT/reorder.py <input.mtx> <output.perm>"
echo ""
echo "Or source the preprocess script before running:"
echo "  source MtxPerm/GROOT/groot_preprocess.sh"
echo ""
