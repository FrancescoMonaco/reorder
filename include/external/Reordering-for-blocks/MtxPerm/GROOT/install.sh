#!/bin/bash
#
# Install script for Groot (Graph-Centric Row Reordering) - Python Version
# Modified to use micromamba with fully isolated binary packages.
#
# Reference:
#     Chen et al., "Groot: Graph-Centric Row Reordering with Tree for
#     Sparse Matrix Multiplications on Tensor Cores", EuroSys 2025
#
# Repository: https://github.com/yuang-chen/Groot-EuroSys25
#
# Usage:
#   ./install.sh           # Create/update micromamba environment
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
echo "Groot Installation Script (Micromamba Version)"
echo "=========================================="
echo "Target directory: ${REPO_DIR}"
echo "Micromamba environment: ${ENV_NAME}"
echo ""

# Initialize micromamba for this bash subshell session
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell bash)"
elif [ -d "$HOME/micromamba" ]; then
    eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"
else
    echo "ERROR: micromamba binary could not be found."
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

# Step 2: Create/update micromamba environment
echo ""
echo "[2/3] Setting up micromamba environment..."

if [ "$CLEAN" = true ]; then
    echo "Removing existing environment..."
    micromamba env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

# Check if environment exists
if micromamba env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Updating packages..."
    micromamba activate "$ENV_NAME"
    micromamba install -c conda-forge pynndescent scipy numpy -y
else
    echo "Creating new environment '$ENV_NAME' (Python 3.11)..."
    # Pulling heavy math/graph libraries from conda-forge prevents pip local-compilation crashes
    micromamba create -n "$ENV_NAME" python=3.11 pynndescent scipy numpy -c conda-forge -y
    micromamba activate "$ENV_NAME"
fi

# Step 3: Verify installation
echo ""
echo "[3/3] Verifying installation..."

cd "$SCRIPT_DIR"
python3 -c "
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
echo "  micromamba activate $ENV_NAME"
echo ""
echo "Usage:"
echo "  python3 MtxPerm/GROOT/reorder.py <input.mtx> <output.perm>"
echo ""
echo "Or source the preprocess script before running:"
echo "  source MtxPerm/GROOT/groot_preprocess.sh"
echo ""