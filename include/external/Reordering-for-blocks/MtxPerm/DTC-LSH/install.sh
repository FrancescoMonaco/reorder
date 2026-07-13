#!/bin/bash
#
# Install script for DTC-LSH (TCA reordering from DTC-SpMM, ASPLOS'24)
#
# Creates a dedicated conda environment "DTC-LSH" with all dependencies:
#   - cugraph, cudf (GPU graph/dataframe libraries)
#   - datasketch (MinHash/LSH)
#   - cupy (GPU arrays)
#   - libMHCUDA / minhashcuda (GPU-accelerated MinHash)
#   - scipy, numpy (matrix I/O)
#
# Based on DTC-SpMM README section 7: "Use TCA-reordering"
# https://github.com/HPMLL/DTC-SpMM_ASPLOS24
#
# Usage:
#   ./install.sh              # Standard install
#   ./install.sh --clean      # Remove env and rebuild from scratch
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
ENV_NAME="DTC-LSH"
MINHASH_DIR="${SCRIPT_DIR}/minhashcuda_build"

# Parse arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
    esac
done

echo "=========================================="
echo "DTC-LSH (TCA Reordering) Installation"
echo "=========================================="

# Initialize conda
if [ -f /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh ]; then
    source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Load modules
module load CUDA/ 2>/dev/null || true
module load GCC/13.3.0 2>/dev/null || true

# ============================================
# Step 1: Create conda environment
# ============================================
echo ""
echo "[1/3] Setting up conda environment '${ENV_NAME}'..."

if [ "$CLEAN_BUILD" = true ]; then
    echo "Removing existing environment..."
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    rm -rf "${MINHASH_DIR}"
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists, activating..."
else
    echo "Creating environment '${ENV_NAME}' (Python 3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"
echo "Activated: ${ENV_NAME} ($(python3 --version))"

# ============================================
# Step 2: Install pip packages
# ============================================
echo ""
echo "[2/3] Installing pip packages..."

echo "  Installing scipy, numpy..."
pip install numpy scipy

echo "  Installing datasketch..."
pip install datasketch

echo "  Installing cugraph-cu12..."
pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

echo "  Installing cudf-cu12..."
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12

echo "  Installing cupy-cuda12x..."
pip install cupy-cuda12x

# ============================================
# Step 3: Build minhashcuda (libMHCUDA)
# ============================================
echo ""
echo "[3/3] Building minhashcuda (libMHCUDA)..."

if [ -d "${MINHASH_DIR}" ] && python3 -c "import libMHCUDA" 2>/dev/null; then
    echo "minhashcuda already installed, skipping build..."
else
    if [ ! -d "${MINHASH_DIR}" ]; then
        git clone https://github.com/src-d/minhashcuda.git "${MINHASH_DIR}"
    fi
    cd "${MINHASH_DIR}"
    NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-I${NUMPY_INC}" \
          .
    make -j4
    # setup.py install fails due to missing libpython3.X.a in conda envs.
    # Manually install the built .so into site-packages instead.
    SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
    cp libMHCUDA.so "${SITE_PKGS}/"
    cd "${SCRIPT_DIR}"
fi

# ============================================
# Verification
# ============================================
echo ""
echo "Verifying installation..."

PASS=true
for mod in cugraph cudf datasketch libMHCUDA cupy scipy numpy; do
    if python3 -c "import ${mod}" 2>/dev/null; then
        echo "  ${mod}: OK"
    else
        echo "  ${mod}: FAILED"
        PASS=false
    fi
done

echo ""
if [ "$PASS" = true ]; then
    echo "=========================================="
    echo "DTC-LSH installation complete!"
    echo "=========================================="
else
    echo "=========================================="
    echo "WARNING: Some imports failed (see above)."
    echo "libMHCUDA may require a GPU node to build."
    echo "=========================================="
fi
echo ""
echo "Usage:"
echo "  conda activate ${ENV_NAME}"
echo "  python3 MtxPerm/DTC-LSH/reorder.py <matrix.mtx> <output.perm> [--thres 16]"
