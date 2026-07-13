#!/bin/bash
#
# Install script for AccSpMM (PPoPP 2025 TF32 Tensor Core SpMM)
# Clones the repository and builds the CUDA SpMM kernel
#
# Requirements:
#   - CUDA Toolkit >= 11.8
#   - nvcc compiler in PATH
#   - CMake 3.18+
#   - GCC
#
# Usage:
#   ./install_accspmm.sh                # Standard install
#   ./install_accspmm.sh --clean        # Clean and reinstall
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ACCSPMM_DIR="${SCRIPT_DIR}/AccSpMM"
ACCSPMM_REPO="https://github.com/YaoJianyu77/AccSpMM.git"

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
echo "AccSpMM Installation Script"
echo "=========================================="
echo "Target directory: ${ACCSPMM_DIR}"
echo "Repository: ${ACCSPMM_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/5] Checking prerequisites..."

# Try to load modules if not found (common on clusters)
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found, attempting to load CUDA module..."
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || module load CUDA/ 2>/dev/null || true
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please install CUDA Toolkit >= 11.8 and ensure nvcc is in PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "  CUDA found: $CUDA_VERSION"

if ! command -v gcc &> /dev/null; then
    echo "gcc not found, attempting to load GCC module..."
    module load GCC/13.3.0 2>/dev/null || module load gcc 2>/dev/null || true
fi

if ! command -v gcc &> /dev/null; then
    echo "ERROR: gcc not found in PATH"
    exit 1
fi
GCC_VERSION=$(gcc --version | head -1)
echo "  GCC found: $GCC_VERSION"

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found in PATH"
    echo "Please install CMake 3.18+"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -1)
echo "  CMake found: $CMAKE_VERSION"

# Step 2: Detect GPU architecture
echo ""
echo "[2/5] Detecting GPU architecture..."
GPU_ARCH=""
if python3 -c "import torch" 2>/dev/null; then
    GPU_ARCH=$(python3 -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
fi
if [ -z "$GPU_ARCH" ]; then
    # Try nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo "")
        case "$GPU_NAME" in
            *A100*) GPU_ARCH="80" ;;
            *A800*) GPU_ARCH="80" ;;
            *H100*) GPU_ARCH="90" ;;
            *RTX*4090*) GPU_ARCH="89" ;;
            *RTX*3090*) GPU_ARCH="86" ;;
            *) GPU_ARCH="80" ;;
        esac
    else
        GPU_ARCH="80"  # Default to A100
    fi
    echo "  GPU architecture: sm_${GPU_ARCH} (detected from GPU name or default)"
else
    echo "  GPU architecture: sm_${GPU_ARCH} (detected via PyTorch)"
fi

# Step 3: Clone repository
echo ""
echo "[3/5] Cloning repository..."
if [ "$CLEAN_BUILD" = true ] && [ -d "${ACCSPMM_DIR}" ]; then
    echo "  Removing existing directory for clean build..."
    rm -rf "${ACCSPMM_DIR}"
fi

if [ -d "${ACCSPMM_DIR}" ]; then
    echo "  Directory ${ACCSPMM_DIR} already exists. Pulling latest changes..."
    cd "${ACCSPMM_DIR}"
    git pull || echo "  Warning: git pull failed, continuing with existing code"
else
    git clone "${ACCSPMM_REPO}" "${ACCSPMM_DIR}"
    cd "${ACCSPMM_DIR}"
fi
echo "  Repository ready"

# Step 4: Build
echo ""
echo "[4/5] Building AccSpMM..."
cd "${ACCSPMM_DIR}"
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}"
make -j$(nproc)
cd "${ACCSPMM_DIR}"

echo "  Build completed"

# Step 5: Verify binary
echo ""
echo "[5/5] Verifying installation..."
BINARY_PATHS=(
    "${ACCSPMM_DIR}/mma"
    "${ACCSPMM_DIR}/build/mma"
)

BINARY_FOUND=""
for BINARY_PATH in "${BINARY_PATHS[@]}"; do
    if [ -f "${BINARY_PATH}" ]; then
        echo "  Binary found: ${BINARY_PATH}"
        BINARY_FOUND="${BINARY_PATH}"
        break
    fi
done

if [ -z "$BINARY_FOUND" ]; then
    echo "WARNING: Could not locate AccSpMM binary in expected locations"
    echo "Expected one of:"
    for BINARY_PATH in "${BINARY_PATHS[@]}"; do
        echo "  - ${BINARY_PATH}"
    done
    echo ""
    echo "Searching for any 'mma' binary in build tree..."
    find "${ACCSPMM_DIR}" -name "mma" -type f 2>/dev/null || echo "  None found"
    echo ""
    echo "Please check build output and update accspmm_utils.py with correct path"
    exit 1
fi

echo ""
echo "=========================================="
echo "AccSpMM installation completed successfully!"
echo "=========================================="
echo "Binary: ${BINARY_FOUND}"
echo ""
echo "Next steps:"
echo "  1. Test: python3 operators/accspmm_spmm.py <matrix.mtx> --n-cols 32"
echo "  2. Full pipeline: python3 scripts/test_pipeline.py <matrix.mtx>"
echo ""
