#!/bin/bash
#
# Install script for SMaT (Sparse Matrix Tensor Core-accelerated library)
# Clones the repository and builds the CUDA SpMM kernel
#
# Requirements:
#   - CUDA Toolkit 12.0+
#   - nvcc compiler in PATH
#   - GCC 12.3.0+
#   - libgflags-dev (Linux): sudo apt-get install libgflags-dev
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SMAT_DIR="${SCRIPT_DIR}/smat"
SMAT_REPO="https://github.com/spcl/smat.git"
SMAT_VERSION="main"  # Can be changed to specific tag/branch

echo "=========================================="
echo "SMaT Installation Script"
echo "=========================================="
echo "Target directory: ${SMAT_DIR}"
echo "Repository: ${SMAT_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/5] Checking prerequisites..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please install CUDA Toolkit 12.0+ and ensure nvcc is in PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $NF}')
echo "✓ CUDA found: $CUDA_VERSION"

if ! command -v gcc &> /dev/null; then
    echo "ERROR: gcc not found in PATH"
    exit 1
fi
GCC_VERSION=$(gcc --version | head -1)
echo "✓ GCC found: $GCC_VERSION"

# Check for gflags (optional - will try to install if missing)
if ! command -v pkg-config &> /dev/null || ! pkg-config --exists gflags; then
    echo "⚠ gflags not found - will attempt to install or use compile.sh instead"
fi

# Step 2: Clone repository
echo ""
echo "[2/5] Cloning SMaT repository..."
if [ -d "${SMAT_DIR}" ]; then
    echo "Directory already exists: ${SMAT_DIR}"
    read -p "Update existing installation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "${SMAT_DIR}"
        git fetch origin
        git checkout "${SMAT_VERSION}"
        git pull origin "${SMAT_VERSION}"
    fi
else
    git clone --depth 1 -b "${SMAT_VERSION}" "${SMAT_REPO}" "${SMAT_DIR}"
fi
echo "✓ Repository ready at ${SMAT_DIR}"

# Step 3: Try to build/install gflags if needed
echo ""
echo "[3/5] Checking gflags dependency..."
GFLAGS_FOUND=0

# Check if gflags is already available
if pkg-config --exists gflags 2>/dev/null; then
    echo "✓ gflags found via pkg-config"
    GFLAGS_FOUND=1
elif [ -f "/usr/include/gflags/gflags.h" ] || [ -f "/usr/local/include/gflags/gflags.h" ]; then
    echo "✓ gflags headers found in system"
    GFLAGS_FOUND=1
fi

# If not found, suggest installation
if [ $GFLAGS_FOUND -eq 0 ]; then
    echo "⚠ gflags not found. Installing locally to ${SMAT_DIR}/gflags_install"
    echo ""
    
    # Build gflags locally
    GFLAGS_INSTALL_DIR="${SMAT_DIR}/gflags_install"
    cd /tmp
    
    if [ ! -d "gflags" ]; then
        echo "Cloning gflags..."
        git clone --depth 1 https://github.com/gflags/gflags.git
    fi
    
    cd gflags
    mkdir -p build
    cd build
    
    echo "Building gflags..."
    cmake .. -DCMAKE_INSTALL_PREFIX="${GFLAGS_INSTALL_DIR}" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON
    make -j$(nproc)
    make install
    
    echo "✓ gflags installed to ${GFLAGS_INSTALL_DIR}"
    
    # Set environment for subsequent CMake
    export CMAKE_PREFIX_PATH="${GFLAGS_INSTALL_DIR}:${CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="${GFLAGS_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
    export PKG_CONFIG_PATH="${GFLAGS_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"
    
    cd "${SMAT_DIR}"
    GFLAGS_FOUND=1
fi

# Step 4: Build CUDA binaries
echo ""
echo "[4/5] Building SMaT CUDA kernels..."
cd "${SMAT_DIR}"

# Try the provided compile script first
if [ -f "src/cuda_hgemm/compile.sh" ]; then
    echo "Using provided compile.sh script..."
    cd src/cuda_hgemm
    chmod +x compile.sh build.sh 2>/dev/null || true
    bash compile.sh
    cd "${SMAT_DIR}"
elif [ -f "CMakeLists.txt" ]; then
    echo "Using CMake build system..."
    mkdir -p build
    cd build
    
    # Add gflags path if we installed it locally
    GFLAGS_INSTALL_DIR="${SMAT_DIR}/gflags_install"
    if [ -d "${GFLAGS_INSTALL_DIR}" ]; then
        export CMAKE_PREFIX_PATH="${GFLAGS_INSTALL_DIR}:${CMAKE_PREFIX_PATH}"
    fi
    
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # A100: 80, H100: 90
    make -j$(nproc)
    cd "${SMAT_DIR}"
else
    echo "ERROR: Could not find compile script or CMakeLists.txt"
    exit 1
fi

echo "✓ Build completed"

# Step 5: Verify binary
echo ""
echo "[5/5] Verifying installation..."
BINARY_PATHS=(
    "${SMAT_DIR}/src/cuda_hgemm/output/bin/hgemm"
    "${SMAT_DIR}/build/bin/hgemm"
)

BINARY_FOUND=0
for BINARY_PATH in "${BINARY_PATHS[@]}"; do
    if [ -f "${BINARY_PATH}" ]; then
        echo "✓ Binary found: ${BINARY_PATH}"
        BINARY_FOUND=1
        break
    fi
done

if [ $BINARY_FOUND -eq 0 ]; then
    echo "WARNING: Could not locate SMaT binary in expected locations"
    echo "Expected one of:"
    for BINARY_PATH in "${BINARY_PATHS[@]}"; do
        echo "  - ${BINARY_PATH}"
    done
    echo ""
    echo "Please check build output and update smat_utils.py with correct path"
fi

echo ""
echo "=========================================="
echo "✓ SMaT installation completed successfully!"
echo "=========================================="
echo "Build directory: ${SMAT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Verify CUDA architecture matches your GPU (edit CMakeLists if needed)"
echo "  2. Test with: python3 operators/smat_spmm.py <matrix.mtx>"
echo "  3. Add SMAT jobs to YAML configuration files"
echo ""
