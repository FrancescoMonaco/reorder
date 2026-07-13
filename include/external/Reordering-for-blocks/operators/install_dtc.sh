#!/bin/bash
#
# Install script for DTC-SpMM (ASPLOS'24)
# Based on: https://github.com/HPMLL/DTC-SpMM_ASPLOS24
#
# Requirements:
#   - CUDA Toolkit >= 11.8 (CUDA 12.1 recommended)
#   - Python 3.9–3.11 (3.12+ has distutils issues)
#   - PyTorch with CUDA support
#   - cmake
#
# Usage:
#   ./install_dtc.sh                          # Standard install
#   ./install_dtc.sh --clean                  # Clean and reinstall
#   TORCH_CUDA_ARCH_LIST="8.9" ./install_dtc.sh  # Specific arch (e.g., RTX 4090)
#

set -e  # Exit on first error

# Initialize modules if available (needed for many clusters)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DTC_DIR="${SCRIPT_DIR}/DTC-SpMM"
DTC_REPO="https://github.com/HPMLL/DTC-SpMM_ASPLOS24.git"

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
echo "DTC-SpMM Installation Script (ASPLOS'24)"
echo "=========================================="
echo "Target directory: ${DTC_DIR}"
echo "Repository: ${DTC_REPO}"
echo ""

# ============================================
# Step 1: Check prerequisites
# ============================================
echo "[1/5] Checking prerequisites..."

# Determine the best python executable
PYTHON_EXE=""
for py in python3 python; do
    if command -v $py &> /dev/null && $py -c "import torch" &> /dev/null; then
        PYTHON_EXE=$(command -v $py)
        break
    fi
done
if [ -z "$PYTHON_EXE" ]; then
    echo "ERROR: PyTorch not found in your Python environment."
    echo "Please install PyTorch (Python 3.9-3.11 recommended):"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi
PYTHON_VERSION=$($PYTHON_EXE --version)
echo "✓ Using Python: $PYTHON_EXE ($PYTHON_VERSION)"

# Check Python version (warn if 3.12+)
PY_MAJOR=$($PYTHON_EXE -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_EXE -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 12 ]; then
    echo "⚠ WARNING: Python 3.12+ detected. DTC-SpMM may fail due to distutils removal."
    echo "  Consider using Python 3.9-3.11 for best compatibility."
fi

# Ensure we have a modern GCC for GLIBCXX compatibility (required by cmake/glog)
# and for general compilation consistency.
echo "Loading GCC and CUDA modules..."
# If on an EasyBuild cluster, GCCcore/binutils are often needed for consistency
module load GCCcore/13.2.0 2>/dev/null || module load gcc/13.2.0 2>/dev/null || module load GCCcore/12.3.0 2>/dev/null || module load gcc/12.1.0 2>/dev/null || true

# Try to load CUDA module on clusters if nvcc is missing
if ! command -v nvcc &> /dev/null; then
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || module load cuda/12.1 2>/dev/null || true
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please load CUDA: module load cuda"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "✓ CUDA (nvcc) found: $CUDA_VERSION"

# Set CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    NVCC_PATH=$(command -v nvcc)
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
fi
echo "✓ CUDA_HOME: $CUDA_HOME"

# Check cmake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Install with: conda install cmake"
    exit 1
fi
echo "✓ cmake found"

# Check PyTorch
TORCH_VERSION=$($PYTHON_EXE -c "import torch; print(torch.__version__)")
TORCH_CUDA=$($PYTHON_EXE -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
echo "✓ PyTorch: $TORCH_VERSION (CUDA $TORCH_CUDA)"

# Detect or set GPU architecture
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    GPU_ARCH=$($PYTHON_EXE -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
    if [ -n "$GPU_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
        echo "✓ Detected GPU: sm_$(echo $GPU_ARCH | tr -d '.')"
    else
        # Default to RTX 4090 / A100
        export TORCH_CUDA_ARCH_LIST="8.0 8.9"
        echo "⚠ Could not detect GPU. Using defaults: $TORCH_CUDA_ARCH_LIST"
    fi
else
    echo "✓ Using TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
fi

# Limit parallel jobs (avoid OOM on login nodes)
export MAX_JOBS=${MAX_JOBS:-4}

# ============================================
# Step 2: Clone repository
# ============================================
echo ""
echo "[2/5] Cloning DTC-SpMM repository..."

if [ "$CLEAN_BUILD" = true ] && [ -d "${DTC_DIR}" ]; then
    echo "Removing existing directory for clean build..."
    rm -rf "${DTC_DIR}"
fi

if [ -d "${DTC_DIR}" ]; then
    echo "Directory exists. Updating..."
    cd "${DTC_DIR}"
    git pull || echo "Warning: git pull failed"
    git submodule update --init --recursive || true
else
    git clone --recursive "${DTC_REPO}" "${DTC_DIR}"
fi

cd "${DTC_DIR}"
export DTC_HOME="${DTC_DIR}"
echo "✓ DTC_HOME=$DTC_HOME"

# ============================================
# Step 3: Build Sputnik (dependency)
# ============================================
echo ""
echo "[3/5] Building Sputnik dependency..."

cd "${DTC_HOME}/third_party"

# Build glog first (Sputnik dependency)
if [ -d "glog" ]; then
    echo "Building glog..."
    cd glog
    if [ "$CLEAN_BUILD" = true ]; then rm -rf build; fi
    mkdir -p build && cd build
    cmake -DCMAKE_INSTALL_PREFIX="${DTC_HOME}/third_party/glog/build" \
          -DBUILD_SHARED_LIBS=OFF \
          -DWITH_GTEST=OFF \
          -DWITH_GFLAGS=OFF \
          -DCMAKE_DISABLE_FIND_PACKAGE_gflags=ON \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          ..
    make -j${MAX_JOBS}
    make install
    cd ../..
    
    export GLOG_PATH="${DTC_HOME}/third_party/glog"
    export LD_LIBRARY_PATH="${GLOG_PATH}/build/lib:${GLOG_PATH}/build/lib64:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${GLOG_PATH}/build/lib:${GLOG_PATH}/build/lib64:${LIBRARY_PATH}"
    export CPLUS_INCLUDE_PATH="${GLOG_PATH}/build/include:${CPLUS_INCLUDE_PATH}"
    echo "✓ glog built"
fi

# Build Sputnik
if [ -d "sputnik" ]; then
    echo "Building Sputnik..."
    cd sputnik
    if [ "$CLEAN_BUILD" = true ]; then rm -rf build; fi
    mkdir -p build && cd build
    
    # Find glog library
    GLOG_LIB=$(find "${DTC_HOME}/third_party/glog/build" -name "libglog.so" -o -name "libglog.a" 2>/dev/null | head -1)
    if [ -z "$GLOG_LIB" ]; then
        GLOG_LIB="${DTC_HOME}/third_party/glog/build/lib/libglog.so"
    fi
    
    # Convert TORCH_CUDA_ARCH_LIST to cmake format (e.g., "8.0 8.9" -> "80;89")
    CUDA_ARCHS=$(echo "$TORCH_CUDA_ARCH_LIST" | tr ' ' ';' | tr -d '.')
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TEST=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DCUDA_ARCHS="${CUDA_ARCHS}" \
        -DGLOG_INCLUDE_DIR="${DTC_HOME}/third_party/glog/build/include" \
        -DGLOG_LIBRARY="${GLOG_LIB}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -j${MAX_JOBS}
    cd ../..
    
    export SPUTNIK_PATH="${DTC_HOME}/third_party/sputnik"
    export LD_LIBRARY_PATH="${SPUTNIK_PATH}/build/sputnik:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${SPUTNIK_PATH}/build/sputnik:${LIBRARY_PATH}"
    echo "✓ Sputnik built"
fi

# ============================================
# Step 4: Build DTC-SpMM
# ============================================
echo ""
echo "[4/5] Building DTC-SpMM..."

cd "${DTC_HOME}/DTC-SpMM"

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf build dist *.egg-info
fi

# The setup.py expects these environment variables
export SPUTNIK_PATH="${DTC_HOME}/third_party/sputnik"
export GLOG_PATH="${DTC_HOME}/third_party/glog"

echo "Building with:"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  SPUTNIK_PATH=$SPUTNIK_PATH"
echo "  GLOG_PATH=$GLOG_PATH"

$PYTHON_EXE setup.py install

# ============================================
# Step 5: Verify installation
# ============================================
echo ""
echo "[5/5] Verifying installation..."

cd "${SCRIPT_DIR}"  # Go back to avoid import issues

# Get PyTorch lib path for runtime linking
TORCH_LIB=$($PYTHON_EXE -c "import torch; print(torch.__path__[0] + '/lib')")

# Set library paths for verification
export LD_LIBRARY_PATH="${TORCH_LIB}:${GLOG_PATH}/build/lib64:${GLOG_PATH}/build/lib:${SPUTNIK_PATH}/build/sputnik:${LD_LIBRARY_PATH}"

if $PYTHON_EXE -c "import DTCSpMM; print('✓ DTCSpMM module loaded successfully')"; then
    echo ""
    echo "=========================================="
    echo "DTC-SpMM installation complete!"
    echo "=========================================="
    echo ""
    echo "Before running, set library paths:"
    echo "  export LD_LIBRARY_PATH=\"${TORCH_LIB}:${GLOG_PATH}/build/lib64:${SPUTNIK_PATH}/build/sputnik:\$LD_LIBRARY_PATH\""
    echo ""
    echo "Usage in Python:"
    echo "  import DTCSpMM"
    echo "  DTCSpMM.preprocess_gpu(...)"
    echo "  DTCSpMM.run_DTCSpMM(...)"
    echo ""
else
    echo "ERROR: Could not import DTCSpMM module"
    echo "Check the build output above for errors."
    exit 1
fi
