#!/bin/bash
#
# Install script for BLEST (Blazingly Efficient BFS using Tensor Cores)
# Clones the repository, patches CMakeLists.txt, builds blest_driver
#
# Requirements:
#   - CUDA Toolkit (>= 13.0)
#   - g++ (>= 12.3)
#   - cmake (>= 3.18)
#   - libcurl
#   - OpenMP
#
# Usage:
#   bash MtxPerm/BLEST/install.sh                # Standard install
#   bash MtxPerm/BLEST/install.sh --clean        # Clean and reinstall
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BLEST_DIR="${SCRIPT_DIR}/blest"
BLEST_REPO="https://github.com/delbek/blest"

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
echo "BLEST Installation Script"
echo "=========================================="
echo "Target directory: ${BLEST_DIR}"
echo "Repository: ${BLEST_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/5] Checking prerequisites..."

module load CUDA/ 2>/dev/null || module load cuda 2>/dev/null || true
module load GCC/13.3.0 2>/dev/null || true
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "  CUDA found: $CUDA_VERSION"

if ! command -v g++ &> /dev/null; then
    echo "g++ not found, attempting to load GCC module..."
    module load GCC/13.3.0 2>/dev/null || echo "Warning: Could not load GCC module"
fi
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found in PATH"
    exit 1
fi
GCC_VERSION=$(g++ --version | head -1)
echo "  g++ found: $GCC_VERSION"

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found in PATH"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -1)
echo "  cmake found: $CMAKE_VERSION"

# Detect GPU architecture
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    # Try to detect via nvidia-smi compute capability
    RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    # Only use if it looks like a version number (e.g. "8.0", "9.0")
    if echo "$RAW" | grep -qE '^[0-9]+\.[0-9]+$'; then
        GPU_ARCH=$(echo "$RAW" | tr -d '.')
    fi
fi
# Fallback: try python + torch
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH=$(python3 -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
fi
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH="80"  # Default to A100
    echo "  Warning: Could not detect GPU, defaulting to sm_${GPU_ARCH}"
else
    echo "  Detected GPU: sm_${GPU_ARCH}"
fi

# Step 2: Clone repository
echo ""
echo "[2/5] Cloning repository..."
if [ "$CLEAN_BUILD" = true ] && [ -d "${BLEST_DIR}" ]; then
    echo "Removing existing directory for clean build..."
    rm -rf "${BLEST_DIR}"
fi

if [ -d "${BLEST_DIR}" ]; then
    echo "Directory ${BLEST_DIR} already exists. Pulling latest changes..."
    cd "${BLEST_DIR}"
    git pull || echo "Warning: git pull failed, continuing with existing code"
else
    git clone "${BLEST_REPO}" "${BLEST_DIR}"
fi

# Step 3: Copy custom driver into repo
echo ""
echo "[3/5] Copying custom driver..."
cp "${SCRIPT_DIR}/blest_driver.cu" "${BLEST_DIR}/blest_driver.cu"
echo "  Copied blest_driver.cu"

# Step 4: Patch CMakeLists.txt to add blest_driver target
echo ""
echo "[4/5] Patching CMakeLists.txt..."
cd "${BLEST_DIR}"

# Remove 'native' CUDA arch so our -DCMAKE_CUDA_ARCHITECTURES flag works on headless nodes
sed -i 's/set(CMAKE_CUDA_ARCHITECTURES native)/# set(CMAKE_CUDA_ARCHITECTURES native) # patched: use cmake flag instead/' CMakeLists.txt

# Make CURL optional (blest_driver doesn't need it)
if grep -q "CURL REQUIRED" CMakeLists.txt; then
    sed -i 's/find_package(CURL REQUIRED)/find_package(CURL QUIET)/' CMakeLists.txt
    # Wrap original blest target in if(CURL_FOUND)
    sed -i '/^add_executable(blest$/i if(CURL_FOUND)' CMakeLists.txt
    # Find the closing paren of target_compile_options for blest and add endif()
    # We add it after the first target_compile_options block
    python3 -c "
import re
with open('CMakeLists.txt') as f:
    content = f.read()
# Find first target_compile_options block end (for blest target)
# It ends with a closing paren on its own line
parts = content.split('target_compile_options(blest', 1)
if len(parts) == 2:
    # Find the closing ) of this block
    idx = parts[1].find('\n)')
    if idx >= 0:
        parts[1] = parts[1][:idx+2] + '\nendif()\n' + parts[1][idx+2:]
        content = 'target_compile_options(blest'.join(parts)
with open('CMakeLists.txt', 'w') as f:
    f.write(content)
"
    echo "  Made CURL optional in CMakeLists.txt"
fi

# Check if already patched
if grep -q "blest_driver" CMakeLists.txt; then
    echo "  CMakeLists.txt already patched with blest_driver"
else
    cat >> CMakeLists.txt << 'PATCH'

# Custom BFS driver for external reordering pipeline (no CURL needed)
add_executable(blest_driver
    blest_driver.cu
    # Gorder
    Gorder/Graph.cpp
    Gorder/Util.cpp
    Gorder/UnitHeap.cpp
    #
)

target_link_libraries(blest_driver
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_options(blest_driver PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -g -std=c++20>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info --std=c++20 -Xptxas=-v>
)
PATCH
    echo "  Patched CMakeLists.txt with blest_driver target"
fi

# Step 5: Build
echo ""
echo "[5/5] Building..."
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}"
make -j$(nproc) blest_driver

# Verify
echo ""
if [ -f "blest_driver" ]; then
    echo "Build successful!"
    echo "Binary: ${BLEST_DIR}/build/blest_driver"
else
    echo "ERROR: Build failed - blest_driver binary not found"
    exit 1
fi

echo ""
echo "Installation complete."
