#!/bin/bash
#
# Install script for SPARTA (SParse AcceleRation on Tensor Architecture)
# Clones the repository and builds the blocking/reordering tools
#
# Requirements:
#   - GCC
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SPARTA_DIR="${SCRIPT_DIR}/repo"
SPARTA_REPO="https://github.com/HicrestLaboratory/SPARTA.git"

echo "=========================================="
echo "SPARTA Installation Script"
echo "=========================================="
echo "Target directory: ${SPARTA_DIR}"
echo "Repository: ${SPARTA_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/3] Checking prerequisites..."

if ! command -v gcc &> /dev/null; then
    echo "ERROR: gcc not found in PATH"
    exit 1
fi
GCC_VERSION=$(gcc --version | head -1)
echo "✓ GCC found: $GCC_VERSION"

# Step 2: Clone repository
echo ""
echo "[2/3] Cloning SPARTA repository..."
if [ -d "${SPARTA_DIR}" ] && [ -f "${SPARTA_DIR}/makefile" ]; then
    echo "Directory already exists and looks valid: ${SPARTA_DIR}"
    echo "Skipping git update to avoid credential issues in non-interactive mode."
else
    if [ -d "${SPARTA_DIR}" ]; then
        echo "Directory exists but seems incomplete or invalid. Removing..."
        rm -rf "${SPARTA_DIR}"
    fi
    git clone "${SPARTA_REPO}" "${SPARTA_DIR}"
fi
echo "✓ Repository ready at ${SPARTA_DIR}"

# Step 3: Build
echo ""
echo "[3/3] Building SPARTA..."
cd "${SPARTA_DIR}"

# SPARTA uses a simple Makefile
# We want to build the 'serial' target to get the blocking tools without needing full CUDA setup if possible,
# but the user mentioned running on a cluster which likely has CUDA.
# The README says:
# '''make serial''' to compile without cuda
# or
# '''make all''' to compile also the cuda test

# We'll build the 'serial' target to get the blocking tools without needing CUDA setup.
echo "Building with 'make serial'..."
if make serial; then
    echo "✓ Build successful (Serial only)"
else
    echo "ERROR: Build failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "SPARTA installed successfully!"
echo "Blocking tool location: ${SPARTA_DIR}/programs/general/TEST_blocking_VBR"
echo "=========================================="
