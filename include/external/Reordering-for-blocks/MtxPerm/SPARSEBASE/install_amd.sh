#!/bin/bash
# Install AMD (from SuiteSparse) for SparseBase
# AMD: Approximate Minimum Degree ordering
# Source: https://github.com/DrTimothyAldenDavis/SuiteSparse

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITESPARSE_DIR="$SCRIPT_DIR/suitesparse_build"
INSTALL_DIR="$SCRIPT_DIR/amd_install"
SUITESPARSE_VERSION="7.8.3"

echo "=========================================="
echo "AMD (SuiteSparse) Installation"
echo "=========================================="
echo "Installation directory: $INSTALL_DIR"
echo "SuiteSparse version: $SUITESPARSE_VERSION"
echo ""

# Clean previous installation if requested
if [[ "$1" == "--clean" ]]; then
    echo "Cleaning previous installation..."
    rm -rf "$SUITESPARSE_DIR"
    rm -rf "$INSTALL_DIR"
    echo "Clean complete!"
    echo ""
fi

# Step 1: Clone SuiteSparse
if [ ! -d "$SUITESPARSE_DIR" ]; then
    echo "Cloning SuiteSparse..."
    git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git "$SUITESPARSE_DIR"
    cd "$SUITESPARSE_DIR"
    git checkout v${SUITESPARSE_VERSION}
else
    echo "SuiteSparse directory already exists, skipping clone..."
    cd "$SUITESPARSE_DIR"
fi

# Step 2: Build only AMD and its dependencies (SuiteSparse_config)
# Use direct Makefile build to avoid BLAS dependency
echo "Building AMD (using Makefile, no BLAS required)..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/lib"
mkdir -p "$INSTALL_DIR/include"

# Build SuiteSparse_config (dependency) - standalone without CMake
echo "Building SuiteSparse_config..."
cd SuiteSparse_config
# Configure CMake with proper install prefix to avoid permission issues
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release -DNSTATIC=ON
cmake --build . --config Release -j8
cmake --install .
cd ../..

# Build AMD - standalone without CMake or BLAS
echo "Building AMD..."
cd AMD
# Configure CMake with proper install prefix to avoid permission issues
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release -DNSTATIC=ON
cmake --build . --config Release -j8
cmake --install .

echo ""
echo "Organizing include files for SparseBase compatibility..."
cd "$INSTALL_DIR"
# SparseBase expects AMD headers directly in include/, not include/suitesparse
if [ -d "include/suitesparse" ]; then
    echo "Copying headers from include/suitesparse/ to include/"
    cp -v include/suitesparse/*.h include/ 2>/dev/null || true
    echo "Headers copied"
fi

echo ""
echo "=========================================="
echo "AMD Installation Complete!"
echo "=========================================="
echo "Installed to: $INSTALL_DIR"
echo ""
echo "To use with SparseBase, set these CMake variables:"
echo "  -DUSE_AMD_ORDER=ON"
echo "  -DAMD_LIB_DIR=$INSTALL_DIR/lib"
echo "  -DAMD_INC_DIR=$INSTALL_DIR/include"
echo ""
