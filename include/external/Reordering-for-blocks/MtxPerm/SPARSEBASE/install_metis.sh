#!/bin/bash
# Install METIS library for SparseBase
# METIS: Graph partitioning and nested dissection ordering
# Source: https://github.com/KarypisLab/METIS

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METIS_DIR="$SCRIPT_DIR/metis_build"
INSTALL_DIR="$SCRIPT_DIR/metis_install"
METIS_VERSION="5.1.0"

echo "=========================================="
echo "METIS Installation"
echo "=========================================="
echo "Installation directory: $INSTALL_DIR"
echo "METIS version: $METIS_VERSION"
echo ""

# Clean previous installation if requested
if [[ "$1" == "--clean" ]]; then
    echo "Cleaning previous installation..."
    rm -rf "$METIS_DIR"
    rm -rf "$INSTALL_DIR"
    echo "Clean complete!"
    echo ""
fi

# Step 1: Download METIS
if [ ! -d "$METIS_DIR" ]; then
    echo "Downloading METIS $METIS_VERSION..."
    mkdir -p "$METIS_DIR"
    cd "$METIS_DIR"
    
    # Download from official source - use 5.1.0 (latest stable)
    echo "Downloading from official source..."
    wget https://karypis.github.io/glaros/files/sw/metis/metis-${METIS_VERSION}.tar.gz
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download METIS"
        echo "Please manually download from:"
        echo "  https://karypis.github.io/glaros/files/sw/metis/metis-${METIS_VERSION}.tar.gz"
        echo "and place it in: $METIS_DIR"
        exit 1
    fi
    
    tar -xzf metis-${METIS_VERSION}.tar.gz
    cd metis-${METIS_VERSION}
else
    echo "METIS directory already exists, skipping download..."
    cd "$METIS_DIR/metis-${METIS_VERSION}"
fi

# Step 2: Configure METIS (this builds GKlib automatically)
echo "Configuring METIS..."
make config prefix="$INSTALL_DIR"

# Step 3: Build
echo "Building METIS..."
# The build directory is created as build/Linux-x86_64 (or similar)
BUILD_SUBDIR=$(find build -maxdepth 1 -type d -name "*-*" | head -n 1)
if [ -z "$BUILD_SUBDIR" ]; then
    BUILD_SUBDIR="build"
fi
cd "$BUILD_SUBDIR"
make -j$(nproc 2>/dev/null || echo 4)

# Step 4: Install
echo "Installing METIS to $INSTALL_DIR..."
make install

echo ""
echo "=========================================="
echo "METIS Installation Complete!"
echo "=========================================="
echo "Installed to: $INSTALL_DIR"
echo ""
echo "To use with SparseBase, set these CMake variables:"
echo "  -DUSE_METIS=ON"
echo "  -DMETIS_LIB_DIR=$INSTALL_DIR/lib"
echo "  -DMETIS_INC_DIR=$INSTALL_DIR/include"
echo ""
