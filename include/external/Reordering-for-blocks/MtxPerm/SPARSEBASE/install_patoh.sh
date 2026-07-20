#!/bin/bash
# Install PaToH library for SparseBase
# PaToH: Partitioning Tool for Hypergraphs
# Source: https://faculty.cc.gatech.edu/~umit/software.html

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATOH_DIR="$SCRIPT_DIR/patoh_build"
INSTALL_DIR="$SCRIPT_DIR/patoh_install"

echo "=========================================="
echo "PaToH Installation"
echo "=========================================="
echo "Installation directory: $INSTALL_DIR"
echo ""

# Clean previous installation if requested
if [[ "$1" == "--clean" ]]; then
    echo "Cleaning previous installation..."
    rm -rf "$PATOH_DIR"
    rm -rf "$INSTALL_DIR"
    echo "Clean complete!"
    echo ""
fi

# Step 1: Download PaToH
if [ ! -d "$PATOH_DIR" ]; then
    echo "Downloading PaToH..."
    mkdir -p "$PATOH_DIR"
    cd "$PATOH_DIR"
    
    # PaToH is freely available for academic use
    # Download the Linux 64-bit version (latest v3.3)
    wget https://faculty.cc.gatech.edu/~umit/PaToH/patoh-Linux-x86_64.tar.gz || {
        echo "ERROR: Failed to download PaToH"
        echo "Please manually download from:"
        echo "  https://faculty.cc.gatech.edu/~umit/software.html"
        echo "And place patoh-Linux-x86_64.tar.gz in: $PATOH_DIR"
        exit 1
    }
    
    tar -xzf patoh-Linux-x86_64.tar.gz
else
    echo "PaToH directory already exists, skipping download..."
fi

# Step 2: Install headers and libraries
echo "Installing PaToH..."
mkdir -p "$INSTALL_DIR/lib"
mkdir -p "$INSTALL_DIR/include"

# PaToH tarball extracts to build/Linux-x86_64/ directory
PATOH_SRC="$PATOH_DIR/build/Linux-x86_64"

if [ ! -d "$PATOH_SRC" ]; then
    echo "ERROR: PaToH files not found in expected location: $PATOH_SRC"
    echo "Directory contents:"
    find "$PATOH_DIR" -type f -name "*.a" -o -name "*.h"
    exit 1
fi

# Copy header and library files
echo "Copying from: $PATOH_SRC"
cp "$PATOH_SRC"/*.h "$INSTALL_DIR/include/" || {
    echo "ERROR: Failed to copy header files"
    exit 1
}
cp "$PATOH_SRC"/*.a "$INSTALL_DIR/lib/" || {
    echo "ERROR: Failed to copy library files"
    exit 1
}

echo ""
echo "=========================================="
echo "PaToH Installation Complete!"
echo "=========================================="
echo "Installed to: $INSTALL_DIR"
echo ""
echo "To use with SparseBase, set these CMake variables:"
echo "  -DUSE_PATOH=ON"
echo "  -DPATOH_LIB_DIR=$INSTALL_DIR/lib"
echo "  -DPATOH_INC_DIR=$INSTALL_DIR/include"
echo ""
