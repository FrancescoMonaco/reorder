#!/bin/bash
# Install Rabbit Order library for SparseBase
# Rabbit Order: Cache-aware sparse matrix reordering
# Source: https://github.com/araij/rabbit_order
# Dependencies: Boost, libnuma

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RABBIT_DIR="$SCRIPT_DIR/rabbit_build"
INSTALL_DIR="$SCRIPT_DIR/rabbit_install"

echo "=========================================="
echo "Rabbit Order Installation"
echo "=========================================="
echo "Installation directory: $INSTALL_DIR"
echo ""

# Check for dependencies
echo "Checking dependencies..."
if ! ldconfig -p 2>/dev/null | grep -q libnuma; then
    echo "WARNING: libnuma not found. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install libnuma-dev"
    echo ""
fi

if ! ldconfig -p 2>/dev/null | grep -q libboost; then
    echo "WARNING: Boost not found. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install libboost-dev"
    echo ""
fi

# Clean previous installation if requested
if [[ "$1" == "--clean" ]]; then
    echo "Cleaning previous installation..."
    rm -rf "$RABBIT_DIR"
    rm -rf "$INSTALL_DIR"
    echo "Clean complete!"
    echo ""
fi

# Step 1: Clone Rabbit Order
if [ ! -d "$RABBIT_DIR" ]; then
    echo "Cloning Rabbit Order..."
    git clone https://github.com/araij/rabbit_order.git "$RABBIT_DIR"
else
    echo "Rabbit Order directory already exists, skipping clone..."
fi

cd "$RABBIT_DIR"

# Step 2: Install (Rabbit Order is header-only, just copy the header)
echo "Installing Rabbit Order (header-only library)..."
mkdir -p "$INSTALL_DIR/include"

# Copy the header file
cp rabbit_order.hpp "$INSTALL_DIR/include/"

# Optionally build the demo (for reference, not needed for SparseBase)
if [ -d "demo" ]; then
    echo "Building demo program (optional)..."
    cd demo
    make -j$(nproc 2>/dev/null || echo 4) 2>/dev/null || echo "Demo build failed (not critical)"
    cd ..
fi

echo ""
echo "=========================================="
echo "Rabbit Order Installation Complete!"
echo "=========================================="
echo "Installed to: $INSTALL_DIR"
echo ""
echo "Rabbit Order is a HEADER-ONLY library."
echo "To use with SparseBase, set these CMake variables:"
echo "  -DUSE_RABBIT_ORDER=ON"
echo "  -DRABBIT_ORDER_INC_DIR=$INSTALL_DIR/include"
echo ""
echo "Note: Ensure Boost and libnuma are loaded/installed:"
echo "  module load Boost"
echo "  module load numactl"
echo ""
