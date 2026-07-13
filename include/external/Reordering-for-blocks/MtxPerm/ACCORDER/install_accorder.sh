#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZENODO_URL="https://zenodo.org/api/records/14214504/files/zenodo-version.zip/content"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"

# Load Boost and numactl modules for headers/libraries
module purge 2>/dev/null || true
module load Boost/1.82.0-GCC-12.3.0
module load numactl/2.0.16-GCCcore-12.3.0

# Use system compiler (EasyBuild GCC/binutils may crash on some nodes)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# Remove EasyBuild binutils from PATH (its assembler crashes on some nodes)
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "binutils" | tr '\n' ':' | sed 's/:$//')

# 1. Download and extract order/ sources from Zenodo (skip if already present)
if [ ! -f "$SRC_DIR/my_order.hpp" ]; then
    echo "Downloading accOrder sources from Zenodo..."
    TMP_ZIP="${TMP_ZIP:-$(mktemp /tmp/accspmm_XXXXXX.zip)}"
    if [ ! -f "$TMP_ZIP" ] || [ ! -s "$TMP_ZIP" ]; then
        curl -sL "$ZENODO_URL" -o "$TMP_ZIP"
    fi
    mkdir -p "$SRC_DIR"
    unzip -jo "$TMP_ZIP" "zenodo-version/spmm_compute/order/*" -d "$SRC_DIR/"
    echo "Sources extracted to $SRC_DIR/"
fi

# 2. Build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
/usr/bin/cmake "$SCRIPT_DIR" && make -j"$(nproc)"
echo "Built: $BUILD_DIR/accorder_perm"
