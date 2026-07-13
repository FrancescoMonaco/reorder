#!/bin/bash
# Reproducible setup script for SparseBase library
# This script clones, builds, and installs SparseBase locally within the MtxPerm directory

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARSEBASE_DIR="$SCRIPT_DIR/sparsebase"
BUILD_DIR="$SPARSEBASE_DIR/build"
INSTALL_DIR="$SCRIPT_DIR/sparsebase_install"

# SparseBase version to use
SPARSEBASE_VERSION="main"

# Parse arguments
CLEAN=false
BUILD_ONLY=false
ENABLE_AMD=false
ENABLE_METIS=false
ENABLE_RABBIT=false
ENABLE_PATOH=false
ENABLE_ALL=false

for arg in "$@"; do
    if [[ "$arg" == "--clean" ]]; then
        CLEAN=true
    elif [[ "$arg" == "--build-only" ]]; then
        BUILD_ONLY=true
    elif [[ "$arg" == "--with-all" ]]; then
        ENABLE_ALL=true
        ENABLE_AMD=true
        ENABLE_METIS=true
        ENABLE_RABBIT=true
        ENABLE_PATOH=true
    elif [[ "$arg" == "--with-amd" ]]; then
        ENABLE_AMD=true
    elif [[ "$arg" == "--with-metis" ]]; then
        ENABLE_METIS=true
    elif [[ "$arg" == "--with-rabbit" ]]; then
        ENABLE_RABBIT=true
    elif [[ "$arg" == "--with-patoh" ]]; then
        ENABLE_PATOH=true
    fi
done

if $CLEAN; then
    echo "=========================================="
    echo "Cleaning previous installation"
    echo "=========================================="
    echo "Removing: $SPARSEBASE_DIR"
    echo "Removing: $INSTALL_DIR"
    echo "Removing: $SCRIPT_DIR/build"
    rm -rf "$SPARSEBASE_DIR"
    rm -rf "$INSTALL_DIR"
    rm -rf "$SCRIPT_DIR/build"
    echo "Clean complete!"
    echo ""
fi

echo "=========================================="
echo "SparseBase Setup for MtxPerm"
echo "=========================================="
echo "Installation directory: $INSTALL_DIR"
echo "SparseBase version: $SPARSEBASE_VERSION"
if $BUILD_ONLY; then
    echo "Mode: Build tools only (skipping SparseBase installation)"
fi
if $ENABLE_AMD; then
    echo "AMD reordering: ENABLED"
fi
if $ENABLE_METIS; then
    echo "METIS reordering: ENABLED"
fi
if $ENABLE_RABBIT; then
    echo "Rabbit Order reordering: ENABLED"
fi
if $ENABLE_PATOH; then
    echo "PaToH partitioning: ENABLED"
fi
echo ""

if ! $BUILD_ONLY; then
    # Step 1: Clone SparseBase if not already present
    if [ ! -d "$SPARSEBASE_DIR" ]; then
        echo "Cloning SparseBase..."
        git clone https://github.com/sparcityeu/SparseBase.git "$SPARSEBASE_DIR"
        cd "$SPARSEBASE_DIR"
        git checkout "$SPARSEBASE_VERSION"
    else
        echo "SparseBase directory already exists, skipping clone..."
    fi

    # Step 2: Create build directory
    echo "Setting up build directory..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Step 3: Configure with CMake
    echo "Configuring SparseBase with CMake..."
    
    # Clean environment to avoid EasyBuild module conflicts
    # The issue: EasyBuild's GCCcore module pollutes the environment with incompatible binutils
    # Solution: Unload problematic modules and use system compiler with clean environment
    
    echo "Cleaning environment from EasyBuild module pollution..."
    
    # Save original PATH
    ORIGINAL_PATH="$PATH"
    
    # Remove EasyBuild paths that might contain incompatible tools
    if [ -n "$EBROOTGCCCORE" ]; then
        echo "Detected GCCcore module: $EBROOTGCCCORE"
        # Remove GCCcore paths from environment
        PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "GCCcore" | tr '\n' ':' | sed 's/:$//')
        unset EBROOTGCCCORE
        unset EBVERSIONGCCCORE
    fi
    
    # Clean up other potentially conflicting EasyBuild variables
    unset EBROOTGCC
    unset EBVERSIONGCC
    
    # Use system compiler with clean PATH
    # Check if CC/CXX are already set (e.g. by modules), otherwise look for gcc
    if [ -n "$CC" ] && [ -n "$CXX" ]; then
        echo "Using environment compilers:"
        echo "  CC=$CC"
        echo "  CXX=$CXX"
    elif [ -f "/usr/bin/gcc" ]; then
        export CC="/usr/bin/gcc"
        export CXX="/usr/bin/g++"
        export PATH="/usr/bin:/bin:$PATH"
        echo "Using system compilers with clean environment:"
        echo "  CC=$CC"
        echo "  CXX=$CXX"
        echo "  PATH prioritizes /usr/bin"
    else
        echo "ERROR: System compiler not found at /usr/bin/gcc"
        echo "Please ensure GCC is installed on the system or load a proper GCC module:"
        echo "  module purge"
        echo "  module load GCC/12.3.0"
        exit 1
    fi
    
    # Build CMake command with optional dependencies
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF"
    
    # AMD configuration
    if $ENABLE_AMD; then
        AMD_INSTALL="$SCRIPT_DIR/amd_install"
        if [ -d "$AMD_INSTALL" ]; then
            CMAKE_ARGS="$CMAKE_ARGS \
                -DUSE_AMD_ORDER=ON \
                -DAMD_LIB_DIR=$AMD_INSTALL/lib \
                -DAMD_INC_DIR=$AMD_INSTALL/include"
        else
            echo "WARNING: AMD installation not found at $AMD_INSTALL"
            echo "Run: bash install_amd.sh"
            CMAKE_ARGS="$CMAKE_ARGS -DUSE_AMD_ORDER=OFF"
        fi
    else
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_AMD_ORDER=OFF"
    fi
    
    # METIS configuration
    if $ENABLE_METIS; then
        METIS_INSTALL="$SCRIPT_DIR/metis_install"
        if [ -d "$METIS_INSTALL" ]; then
            CMAKE_ARGS="$CMAKE_ARGS \
                -DUSE_METIS=ON \
                -DMETIS_LIB_DIR=$METIS_INSTALL/lib \
                -DMETIS_INC_DIR=$METIS_INSTALL/include"
        else
            echo "WARNING: METIS installation not found at $METIS_INSTALL"
            echo "Run: bash install_metis.sh"
            CMAKE_ARGS="$CMAKE_ARGS -DUSE_METIS=OFF"
        fi
    else
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_METIS=OFF"
    fi
    
    # Rabbit Order configuration
    if $ENABLE_RABBIT; then
        RABBIT_INSTALL="$SCRIPT_DIR/rabbit_install"
        if [ -d "$RABBIT_INSTALL" ]; then
            CMAKE_ARGS="$CMAKE_ARGS \
                -DUSE_RABBIT_ORDER=ON \
                -DRABBIT_ORDER_LIB_DIR=$RABBIT_INSTALL/lib \
                -DRABBIT_ORDER_INC_DIR=$RABBIT_INSTALL/include"
        else
            echo "WARNING: Rabbit Order installation not found at $RABBIT_INSTALL"
            echo "Run: bash install_rabbit.sh"
            CMAKE_ARGS="$CMAKE_ARGS -DUSE_RABBIT_ORDER=OFF"
        fi
    else
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_RABBIT_ORDER=OFF"
    fi
    
    # PaToH configuration
    if $ENABLE_PATOH; then
        PATOH_INSTALL="$SCRIPT_DIR/patoh_install"
        if [ -d "$PATOH_INSTALL" ]; then
            CMAKE_ARGS="$CMAKE_ARGS \
                -DUSE_PATOH=ON \
                -DPATOH_LIB_DIR=$PATOH_INSTALL/lib \
                -DPATOH_INC_DIR=$PATOH_INSTALL/include"
        else
            echo "WARNING: PaToH installation not found at $PATOH_INSTALL"
            echo "Run: bash install_patoh.sh"
            CMAKE_ARGS="$CMAKE_ARGS -DUSE_PATOH=OFF"
        fi
    else
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_PATOH=OFF"
    fi
    
    cmake .. $CMAKE_ARGS

    # Step 4: Build
    echo "Building SparseBase..."
    cmake --build . --parallel $(nproc 2>/dev/null || echo 4)

    # Step 5: Install
    echo "Installing SparseBase to $INSTALL_DIR..."
    cmake --install .

    echo ""
fi
echo "=========================================="
echo "Building Permutation Tools"
echo "=========================================="

# Step 6: Build tools
TOOLS_BUILD_DIR="$SCRIPT_DIR/build"
echo "Creating build directory..."
rm -rf "$TOOLS_BUILD_DIR"
mkdir -p "$TOOLS_BUILD_DIR"
cd "$TOOLS_BUILD_DIR"

echo "Configuring..."
cmake .. -DHAVE_SLASHBURN_REORDER=ON

echo "Building..."
cmake --build . --parallel $(nproc 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
if ! $BUILD_ONLY; then
    echo "SparseBase installed to: $INSTALL_DIR"
fi
echo "Tools built in: $TOOLS_BUILD_DIR"
echo ""
echo "Usage:"
echo "  ./build/<tool_name> matrix.mtx output.perm"
echo ""
echo "Flags:"
echo "  --clean         Remove all previous builds before starting"
echo "  --build-only    Only rebuild tools (skip SparseBase installation)"
echo "  --with-all      Enable all available reordering methods (AMD, METIS, Rabbit, PaToH)"
echo "  --with-amd      Enable AMD reordering (requires install_amd.sh)"
echo "  --with-metis    Enable METIS reordering (requires install_metis.sh)"
echo "  --with-rabbit   Enable Rabbit Order reordering (requires install_rabbit.sh)"
echo "  --with-patoh    Enable PaToH partitioning (requires install_patoh.sh)"
echo ""
echo "To install dependencies:"
echo "  bash install_amd.sh"
echo "  bash install_metis.sh"
echo "  bash install_rabbit.sh"
echo "  bash install_patoh.sh"
echo ""
