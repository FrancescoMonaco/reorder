#!/bin/bash
#
# Install script for mt-kahypar (Mt-KaHyPar) partitioning-based reordering.
#
# Clones Mt-KaHyPar, builds the shared library `mtkahypar` into a local install
# prefix (no root needed), then builds the `kahypar_perm` wrapper tool.
#
# Reference:
#   Gottesbüren et al., "Scalable High-Quality Hypergraph Partitioning",
#   ACM Trans. Algorithms 20(1), 2024.  https://github.com/kahypar/mt-kahypar
#
# Usage:
#   ./install.sh           # Clone/build/install mt-kahypar + build kahypar_perm
#   ./install.sh --clean   # Remove local artifacts and rebuild from scratch
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/mt-kahypar"
INSTALL_DIR="$SCRIPT_DIR/mt-kahypar-install"
BUILD_DIR="$SCRIPT_DIR/build"
MK_BUILD_DIR="$REPO_DIR/build"
REPO_URL="https://github.com/kahypar/mt-kahypar.git"
REPO_TAG="${MK_TAG:-master}"   # override with MK_TAG=<tag> to pin a version

CLEAN=false
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=true ;;
    esac
done

echo "==========================================="
echo "mt-kahypar Installation"
echo "==========================================="
echo "Source : ${REPO_DIR}"
echo "Install: ${INSTALL_DIR}"
echo "Build  : ${BUILD_DIR}"
echo ""

if [ "$CLEAN" = true ]; then
    echo "Removing local build artifacts..."
    rm -rf "$MK_BUILD_DIR" "$BUILD_DIR" "$INSTALL_DIR"
fi

# ---------------------------------------------------------------------------
# Dependencies: mt-kahypar needs a C++20 compiler and TBB. hwloc is OPTIONAL
# (used only for thread-pinning/NUMA) and is disabled below via
# -DKAHYPAR_DISABLE_HWLOC=On, so we do not need it. Boost is header-only usage.
# We provision a self-contained toolchain (compiler + TBB) via micromamba when
# available; otherwise the system toolchain is used.
# ---------------------------------------------------------------------------
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    ENV_NAME="KAHYPAR"
    if ! micromamba env list | grep -q "^${ENV_NAME} "; then
        echo "Creating micromamba environment '${ENV_NAME}'..."
        # NOTE: the conda-forge TBB packages are 'tbb' (runtime) and 'tbb-devel'
        # (headers + cmake config). There is no 'hwloc' package on conda-forge,
        # and we disable hwloc in the mt-kahypar build anyway.
        micromamba create -n "${ENV_NAME}" -c conda-forge \
            cmake make gxx_linux-64 tbb tbb-devel boost -y
    fi
    micromamba activate "${ENV_NAME}"
    echo "Activated: ${ENV_NAME}"
    # Help mt-kahypar's find_package(TBB) locate the env's TBB.
    if [ -n "${CONDA_PREFIX:-}" ]; then
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"
        export TBB_DIR="${CONDA_PREFIX}/lib/cmake/TBB"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    fi
fi

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# ---------------------------------------------------------------------------
# Step 1: Clone mt-kahypar
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Checking mt-kahypar repository..."
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning mt-kahypar (${REPO_TAG})..."
    git clone --depth 1 --branch "$REPO_TAG" "$REPO_URL" "$REPO_DIR"
else
    echo "Repository already cloned at $REPO_DIR"
fi

# ---------------------------------------------------------------------------
# Step 2: Build + install the mtkahypar shared library (local prefix)
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Building and installing libmtkahypar..."
mkdir -p "$MK_BUILD_DIR" && cd "$MK_BUILD_DIR"
# Only the library is needed (no CLI, no python, no tests).
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DKAHYPAR_PYTHON=OFF \
    -DKAHYPAR_ENABLE_TESTING=OFF \
    -DKAHYPAR_INSTALL_CLI=OFF \
    -DKAHYPAR_BUILD_DEBIAN_PACKAGE=OFF \
    -DKAHYPAR_DISABLE_HWLOC=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
make -j"$(nproc)" mtkahypar
cmake --install . --component MtKaHyPar_Lib

# ---------------------------------------------------------------------------
# Step 3: Build the kahypar_perm wrapper tool
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Building kahypar_perm..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake "$SCRIPT_DIR" -DMTKAHYPAR_INSTALL_DIR="$INSTALL_DIR"
make -j"$(nproc)"

echo ""
echo "==========================================="
echo "mt-kahypar installation complete!"
echo "==========================================="
echo ""
echo "Tool: ${BUILD_DIR}/kahypar_perm"
echo ""
echo "Usage:"
echo "  ${BUILD_DIR}/kahypar_perm <input.mtx> <output.perm> --nparts 128"
echo ""
echo "If the shared library cannot be found at runtime, source:"
echo "  source ${SCRIPT_DIR}/kahypar_preprocess.sh   # Bash"
echo "  source ${SCRIPT_DIR}/kahypar_preprocess.fish # Fish")
