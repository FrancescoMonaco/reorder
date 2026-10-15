#!/bin/bash
# Preprocess script for mt-kahypar reordering.
#
# Makes the mt-kahypar shared library and its TBB dependency discoverable at
# runtime. install.sh installs libmtkahypar into a local prefix (under lib64/
# on 64-bit Linux, lib/ elsewhere) and provisions TBB inside the KAHYPAR
# micromamba environment. This adds all of those to LD_LIBRARY_PATH so the
# kahypar_perm binary runs whether or not the env is already activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/mt-kahypar-install"

# mt-kahypar's own shared library (GNUInstallDirs -> lib64 on 64-bit Linux).
for d in "$INSTALL_DIR/lib64" "$INSTALL_DIR/lib"; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

# TBB lives in the KAHYPAR micromamba env. Prefer the active env if present,
# otherwise fall back to the default micromamba env location used by install.sh.
ENV_LIB="${CONDA_PREFIX:+$CONDA_PREFIX/lib}"
if [ -z "$ENV_LIB" ] && [ -d "$HOME/micromamba/envs/KAHYPAR/lib" ]; then
    ENV_LIB="$HOME/micromamba/envs/KAHYPAR/lib"
fi
[ -n "$ENV_LIB" ] && [ -d "$ENV_LIB" ] && export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"

