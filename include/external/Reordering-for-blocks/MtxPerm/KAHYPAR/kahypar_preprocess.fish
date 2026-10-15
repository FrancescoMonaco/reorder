#!/usr/bin/env fish
# Preprocess script for mt-kahypar reordering.
#
# Makes the mt-kahypar shared library and its TBB dependency discoverable at
# runtime. install.sh installs libmtkahypar into a local prefix (under lib64/
# on 64-bit Linux, lib/ elsewhere) and provisions TBB inside the KAHYPAR
# micromamba environment. This adds all of those to LD_LIBRARY_PATH so the
# kahypar_perm binary runs whether or not the env is already activated.
#
# This fish-compatible variant is provided alongside the original
# kahypar_preprocess.sh (Bash). Use it when your shell is fish.

# Script directory: fish-native approach without external dirname/realpath.
set SCRIPT_PATH (status --current-filename)
set SCRIPT_DIR (string match -ra '(/.*)' $SCRIPT_PATH)[1]
set INSTALL_DIR "$SCRIPT_DIR/mt-kahypar-install"

# mt-kahypar's own shared library (GNUInstallDirs -> lib64 on 64-bit Linux).
for d in "$INSTALL_DIR/lib64" "$INSTALL_DIR/lib"
    if test -d "$d"
        set -x LD_LIBRARY_PATH "$d" $LD_LIBRARY_PATH
    end
end

# TBB lives in the KAHYPAR micromamba env. Prefer the active env if present,
# otherwise fall back to the default micromamba env location used by install.sh.
if set -q CONDA_PREFIX
    set ENV_LIB "$CONDA_PREFIX/lib"
end
if test -z "$ENV_LIB" && test -d "$HOME/micromamba/envs/KAHYPAR/lib"
    set ENV_LIB "$HOME/micromamba/envs/KAHYPAR/lib"
end
if test -n "$ENV_LIB" && test -d "$ENV_LIB"
    set -x LD_LIBRARY_PATH "$ENV_LIB" $LD_LIBRARY_PATH
end

