#!/bin/bash
# Preprocess script for GROOT reordering
# Activates the conda environment with pynndescent and scipy
# source /usr/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
# Initialize the micromamba shell hook (required in non-interactive Slurm
# batch shells, otherwise "critical libmamba Shell not initialized")
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell bash)"
elif [ -d "$HOME/micromamba" ]; then
    eval "$($HOME/micromamba/bin/micromamba shell hook --shell bash)"
else
    echo "ERROR: micromamba binary could not be found." >&2
    exit 1
fi
micromamba activate GROOT
# Sanity check: make sure we really are in GROOT,
# instead of silently continuing with the wrong python
if [[ "${CONDA_DEFAULT_ENV:-}" != "GROOT" ]]; then
    echo "ERROR: Failed to activate GROOT environment (got: ${CONDA_DEFAULT_ENV:-none})" >&2
    exit 1
fi
