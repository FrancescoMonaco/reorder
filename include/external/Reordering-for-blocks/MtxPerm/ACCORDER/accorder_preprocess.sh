#!/bin/bash
# Preprocess script for accOrder reordering
# Boost is header-only (only needed at compile time, not runtime).
# At runtime we only need libnuma. Load numactl but avoid loading
# GCCcore/binutils which can cause "Illegal instruction" on some nodes.
# module load numactl/2.0.16-GCCcore-12.3.0

# Strip EasyBuild GCCcore libs from LD_LIBRARY_PATH to avoid picking up
# incompatible libstdc++/libgomp compiled for different CPU features.
# export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "GCCcore" | grep -v "binutils" | tr '\n' ':' | sed 's/:$//')
