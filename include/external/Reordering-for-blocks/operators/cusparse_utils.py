#!/usr/bin/env python3
"""
Common utilities for cuSPARSE operators.
"""

import sys
from pathlib import Path
import numpy as np

# Import from MtxPerm
# Add MtxPerm directory to path (use insert to prioritize this path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'MtxPerm'))
from utils import load_and_permute_matrix, load_permutation_file


def convert_to_gpu(A_cpu, sparse_format='csr', blocksize=8):
    """
    Convert CPU sparse matrix to GPU in the specified format.

    Args:
        A_cpu: CPU sparse matrix (CSR format)
        sparse_format: Target format ('csr', 'coo')
        blocksize: Ignored (BSR uses C++ implementation)

    Returns:
        GPU sparse matrix in requested format
    """
    import cupy as cp
    import cupyx.scipy.sparse as cupyx_sp

    if sparse_format == 'csr':
        return cupyx_sp.csr_matrix(A_cpu)
    elif sparse_format == 'bsr':
        # BSR format is handled by C++ binary (operators/cusparse_bsr_spmm)
        # This should not be called from Python operators
        raise NotImplementedError(
            "BSR format not supported in Python. Use C++ binary: operators/cusparse_bsr_spmm"
        )
    elif sparse_format == 'coo':
        return cupyx_sp.coo_matrix(A_cpu.tocoo())
    else:
        raise ValueError(f"Unsupported format: {sparse_format}")


def time_operation(operation, n_iterations=5):
    """
    Time a GPU operation using CUDA events.

    Args:
        operation: Callable that performs the GPU operation
        n_iterations: Number of timing iterations

    Returns:
        Average time in milliseconds
    """
    import cupy as cp

    # Warmup
    _ = operation()
    cp.cuda.Stream.null.synchronize()

    # Timed runs using CUDA events
    timings = []
    for _ in range(n_iterations):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        _ = operation()
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        timings.append(elapsed_ms)

    return sum(timings) / len(timings)


def print_timer(label, time_ms):
    """
    Print timing information in C++-compatible format.
    
    Args:
        label: Timer label (e.g., 'loading', 'execution')
        time_ms: Time in milliseconds
    """
    print(f"<Timer>[{label}] {time_ms:.6f} ms", flush=True)
