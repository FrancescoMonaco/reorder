#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import scipy.sparse as sp
import time

try:
    import FS_Block_gpu
    import FS_SpMM
except ImportError as e:
    print(f"Error: FlashSparse modules not found. {e}")
    sys.exit(1)

# Import from MtxPerm directly to avoid cupy dependency
sys.path.append(str(Path(__file__).parent.parent / 'MtxPerm'))
from utils import load_and_permute_matrix

from config import SPMM_N_COLS_DEFAULT, N_ITERATIONS_DEFAULT, PERM_TYPE_DEFAULT

def print_timer(label, time_ms):
    """Print timing information in C++-compatible format."""
    print(f"<Timer>[{label}] {time_ms:.6f} ms", flush=True)

def main():
    parser = argparse.ArgumentParser(description='FlashSparse SpMM benchmark')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')
    parser.add_argument('--n-iterations', type=int, default=N_ITERATIONS_DEFAULT, help=f'Number of timing iterations (default: {N_ITERATIONS_DEFAULT})')
    parser.add_argument('--mode', type=str, default='16_1', choices=['16_1', '8_1', '8_1_balance', '8_1_map'], help='FlashSparse mode (default: 16_1)')
    
    args = parser.parse_args()

    # Load and permute matrix
    t0 = time.perf_counter()
    A_cpu = load_and_permute_matrix(args.matrix_path, args.perm, args.perm_type,
                                    base_perm_path=args.base_perm, base_perm_type=args.base_perm_type)
    loading_ms = (time.perf_counter() - t0) * 1000
    
    # Convert to CSR if not already
    if not sp.isspmatrix_csr(A_cpu):
        A_cpu = A_cpu.tocsr()

    m, k = A_cpu.shape
    nnz = A_cpu.nnz
    
    # Preprocessing for FlashSparse
    t0 = time.perf_counter()
    
    # Padding logic from FlashSparse mdataset2.py
    num_nodes_ori = m
    num_nodes = m
    
    # Determine window and wide based on mode
    if args.mode == '16_1':
        window = 16
        wide = 8
        pad_align = 16
    elif args.mode in ['8_1', '8_1_balance', '8_1_map']:
        window = 8
        wide = 8
        pad_align = 8
    
    if num_nodes % pad_align != 0:
        num_nodes = num_nodes + pad_align - (num_nodes % pad_align)

    # Pad columns as well for non-square matrices
    num_cols = k
    if num_cols % pad_align != 0:
        num_cols = num_cols + pad_align - (num_cols % pad_align)

    # Prepare tensors
    # FlashSparse expects IntTensor for indices and HalfTensor for values
    # Use all ones for values to avoid float16 overflow (FlashSparse is for GNN-style SpMM)
    row_pointers = torch.IntTensor(A_cpu.indptr)
    column_index = torch.IntTensor(A_cpu.indices)
    values = torch.ones(nnz, dtype=torch.float16)

    # CRITICAL: FlashSparse expects row_pointers to have length num_nodes+1 (padded)
    # Pad row_pointers by appending nnz for the extra empty rows
    if num_nodes > m:
        extra_rows = num_nodes - m
        padding = torch.full((extra_rows,), nnz, dtype=torch.int32)
        row_pointers = torch.cat([row_pointers, padding]) 
    
    # Preprocess
    # Note: FS_Block_gpu functions return tensors that are likely on GPU or ready for FS_SpMM
    if args.mode == '8_1_balance':
        partSize = 32 # Default in spmm_fp16_test_args.py
        row_pointers, column_index, values, t_window_rowTensor, t_atomicTensor, exe = FS_Block_gpu.preprocess_gpu_fs_balance(
            row_pointers, column_index, num_nodes, nnz, window, wide, partSize
        )
    else:
        row_pointers, column_index, values, exe = FS_Block_gpu.preprocess_gpu_fs(
            row_pointers, column_index, num_nodes, nnz, window, wide
        )
        
    preprocessing_ms = (time.perf_counter() - t0) * 1000

    # Prepare Dense Matrix B (Features)
    # B is num_cols x n_cols (use padded column dimension to match matrix)
    # Ensure B is on GPU and FP16
    B_gpu = torch.randn(num_cols, args.n_cols).half().cuda()
    
    # Run SpMM
    epoches = args.n_iterations
    
    # FS_SpMM functions return (result_matrix, average_time_ms)
    if args.mode == '16_1':
         _, avg_ms = FS_SpMM.forward_fp16_16(
            row_pointers, column_index, values, B_gpu, num_nodes, args.n_cols, num_nodes_ori, epoches
        )
    elif args.mode == '8_1':
        _, avg_ms = FS_SpMM.forward_fp16_test(
            row_pointers, column_index, values, B_gpu, num_nodes, args.n_cols, num_nodes_ori, epoches, 4
        )
    elif args.mode == '8_1_map':
         _, avg_ms = FS_SpMM.forward_fp16_map(
            row_pointers, column_index, values, B_gpu, num_nodes, args.n_cols, num_nodes_ori, epoches, 4
        )
    elif args.mode == '8_1_balance':
         _, avg_ms = FS_SpMM.forward_fp16_balance(
            row_pointers, column_index, values, t_window_rowTensor, t_atomicTensor, B_gpu, num_nodes, args.n_cols, num_nodes_ori, epoches
        )
    
    avg_time_ms = avg_ms.item()

    print_timer("loading", loading_ms)
    print_timer("preprocessing", preprocessing_ms)
    print_timer("operation", avg_time_ms)

    # Calculate GFLOPs
    gflops = (2 * nnz * args.n_cols) / (avg_time_ms * 1e6)
    print(f"GFLOPs: {gflops:.2f}")

if __name__ == '__main__':
    sys.exit(main())
