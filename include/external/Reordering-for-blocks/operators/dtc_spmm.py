#!/usr/bin/env python3
import sys
import argparse
import torch
import numpy as np
import scipy.sparse as sp
import time

try:
    import DTCSpMM
except ImportError:
    print("Warning: DTCSpMM module not found.")

from cusparse_utils import load_and_permute_matrix, print_timer
from config import SPMM_N_COLS_DEFAULT, N_ITERATIONS_DEFAULT, PERM_TYPE_DEFAULT

def main():
    parser = argparse.ArgumentParser(description='DTC-SpMM benchmark')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')
    parser.add_argument('--n-iterations', type=int, default=N_ITERATIONS_DEFAULT, help=f'Number of timing iterations (default: {N_ITERATIONS_DEFAULT})')
    parser.add_argument('--exeplan', type=str, default='float4_split', help='Execution plan (e.g., float4_split, float_nonsplit)')
    parser.add_argument('--balance', action='store_true', help='Use load balancing')
    
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
    
    # DTC-SpMM constants
    BLK_H = 16
    BLK_W = 8
    
    # Preprocessing
    t0 = time.perf_counter()
    
    num_rows = m
    num_row_windows = (num_rows + BLK_H - 1) // BLK_H
    
    # Prepare tensors
    column_index = torch.from_numpy(A_cpu.indices).int()
    row_pointers = torch.from_numpy(A_cpu.indptr).int()
    
    edgeToColumn = torch.zeros(nnz, dtype=torch.int)
    edgeToRow = torch.zeros(nnz, dtype=torch.int)
    blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
    
    # Move to GPU
    column_index_cuda = column_index.cuda()
    row_pointers_cuda = row_pointers.cuda()
    blockPartition_cuda = blockPartition.cuda()
    edgeToColumn_cuda = edgeToColumn.cuda()
    edgeToRow_cuda = edgeToRow.cuda()
    
    # DTCSpMM Preprocess
    RowWindowOffset, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, block_count = \
        DTCSpMM.preprocess_gpu(column_index_cuda, row_pointers_cuda, num_rows, BLK_H, BLK_W, 
                               blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
                               
    preprocessing_ms = (time.perf_counter() - t0) * 1000
    
    # float4_split requires embedding_dim >= 128; fall back for small n_cols
    if args.n_cols <= 128 and args.exeplan == 'float4_split':
        args.exeplan = 'float4_nonsplit'

    # Prepare dense matrices
    feat_size = args.n_cols
    X = torch.ones((num_rows, feat_size)).cuda()
    
    # DTC-SpMM internally runs the kernel EXE_TIME=1000 times per call.
    # We time the whole call with CUDA events and divide by 1000 to get per-op time.
    DTC_INTERNAL_ITERS = 1000

    # Warmup
    if not args.balance:
        DTCSpMM.run_DTCSpMM(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, nnz, args.exeplan)
    else:
        DTCSpMM.run_DTCSpMM_balance(X, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, args.exeplan)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    timings = []
    for _ in range(args.n_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if not args.balance:
            DTCSpMM.run_DTCSpMM(X, RowWindowOffset, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, nnz, args.exeplan)
        else:
            DTCSpMM.run_DTCSpMM_balance(X, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, num_rows, args.exeplan)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    avg_ms = sum(timings) / len(timings) / DTC_INTERNAL_ITERS

    print_timer("loading", loading_ms)
    print_timer("preprocessing", preprocessing_ms)
    print_timer("operation", avg_ms)

    # Calculate GFLOPs using original nnz (consistent with other kernels)
    gflops = (2 * nnz * args.n_cols) / (avg_ms * 1e6) if avg_ms > 0 else 0
    print(f"GFLOPs: {gflops:.2f}")

if __name__ == '__main__':
    main()
