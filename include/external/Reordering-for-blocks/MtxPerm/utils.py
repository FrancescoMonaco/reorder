#!/usr/bin/env python3
"""
Common utilities for Matrix Permutations.
"""

import numpy as np
from scipy.io import mmread

def load_permutation_file(perm_path, perm_type='ROW'):
    """
    Load permutation file and return row and/or column permutations.
    
    Args:
        perm_path: Path to permutation file
        perm_type: Type of permutation:
            'ROW' - single line with row permutation (only permute rows)
            'SYMMETRIC' - single line, apply to both rows and cols (requires square matrix)
            'ASYMMETRIC' - two lines: first=rows, second=cols
    
    Returns:
        Tuple of (row_perm, col_perm) as numpy arrays (0-indexed), or (perm, None) for ROW
    """
    perm_type = perm_type.upper()
    
    with open(perm_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if perm_type == 'ROW':
        # Single permutation - only apply to rows
        perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1  # Convert to 0-based
        return perm, None
    elif perm_type == 'SYMMETRIC':
        # Single permutation - apply to both rows and columns (requires square matrix)
        perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        return perm, perm.copy()  # Return copy to avoid aliasing
    elif perm_type == 'ASYMMETRIC':
        # Two permutations - read both lines
        if len(lines) < 2:
            raise ValueError(f"ASYMMETRIC permutation requires 2 lines, found {len(lines)}")
        row_perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        col_perm = np.fromstring(lines[1], sep=' ', dtype=np.int64) - 1
        return row_perm, col_perm
    else:
        raise ValueError(f"Unknown permutation type: {perm_type}. Use ROW, SYMMETRIC, or ASYMMETRIC")


def apply_permutation(A_csr, perm_path, perm_type):
    """
    Apply a permutation from file to an already-loaded CSR matrix.

    Args:
        A_csr: Sparse matrix in CSR format
        perm_path: Path to permutation file (1-indexed)
        perm_type: Type of permutation ('ROW', 'SYMMETRIC', or 'ASYMMETRIC')

    Returns:
        Permuted sparse matrix in CSR format
    """
    row_perm, col_perm = load_permutation_file(perm_path, perm_type)
    m, n = A_csr.shape

    if row_perm is not None:
        if len(row_perm) != m:
            raise ValueError(f"Row permutation size ({len(row_perm)}) doesn't match matrix rows ({m})")
        A_csr = A_csr[row_perm, :]

    if col_perm is not None:
        if len(col_perm) != n:
            raise ValueError(f"Column permutation size ({len(col_perm)}) doesn't match matrix columns ({n})")
        A_csr = A_csr[:, col_perm]

    if perm_type.upper() == 'SYMMETRIC' and m != n:
        raise ValueError(f"SYMMETRIC permutation requires square matrix, got {m}x{n}")

    A_csr.sort_indices()
    A_csr.sum_duplicates()
    return A_csr


def load_and_permute_matrix(matrix_path, perm_path=None, perm_type='ROW', dtype=np.float32,
                            base_perm_path=None, base_perm_type='SYMMETRIC'):
    """
    Load a matrix from MatrixMarket format and apply optional permutations.

    Args:
        matrix_path: Path to Matrix Market file
        perm_path: Optional path to permutation file (1-indexed)
        perm_type: Type of permutation ('ROW', 'SYMMETRIC', or 'ASYMMETRIC')
        dtype: Data type for the matrix (default: np.float32)
        base_perm_path: Optional path to base permutation file (applied first)
        base_perm_type: Type of base permutation (default: 'SYMMETRIC')

    Returns:
        Sparse matrix in CSR format with permutations applied
    """
    # Load sparse matrix from MatrixMarket
    A_cpu = mmread(matrix_path).tocsr().astype(dtype)

    # Apply base permutation first (e.g., random scramble)
    if base_perm_path:
        A_cpu = apply_permutation(A_cpu, base_perm_path, base_perm_type)

    # Apply main permutation on top
    if perm_path:
        A_cpu = apply_permutation(A_cpu, perm_path, perm_type)

    return A_cpu

def save_permutation(path, perm):
    """Save permutation to file (1-based, space separated)."""
    # Ensure perm is 0-based numpy array
    perm_1based = perm + 1
    np.savetxt(path, perm_1based.reshape(1, -1), fmt='%d', delimiter=' ')
