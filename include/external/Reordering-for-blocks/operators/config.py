"""
Global configuration parameters for cuSPARSE operators.
"""

# SpMM/SpMV Operation Parameters
ALPHA_DEFAULT = 1.0
BETA_DEFAULT = 0.0

# SpMM Dense Matrix Parameters
SPMM_N_COLS_DEFAULT = 32

# Timing Parameters
N_ITERATIONS_DEFAULT = 5

# BSR Format Parameters
BSR_BLOCKSIZE_DEFAULT = 8

# Permutation Parameters
PERM_TYPE_DEFAULT = 'ROW'
