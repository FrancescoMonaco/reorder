# SPARTA (SParse blocking Algorithm for Tensor core Architectures)

## Overview

SPARTA is a sparse matrix blocking algorithm designed to maximize block density for Tensor Core architectures. Unlike traditional reordering algorithms that consider the full graph structure, SPARTA directly optimizes the **grouping of rows** into blocks of a specified size, targeting the block structure needed by block-sparse formats used in Tensor Core SpMM.

## Algorithm

SPARTA uses a similarity-based approach to group matrix rows:

### 1. Similarity Computation

For each pair of rows, compute a similarity score based on the overlap of their nonzero column indices. The similarity metric considers whether rows would produce dense blocks if placed together.

### 2. Blocking via DenseAMP (Algorithm 5)

The default algorithm (algo=5, DenseAMP) performs:

1. **Row grouping**: Group rows into blocks of the specified block size (B)
2. **Greedy assignment**: For each unassigned row, find the existing group where it would contribute the most nonzeros to dense block tiles
3. **Threshold filtering**: Use a similarity threshold (tau) to decide when to start a new group vs. join an existing one
4. **Two-dimensional blocking**: Separately block rows (row block size B) and columns (column block size b), creating B x b tile groups

### 3. Group-to-Permutation Conversion

The blocking output is a **grouping file** where each row is assigned a group ID. The permutation is obtained by sorting rows by their group ID using a stable sort:

```python
grouping = [group_id_for_row_i for each row]
perm = np.argsort(grouping, kind='stable')
```

### Complexity

- **Time**: Depends on the algorithm variant, typically O(n * nnz / B)
- **Space**: O(n)

## Why It Works

Tensor Cores require data in small dense tiles (e.g., 16x16). If a sparse matrix has blocks where most entries are zero, the Tensor Core is wasted on multiplying zeros. SPARTA directly optimizes for this by:

1. **Block-aware grouping**: Considers the actual block tile dimensions when deciding which rows belong together
2. **Density maximization**: Places rows with overlapping nonzero columns in the same block to maximize tile density
3. **Configurable tile size**: Adapts to the specific Tensor Core tile dimensions of the target hardware

## Effect on Matrix Structure

- **Maximizes block density**: Directly optimizes the fraction of nonzeros per block
- **Row grouping**: Creates groups of rows with similar sparsity patterns
- **Does not explicitly reduce bandwidth**: Focuses on block density rather than diagonal proximity
- **Block-size aware**: The resulting permutation is optimized for a specific block size

## Implementation in MtxPerm

### Wrapper (`SPARTA/reorder.py`)

Calls the pre-compiled SPARTA binary and converts the grouping output to a permutation file:

```python
cmd = [
    str(sparta_bin),
    '-f', str(matrix_path),     # input matrix
    '-R', '1',                   # MTX format
    '-b', str(block_size),       # column block size
    '-B', str(block_size),       # row block size
    '-t', str(tau),              # similarity threshold
    '-o', str(temp_output_prefix),
    '-a', str(algo),             # algorithm (5 = DenseAMP)
    '-v', '1'                    # verbosity
]
subprocess.run(cmd, check=True)

# Parse grouping and convert to permutation
grouping = np.array([int(line) for line in open(grouping_file)])
perm = np.argsort(grouping, kind='stable')
```

### SPARTA Binary

The binary is built from the original SPARTA repository and reads MTX files directly:

- `-f`: Input matrix file (MTX format with `-R 1`)
- `-b`: Column block size (default: 32)
- `-B`: Row block size (default: 32)
- `-t`: Tau - similarity threshold (default: 0.5)
- `-a`: Algorithm variant (default: 5 for DenseAMP)
- `-o`: Output prefix (appends `.g` for grouping file)

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| block_size | 32 | `--block-size` | Block size for row and column blocking |
| tau | 0.5 | `--tau` | Similarity threshold (0-1; higher = more selective) |
| algo | 5 | `--algo` | Blocking algorithm (5 = DenseAMP) |

### Permutation Type

- **Row** (`P*A`): Designed as a row-only reordering. Can also be applied symmetrically.

## References

- Sparsity Pattern-Aware Blocking for Sparse Matrix Computations. (SPARTA repository and documentation)
