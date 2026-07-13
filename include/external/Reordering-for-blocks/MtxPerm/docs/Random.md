# Random Permutations

## Overview

Random permutations serve as a **baseline** in the reordering survey. They provide a reference point for evaluating whether a reordering algorithm provides meaningful improvement over a random shuffle. Two variants are implemented: 1D (row-only) and 2D (independent row and column permutations).

## Algorithm

### Random 1D (Row Permutation)

1. Read the matrix dimensions from the MTX header (no full matrix load)
2. Generate a random permutation of `{0, 1, ..., n_rows-1}` using `numpy.random.permutation`
3. Save as 1-based indices on a single line

This produces a permutation usable for both ROW (`P*A`) and SYMMETRIC (`P*A*P^T`) applications.

### Random 2D (Row + Column Permutation)

1. Read the matrix dimensions from the MTX header
2. Generate **independent** random permutations for rows and columns:
   - Row permutation: random shuffle of `{0, ..., n_rows-1}`
   - Column permutation: random shuffle of `{0, ..., n_cols-1}`
3. Save as two lines: first line = row perm, second line = column perm

This produces an ASYMMETRIC permutation (`P_row * A * P_col^T`).

### Reproducibility

Both scripts use a fixed random seed (default: 42) for reproducibility via `numpy.random.seed`.

### Complexity

- **Time**: O(n) for Fisher-Yates shuffle
- **Space**: O(n)

## Why It Exists

Random permutations serve multiple purposes:
1. **Baseline**: Any meaningful reordering should outperform random
2. **Scrambling**: In random-base experiments, a random symmetric permutation is applied first to destroy the original structure, then reorderings are applied to test structure recovery
3. **Null hypothesis**: Statistical tests compare reordering benefits against the random baseline

## Effect on Matrix Structure

- **Destroys structure**: Bandwidth increases, block density decreases, cache locality is lost
- **Uniform scatter**: Nonzeros are spread uniformly across the matrix
- **Maximizes bandwidth**: The expected bandwidth is close to max(n_rows, n_cols)

## Implementation in MtxPerm

### Random 1D (`RANDOM/GB_random_permutation_1d.py`)

```python
n_rows, _ = read_mtx_dimensions(args.matrix)
perm = np.random.permutation(n_rows) + 1  # 1-based
```

- Only reads header (no full matrix load)
- Works for both square and rectangular matrices
- Single line output

### Random 2D (`RANDOM/GB_random_permutation_2d.py`)

```python
n_rows, n_cols = read_mtx_dimensions(args.matrix)
row_perm = np.random.permutation(n_rows) + 1
col_perm = np.random.permutation(n_cols) + 1
```

- Independent row and column permutations
- Two-line output (row perm, then col perm)
- Works for rectangular matrices

### Degree Permutation (`RANDOM/GB_degree_permutation.py`)

Although located in the RANDOM directory, this is the degree-based ordering (see [Degree.md](Degree.md)):

```python
degrees = np.diff(A.indptr)
perm = np.argsort(degrees, kind='stable') + 1
```

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| seed | 42 | `--seed S` | Random seed for reproducibility |

### Permutation Types

- **Random 1D**: ROW (`P*A`) or SYMMETRIC (`P*A*P^T`)
- **Random 2D**: ASYMMETRIC (`P_row * A * P_col^T`)
