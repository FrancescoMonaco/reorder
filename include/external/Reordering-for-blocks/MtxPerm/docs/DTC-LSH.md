# DTC-LSH: Tensor Core Aware Reordering via MinHash LSH

## Overview

DTC-LSH is the reordering algorithm from **DTC-SpMM** (ASPLOS 2024), which uses **MinHash Locality-Sensitive Hashing (LSH)** to cluster matrix rows by their sparsity pattern similarity. The algorithm performs two-level clustering: first for Tensor Core utilization (TCU-aware), then for cache locality (cache-aware). The name "TCA" (Tensor Core Aware) is used interchangeably with DTC-LSH.

## Algorithm

### Level 1: TCU-Aware Clustering (MinHash LSH)

1. **MinHash signature computation**: For each row, compute a MinHash signature by applying multiple hash functions to the set of nonzero column indices. The MinHash of two sets approximates their Jaccard similarity.
2. **LSH bucketing**: Group rows into buckets using LSH bands. Rows that hash to the same bucket in any band are candidate similar pairs.
3. **Cluster formation**: Group candidate pairs into clusters. Clusters with more than `thres` rows are split recursively.
4. **Ordering within clusters**: Rows within the same cluster are placed consecutively.

### Level 2: Cache-Aware Re-clustering

1. After the initial clustering, re-cluster the result for improved cache locality
2. Uses the same MinHash LSH mechanism but with different parameters tuned for cache-line level granularity

### Complexity

- **Time**: O(n * k) where k is the number of hash functions, plus O(n log n) for sorting
- **Space**: O(n * k) for MinHash signatures

## Why It Works

DTC-SpMM uses Tensor Cores for SpMM, which require small dense tiles. The two-level approach addresses two performance bottlenecks:

1. **TCU utilization**: Level 1 clustering groups rows with similar nonzero patterns, filling more positions in Tensor Core tiles (typically 16x16)
2. **Cache efficiency**: Level 2 re-clustering ensures that within each TCU-optimal group, rows are ordered for sequential cache access

MinHash LSH is particularly suited because:
- **Scalable**: O(n) per hash function, much faster than computing all-pairs Jaccard similarity
- **Probabilistic guarantees**: The probability that two rows are grouped together is proportional to their Jaccard similarity
- **GPU-acceleratable**: MinHash computation is embarrassingly parallel (optional GPU implementation via minhashcuda)

## Effect on Matrix Structure

- **Increases block density**: Similar to GROOT, rows with similar patterns are adjacent
- **Two-level locality**: Optimizes both Tensor Core tile fill and cache access patterns
- **Row-only reordering**: Only permutes rows
- **Works with the DTC-SpMM kernel**: The resulting ordering is specifically designed for the DTC-SpMM sparse format

## Implementation in MtxPerm

### Wrapper (`DTC-LSH/reorder.py`)

Thin wrapper that calls the original `TCA_reorder.py` from the DTC-SpMM codebase:

```python
# Convert MTX to NPZ format
A = mmread(matrix_path).tocoo()
np.savez(npz_path, src_li=A.row, dst_li=A.col, num_nodes=A.shape[0])

# Patch TCA_reorder.py with temp directory paths and run
cmd = [sys.executable, patched_script, "--dataset", dataset_name,
       "--thres", str(thres)]
result = subprocess.run(cmd)

# Read result and convert to .perm
reorder_id = np.load(reorder_npz)["reorder_id"]
perm_1based = reorder_id + 1
```

The wrapper patches the original script's hardcoded paths to use a temporary directory, making it portable.

### Original Script (`operators/DTC-SpMM/reordering/TCA_reorder.py`)

The original DTC-SpMM reordering script from ASPLOS'24.

### GPU-Accelerated MinHash (`DTC-LSH/minhashcuda_build/`)

Optional GPU implementation of MinHash computation using CUDA, providing significant speedup for large matrices.

### Parameters

| Parameter | Default | CLI flag | Description |
|-----------|---------|----------|-------------|
| thres | 128 | `--thres N` | Maximum cluster size before splitting |

### Environment

- Conda environment: `DTC-LSH` (activated via `tca_preprocess.sh`)
- Dependencies: CUDA, GCC 13.3.0, numpy, scipy

### Permutation Type

- **Row** (`P*A`) or **Symmetric** (`P*A*P^T`): Designed for row-only reordering.

## References

- Fan, R., et al. "DTC-SpMM: Bridging the Gap in Accelerating General Sparse Matrix Multiplication with Tensor Cores." *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)*, 2024.
