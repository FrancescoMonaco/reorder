# GROOT/

Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplications on Tensor Cores.

## Reference

```bibtex
@inproceedings{chen2025groot,
  title={Groot: Graph-Centric Row Reordering with Tree for Sparse Matrix Multiplications on Tensor Cores},
  author={Chen, Y. and Xie, J. and Teng, S. and Zeng, W. and Yu, J. X.},
  booktitle={Proceedings of the Twentieth European Conference on Computer Systems (EuroSys)},
  pages={803-817},
  year={2025},
  month={March}
}
```

**Repository:** https://github.com/yuang-chen/Groot-EuroSys25

## Algorithm

Groot reorders sparse matrix rows to improve block density for Tensor Core operations:

1. **k-NN Graph Construction**: Build a k-nearest neighbor graph where nodes are matrix rows and edges connect similar rows (using Jaccard similarity)
2. **MST Extraction**: Extract a Minimum Spanning Tree from the k-NN graph
3. **DFS Traversal**: Perform DFS traversal of the MST, visiting most similar neighbors first

Rows with similar sparsity patterns are placed close together, improving data locality and block utilization.

## Implementation Note

This wrapper uses the **Python implementation** from the GROOT repository, which correctly implements:
- **PyNNDescent** for fast k-NN graph construction
- **Jaccard similarity** metric for comparing row sparsity patterns
- **Neighbor sorting by similarity** during DFS traversal

The original C++ implementation has a bug where KNN parameters are set too low (k=20 instead of k=200), producing random-like permutations. The Python implementation produces correct results.

## Prerequisites

- Conda (miniconda/anaconda)
- Python 3.8+

Dependencies (installed automatically):
- pynndescent
- scipy
- numpy

## Installation

```bash
./MtxPerm/GROOT/install.sh
```

This creates a conda environment named `GROOT` with all required dependencies.

## Usage

```bash
# Activate environment first
source MtxPerm/GROOT/groot_preprocess.sh

# Run reordering
python3 MtxPerm/GROOT/reorder.py input.mtx output.perm

# With custom KNN parameter
python3 MtxPerm/GROOT/reorder.py input.mtx output.perm --knn 32
```

## Output Format

Standard permutation format:
- 1-based indices
- Space-separated on single line
- Row permutation (symmetric: apply to both rows and columns)

## Algorithm Tag

- `GROOT_reorder` - k-NN + MST + DFS reordering
