# MtxPerm/

Matrix permutation generation tools implementing various reordering algorithms.

## Subdirectories

### SPARSEBASE/

C++ tools using the [SparseBase](https://github.com/sparcityeu/SparseBase) library.

**Algorithms:**
- `SB_rcm` - Reverse Cuthill-McKee (bandwidth reduction)
- `SB_degree` - Degree-based ordering
- `SB_gray` - Gray code reordering
- `SB_amd` - Approximate Minimum Degree
- `SB_metis` - METIS graph partitioning
- `SB_rabbit` - Rabbit reordering
- `SB_slashburn` - SlashBurn algorithm

**Build:**
```bash
cd MtxPerm/SPARSEBASE
mkdir build && cd build
cmake .. -DSparseBase_DIR=/path/to/sparsebase
make
```

**Usage:**
```bash
./reorder --mtx input.mtx --output output.perm --algorithm rcm
```

### RANDOM/

Python scripts using `python-graphblas` for random permutations.

**Algorithms:**
- `random1D` - Random row permutation
- `random2D` - Random row + column permutation

**Usage:**
```bash
python MtxPerm/RANDOM/random_perm.py input.mtx output.perm --type 1D
python MtxPerm/RANDOM/random_perm.py input.mtx output.perm --type 2D
```

### SPARTA/

Interface to SPARTA sparse blocking algorithm for tensor core optimization.

**Algorithm:**
- `SPARTA_reorder` - Optimizes block structure for tensor cores

### GROOT/

Interface to Groot graph-centric row reordering (Chen et al., EuroSys 2025).

**Algorithm:**
- `GROOT_reorder` - k-NN graph + MST + DFS traversal for Tensor Core optimization

**Prerequisites:** CMake >= 3.22, CUDA Toolkit, KGraph, Boost

**Install:**
```bash
export KGRAPH_ROOT=/path/to/kgraph
cd MtxPerm/GROOT && ./install.sh
```

**Usage:**
```bash
python3 MtxPerm/GROOT/reorder.py input.mtx output.perm
```

### ANALYSIS/

Matrix structure analysis tools (not permutation generation).

Computes metrics:
- Bandwidth (max, avg, lower, upper)
- Block density at various block sizes (4, 8, 16, 32, 64, 128)
- Locality metrics (row spread, vertical adjacency ratio)

**Output:** JSON with all computed metrics

**Usage:**
```bash
./analyze --mtx input.mtx [--perm input.perm --perm-type SYMMETRIC]
```

## utils.py

Shared Python utilities for permutation handling:

```python
from MtxPerm.utils import load_permutation, apply_permutation

# Load permutation (handles 1-based to 0-based conversion)
row_perm, col_perm = load_permutation("path/to/perm.perm")

# Apply to scipy sparse matrix
reordered = apply_permutation(matrix, row_perm, col_perm)
```

## Output Format

All tools output permutations in the standard `.perm` format:
- 1-based indices
- Space-separated
- One line for symmetric/row, two lines for asymmetric

See [perms/README.md](../perms/README.md) for format details.
