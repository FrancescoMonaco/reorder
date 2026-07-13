# Tests for Distributed MMIO

## 1) Setup Test Matrices

Use [MtxMan](https://github.com/ThomasPasquali/MtxMan/tree/main):

```bash
mtxman sync matrices_test.yaml -bmtx -kmtx
mtxman sync matrices_partitioning.yaml -bmtx -kmtx
```

## 2) Build

```bash
cmake -B build && cmake --build build
```

## 3) Run Tests

Use [SbatchMan](https://sbatchman.readthedocs.io/en/latest/):

```bash
sbatchman init
sbatchman set-cluster-name local
sbatchman configure -f configs.yaml

# Generate partitioned matrices data
sbatchman launch -f jobs_show_partitioning.yaml
```

## 4) Parse Results

```bash
# Turn partitioned matrices data into figures
python3 generate_partitioning_images.py
```

You will find the images in the `results/` subfolder.