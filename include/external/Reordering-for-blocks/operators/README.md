# SpMM Operators

This directory contains SpMM (Sparse Matrix-Matrix Multiplication) operator implementations and installation scripts for various GPU kernels.

## Quick Start

### Prerequisites

1. **CUDA Toolkit** (>= 11.8 recommended)
   ```bash
   module load cuda  # On clusters
   ```

2. **PyTorch with CUDA support**
   ```bash
   # Conda (recommended)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Or pip
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Python dependencies**
   ```bash
   pip install numpy scipy
   ```

### Installing Operator Kernels

Use the master installer for interactive setup:
```bash
cd operators/
./install_all.sh          # Interactive mode
./install_all.sh --check  # Check prerequisites only
./install_all.sh --all    # Install all operators
```

Or install individually:
```bash
./flashsparse_install.sh  # FlashSparse (PPoPP 2025) - uses conda env FlashSparse
./install_dtc.sh          # DTC-SpMM - uses conda env DTCSpMM
./install_aspt.sh         # ASpT (Adaptive Sparse Tiling)
./install_smat.sh         # SMaT
```

## Available Operators

| Operator | Module | Description | Install Script |
|----------|--------|-------------|----------------|
| cuSPARSE CSR | (PyTorch) | NVIDIA cuSPARSE CSR SpMM | Built-in |
| cuSPARSE BSR | (PyTorch) | NVIDIA cuSPARSE BSR SpMM | Built-in |
| FlashSparse | `FS_SpMM`, `FS_Block_gpu` | Tensor Core SpMM (PPoPP'25) | `install_flashsparse.sh` |
| DTC-SpMM | `DTCSpMM` | Dynamic Tensor Core SpMM | `install_dtc.sh` |
| ASpT | `aspt` | Adaptive Sparse Tiling | `install_aspt.sh` |
| SMaT | `smat` | Sparse Matrix Tensor Core | `install_smat.sh` |

## Usage Examples

### cuSPARSE (always available)
```bash
python cusparse_csr_spmm.py matrix.mtx --n-cols 32
python cusparse_bsr_spmm.py matrix.mtx --n-cols 32 --blocksize 8
```

### FlashSparse
```bash
# Activate environment
source operators/flashsparse_preprocess.sh
# Or manually:
module load CUDA/12.1.1
conda activate FlashSparse

python flashsparse_spmm.py matrix.mtx --n-cols 32 --mode 16_1
```

### With Reordering
```bash
python cusparse_csr_spmm.py matrix.mtx --perm permutation.perm --perm-type SYMMETRIC --n-cols 32
```

## Permutation Types

- `ROW` - Apply permutation to rows only
- `SYMMETRIC` - Apply same permutation to rows and columns (P * A * P^T)
- `ASYMMETRIC` - Apply different permutations to rows and columns

## Troubleshooting

### "Module not found" errors
Make sure you've run the installation script and the build succeeded:
```bash
python -c "import FS_SpMM; print('OK')"  # Test FlashSparse
python -c "import DTCSpMM; print('OK')"  # Test DTC-SpMM
```

### GPU architecture mismatch
Set the architecture explicitly:
```bash
TORCH_CUDA_ARCH_LIST="8.6" ./install_flashsparse.sh --clean
```

### DTC-SpMM runtime library errors
Set LD_LIBRARY_PATH before running:
```bash
export LD_LIBRARY_PATH="$PWD/DTC-SpMM/third_party/glog/build/lib:$PWD/DTC-SpMM/third_party/sputnik/build/sputnik:$LD_LIBRARY_PATH"
```

## Cluster Usage (SLURM)

Add to your job script:
```bash
# For cuSPARSE, ASPT, SMAT
module load cuda
source ~/.venv/bin/activate

# For FlashSparse
source operators/flashsparse_preprocess.sh

# For DTC-SpMM
source operators/dtc_preprocess.sh
```
