#!/usr/bin/env python3
"""Show a sparse matrix and its block structure side by side."""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 200
block_size = 20  # 20x20 blocks → 10x10 grid
n_blocks = N // block_size

# --- Pick 10% of blocks (20x20) + 1/4 that in double-size blocks (40x40) ---
total_blocks = n_blocks * n_blocks
max_nnz_per_block = int(block_size * block_size * 0.05)

# Small blocks (20x20): 10% of grid
n_small = int(total_blocks * 0.10)
flat_idx_small = rng.choice(total_blocks, size=n_small, replace=False)

# Large blocks (40x40 = 2x2 groups of 20x20): 1/4 of n_small
n_big_blocks = N // (2 * block_size)
total_big = n_big_blocks * n_big_blocks
n_large = max(1, n_small // 4)
flat_idx_large = rng.choice(total_big, size=n_large, replace=False)

rows, cols, vals = [], [], []
occupied_small = set()

# Place small blocks
for idx in flat_idx_small:
    br, bc = divmod(idx, n_blocks)
    occupied_small.add((br, bc))
    nnz = rng.integers(1, max_nnz_per_block + 1)
    r_idx = rng.integers(br * block_size, (br + 1) * block_size, size=nnz)
    c_idx = rng.integers(bc * block_size, (bc + 1) * block_size, size=nnz)
    rows.extend(r_idx)
    cols.extend(c_idx)
    vals.extend(rng.random(nnz))

# Place large (40x40) blocks — each covers a 2x2 group of small blocks
big_block_size = 2 * block_size
max_nnz_big = int(big_block_size * big_block_size * 0.05)
for idx in flat_idx_large:
    br_big, bc_big = divmod(idx, n_big_blocks)
    # Mark constituent small blocks as occupied
    for dr in range(2):
        for dc in range(2):
            occupied_small.add((br_big * 2 + dr, bc_big * 2 + dc))
    nnz = rng.integers(1, max_nnz_big + 1)
    r_idx = rng.integers(br_big * big_block_size, (br_big + 1) * big_block_size, size=nnz)
    c_idx = rng.integers(bc_big * big_block_size, (bc_big + 1) * big_block_size, size=nnz)
    rows.extend(r_idx)
    cols.extend(c_idx)
    vals.extend(rng.random(nnz))

# Build flat_idx for the row/col fill-in logic below
flat_idx = [br * n_blocks + bc for (br, bc) in occupied_small]

A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

# --- Ensure at least one nonzero per row and column (within occupied blocks) ---
# Find which block-rows and block-cols have at least one occupied block
occ_block_rows = set()  # block rows that have an occupied block
occ_block_cols = set()  # block cols that have an occupied block
block_map = {}  # (br, bc) pairs that are occupied
for idx in flat_idx:
    br, bc = divmod(idx, n_blocks)
    block_map.setdefault(br, []).append(bc)
    block_map.setdefault(-bc - 1, []).append(br)  # negative key = col lookup
    occ_block_rows.add(br)
    occ_block_cols.add(bc)

A = A.tolil()
for i in range(N):
    if A[i, :].nnz == 0:
        br = i // block_size
        if br in occ_block_rows:
            bc = rng.choice(block_map[br])
        else:
            bc = rng.choice(list(occ_block_cols))
        j = rng.integers(bc * block_size, (bc + 1) * block_size)
        A[i, j] = rng.random()
for j in range(N):
    if A[:, j].nnz == 0:
        bc = j // block_size
        key = -bc - 1
        if key in block_map:
            br = rng.choice(block_map[key])
        else:
            br = rng.choice(list(occ_block_rows))
        i = rng.integers(br * block_size, (br + 1) * block_size)
        A[i, j] = rng.random()
A = A.tocsr()
A.eliminate_zeros()

# --- Block occupancy (full NxN image) ---
block_occ = np.zeros((n_blocks, n_blocks), dtype=bool)
coo = A.tocoo()
for r, c in zip(coo.row, coo.col):
    block_occ[r // block_size, c // block_size] = True

blocked = np.kron(block_occ.astype(float), np.ones((block_size, block_size)))

# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
ticks = np.arange(0, N + 1, block_size)

dense = np.zeros((N, N))
dense[A.nonzero()] = 1.0

# Left: sparsity pattern
ax1.imshow(dense, cmap="Greys", origin="upper", interpolation="nearest")
overall_density = A.nnz / (N * N) * 100
ax1.set_title(f"{A.nnz} entries, {overall_density:.1f}% density")
ax1.set_xticks(ticks - 0.5)
ax1.set_xticklabels(ticks)
ax1.set_yticks(ticks - 0.5)
ax1.set_yticklabels(ticks)

# Middle: block occupancy
ax2.imshow(blocked, cmap="Blues", origin="upper", interpolation="nearest")
for i in range(n_blocks + 1):
    coord = i * block_size - 0.5
    ax2.axhline(coord, color="gray", linewidth=0.3, alpha=0.5)
    ax2.axvline(coord, color="gray", linewidth=0.3, alpha=0.5)
ax2.set_xticks(ticks - 0.5)
ax2.set_xticklabels(ticks)
ax2.set_yticks(ticks - 0.5)
ax2.set_yticklabels(ticks)
nnz_blocks = int(block_occ.sum())
ax2.set_title(f"{nnz_blocks} nonzero blocks")

# Right: entries + block outlines
ax3.imshow(dense, cmap="Greys", origin="upper", interpolation="nearest")
# Draw outlines around non-zero blocks
from matplotlib.patches import Rectangle
for br in range(n_blocks):
    for bc in range(n_blocks):
        if block_occ[br, bc]:
            rect = Rectangle(
                (bc * block_size - 0.5, br * block_size - 0.5),
                block_size, block_size,
                linewidth=1.2, edgecolor="steelblue", facecolor="lightblue", alpha=0.35,
            )
            ax3.add_patch(rect)
ax3.set_xticks(ticks - 0.5)
ax3.set_xticklabels(ticks)
ax3.set_yticks(ticks - 0.5)
ax3.set_yticklabels(ticks)
block_density = A.nnz / (nnz_blocks * block_size * block_size) * 100
ax3.set_title(f"{block_density:.1f}% block density")

plt.tight_layout()
plt.savefig("plots/spy_plots/block_spy.png", dpi=200)

# --- Second figure: original (scrambled) vs reordered (block-structured) ---
# A already has nice block structure; scramble it with a random perm for the "original"
perm = rng.permutation(N)
A_scrambled = A[perm, :][:, perm]

def draw_block_overlay(ax, mat, N, block_size, n_blocks, ticks):
    """Draw entries + block outlines for a matrix on the given axis."""
    dense = np.zeros((N, N))
    dense[mat.nonzero()] = 1.0
    ax.imshow(dense, cmap="Greys", origin="upper", interpolation="nearest")
    # Compute block occupancy
    blk = np.zeros((n_blocks, n_blocks), dtype=bool)
    coo = mat.tocoo()
    for r, c in zip(coo.row, coo.col):
        blk[r // block_size, c // block_size] = True
    from matplotlib.patches import Rectangle
    for br in range(n_blocks):
        for bc in range(n_blocks):
            if blk[br, bc]:
                rect = Rectangle(
                    (bc * block_size - 0.5, br * block_size - 0.5),
                    block_size, block_size,
                    linewidth=1.2, edgecolor="steelblue", facecolor="lightblue", alpha=0.35,
                )
                ax.add_patch(rect)
    ax.set_xticks(ticks - 0.5)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks - 0.5)
    ax.set_yticklabels(ticks)
    return int(blk.sum())

fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

nnz_orig = draw_block_overlay(ax_l, A_scrambled, N, block_size, n_blocks, ticks)
bd_orig = A_scrambled.nnz / (nnz_orig * block_size * block_size) * 100
ax_l.set_title(f"Original – {nnz_orig} blocks, {bd_orig:.1f}% block density")

nnz_reord = draw_block_overlay(ax_r, A, N, block_size, n_blocks, ticks)
bd_reord = A.nnz / (nnz_reord * block_size * block_size) * 100
ax_r.set_title(f"Reordered – {nnz_reord} blocks, {bd_reord:.1f}% block density")

fig2.tight_layout()
fig2.savefig("plots/spy_plots/block_spy_reorder.png", dpi=200)
plt.show()
