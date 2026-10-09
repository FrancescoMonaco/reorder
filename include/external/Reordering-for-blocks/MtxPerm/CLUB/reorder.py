#!/usr/bin/env python3
"""
CLUB reordering wrapper.

Loads a Matrix Market file, runs one of the C++ CLUB implementations
(see TECHNIQUE below; default "stripe" = bipartite-LP coarse order +
exact stripe-gain refinement), and writes the resulting row permutation
as a 1-based .perm file.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[5]))

from MtxPerm.utils import save_permutation

try:
    from CLUB._reorder_impl import (
        reorder2,
        seed_refine,
        fingerprint,
        # multilevel_mask,
        # micro_macro_mask,
        # recursive_bisection,
        # bipartite_lp,
        # multilevel_fm,
        # stripe_refine,
        # chain,
        # jaccard,
    )
except ImportError as exc:
    print(
        "Error: could not import reorder._reorder_impl. "
        "Build the CMake target before running CLUB/reorder.py.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

# Reordering technique to use. One of:
#   "stripe"      - Track S: bipartite-LP coarse order + exact stripe-gain
#                   swap refinement at each block size in --bs-list (best
#                   quality; default)
#   "chain"       - Block-native sketch sort ("anchor sort"): ONE mask at
#                   W==bs + ONE sort by (rarest-window anchor, salted set
#                   hash). Gray-speed class O(nnz + n log n). Use --bs to
#                   pick the target block size; --max-passes > 0 adds one
#                   stripe_refine polish at the same bs.
#   "jaccard"     - THE original CLUB idea, sub-quadratic: 0-1 sketch ->
#                   capped inverted index -> exact Jaccard -> union-find
#                   clusters (tau) -> stripe-union chain inside clusters;
#                   alternates row/column passes until block cost stops
#                   improving (both sides permuted; --row-only to disable).
#                   Optional stripe_refine polish via --refine-passes.
#   "multilevel-fm" - heavy-edge coarsening + balanced FM (Track P)
#   "bipartite-lp"  - IDF-weighted bipartite LP + strength order (Track R)
#   "recursive"   - k-way hypergraph partitioning via n-level recursive
#                   bisection (FM bisection at each level)
#   "micromacro"  - single-window sketch + micro-macro clustering (BFS on
#                   the macro graph of micro-clusters)
#   "multilevel"  - multi-resolution sketch (list of window sizes) +
#                   lexicographic clustering
#   "classic"     - two-sided alternating reordering (reorder2)
#   "seed-refine" - Seed-and-refine: Mask/ClusterLex seed + bounded exact
#                   LocalRefine (ExactJaccard + greedy chain, commit-if-better),
#                   alternating row/column passes (near-linear, OpenMP).
#                   Use --L/--stride-s/--block-size to tune; --max-iters = T.
TECHNIQUE = "classic"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLUB reordering permutation for sparse matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix_path", help="Path to Matrix Market file")
    parser.add_argument("output_path", help="Path to save permutation file")
    parser.add_argument("--w", type=int, default=4, help="Sketch window size")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=4,
        help="Maximum number of alternating row/column passes",
    )
    parser.add_argument(
        "--technique",
        type=str,
        default=None,
        choices=["stripe", "chain", "jaccard", "multilevel-fm", "bipartite-lp", "recursive", "micromacro", "multilevel", "classic", "seed-refine", "fingerprint"],
        help=(
            "Reordering technique to use. Overrides the module-level "
            "TECHNIQUE variable: 'stripe' (bipartite-LP coarse order + "
            "exact stripe-gain swap refinement, best quality), "
            "'multilevel-fm' (heavy-edge coarsening + "
            "balanced FM), 'bipartite-lp' (IDF-weighted bipartite LP + "
            "strength ordering), 'recursive' (k-way hypergraph partitioning "
            "via n-level recursive bisection), 'micromacro' (single-window "
            "sketch + micro-macro clustering), 'multilevel' (multi-resolution "
            "sketch + lexicographic clustering), 'chain' (block-native sketch "
            "sort, Gray-speed O(nnz + n log n)), 'classic' (two-sided reorder2), "
            "'seed-refine' (Mask/ClusterLex seed + exact LocalRefine, "
            "alternating passes), 'fingerprint' (row-only Bloom archetypes + "
            "leader clustering + stripe_refine polish, no identity guard)"
        ),
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=8,
        help="Stripe/block size for 'chain' (W is forced to bs if --w is default)",
    )
    parser.add_argument(
        "--k-hits",
        type=int,
        default=8,
        help=(
            "Number of MinHash lanes for 'chain' (only used when the window "
            "bitmap does not fit in memory; otherwise the exact window "
            "bitmap is the sort key)"
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Jaccard threshold for union-find clustering in 'jaccard'",
    )
    parser.add_argument(
        "--refine-passes",
        type=int,
        default=0,
        help="stripe_refine polish passes for 'jaccard' (0 = pure clustering)",
    )
    parser.add_argument(
        "--row-only",
        action="store_true",
        help=(
            "For 'jaccard': permute rows only (default alternates rows and "
            "columns until block cost converges; two-line .perm written)"
        ),
    )
    parser.add_argument(
        "--micro-threshold",
        type=int,
        default=0,
        help="Micro-cluster size threshold for 'micromacro' (0 = auto: sqrt(n))",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of groups for 'recursive'/'multilevel-fm' (0 = auto: 8)",
    )
    parser.add_argument(
        "--lp-sweeps",
        type=int,
        default=8,
        help="Label-propagation sweeps for 'bipartite-lp'",
    )
    parser.add_argument(
        "--bs-list",
        type=int,
        nargs="+",
        default=[16, 32],
        help=(
            "Block sizes refined by 'stripe' (each gets its own W==bs sketch "
            "so deltas match the measured metric; finer lists like 4 8 16 32 "
            "improve small-block quality at some cost)"
        ),
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=6,
        help=(
            "Hill-climb passes per block size for 'stripe'. For 'chain' "
            "leave 0 (pure sort, Gray-speed); a small value like 2 adds a "
            "stripe-refine polish at ~2-5x cost for marginal gains"
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Swap window (in neighbouring stripes) per pass for 'stripe'",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=256,
        help="LocalRefine window length (rows) for 'seed-refine'",
    )
    parser.add_argument(
        "--stride-s",
        type=int,
        default=256,
        help="LocalRefine stride (rows) for 'seed-refine' (default = --L, disjoint windows)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Measured block size BxB for 'seed-refine' (block-count objective)",
    )
    parser.add_argument(
        "--k-words",
        type=int,
        default=4,
        help="Bloom words per fingerprint for 'fingerprint' (bits = 64*k)",
    )
    parser.add_argument(
        "--tau-bits",
        type=int,
        default=8,
        help="Min shared Bloom bits to join an archetype in 'fingerprint'",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=256,
        help="Archetype dictionary cap for 'fingerprint' (spillover assigns best)",
    )
    parser.add_argument(
        "--k0",
        type=int,
        default=8,
        help="Seed archetypes for 'fingerprint' (first K0 non-empty rows by mass)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    matrix_path = Path(args.matrix_path).resolve()
    output_path = Path(args.output_path).resolve()
    verbose = not args.quiet

    if not matrix_path.exists():
        print(f"Error: Matrix file not found: {matrix_path}", file=sys.stderr)
        raise SystemExit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    technique = args.technique if args.technique is not None else TECHNIQUE
    col_perm = np.asarray([], dtype=int)

    try:
        if verbose:
            print(f"Loading matrix: {matrix_path}")
        t0 = time.perf_counter()
        _M = scipy.io.mmread(str(matrix_path))
        matrix = _M.tocsr() if hasattr(_M, "tocsr") else scipy.sparse.coo_matrix(_M).tocsr()
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"<Timer>[loading] {load_ms:.6f} ms")

        if verbose:
            print(f"Running CLUB reordering (technique={technique})...")
        t0 = time.perf_counter()
        if technique == "classic":
            perm = reorder2(str(matrix_path), args.w, args.max_iters)
        elif technique == "multilevel":
            perm = multilevel_mask(str(matrix_path), [32, 16, 4, 1])
        elif technique == "micromacro":
            perm = micro_macro_mask(str(matrix_path), args.w, args.micro_threshold)
        elif technique == "recursive":
            perm = recursive_bisection(str(matrix_path), args.w, args.k)
        elif technique == "bipartite-lp":
            perm = bipartite_lp(str(matrix_path), args.w, args.micro_threshold, args.lp_sweeps)
        elif technique == "multilevel-fm":
            perm = multilevel_fm(str(matrix_path), args.w, args.k, args.micro_threshold)
        elif technique == "stripe":
            perm = stripe_refine(
                str(matrix_path), args.w, args.bs_list, args.max_passes, args.window
            )
        elif technique == "chain":
            if args.bs <= 0:
                raise ValueError("--bs must be a positive block size for 'chain'")
            w = args.w if "--w" in sys.argv else args.bs
            perm = chain(
                str(matrix_path), w, args.bs, args.k_hits, args.max_passes, args.window
            )
        elif technique == "jaccard":
            if args.bs <= 0:
                raise ValueError("--bs must be a positive block size for 'jaccard'")
            w = args.w if "--w" in sys.argv else args.bs
            P, Q = jaccard(
                str(matrix_path), w, args.bs, args.k_hits, args.tau,
                args.max_iters, args.refine_passes, not args.row_only, [],
            )
            perm = np.asarray(P)
            col_perm = np.asarray(Q)
        elif technique == "seed-refine":
            P, Q, blocks = seed_refine(
                str(matrix_path), args.w, args.L, args.stride_s, args.max_iters, args.block_size
            )
            perm = np.asarray(P)
            col_perm = np.asarray(Q)
            if verbose:
                print(f"seed-refine best {args.block_size}x{args.block_size} blocks: {blocks}")
        elif technique == "fingerprint":
            # Row-only Bloom archetypes: mask(W) -> fingerprints -> leader
            # clustering (overlap>=tau joins, else new archetype) -> chain ->
            # stripe_refine at bs. No identity guard: raw algorithm output.
            bs = args.bs if args.bs > 0 else 16
            w = args.w if "--w" in sys.argv else (bs // 2 if bs >= 8 else 4)
            perm = np.asarray(
                fingerprint(
                    str(matrix_path), w, args.k_words, args.tau_bits,
                    args.kmax, args.k0, bs, args.max_passes, args.window,
                )
            )
        else:
            raise ValueError(f"Unknown technique: {technique!r}")
        perm = np.asarray(perm)
        reorder_ms = (time.perf_counter() - t0) * 1000
        print(f"<Timer>[reordering] {reorder_ms:.6f} ms")

        if len(perm) != matrix.shape[0]:
            raise RuntimeError(
                f"Permutation length ({len(perm)}) does not match matrix rows ({matrix.shape[0]})"
            )

        if len(col_perm) == matrix.shape[1] and len(col_perm) > 0:
            # Two-line asymmetric perm (row perm line 1, col perm line 2),
            # 1-based. Lines are written independently because row and column
            # counts can differ on rectangular matrices (np.vstack would fail).
            perm_1b = " ".join(map(str, (np.asarray(perm) + 1).tolist()))
            col_1b = " ".join(map(str, (np.asarray(col_perm) + 1).tolist()))
            with open(output_path, "w") as f:
                f.write(perm_1b + "\n")
                f.write(col_1b + "\n")
        else:
            save_permutation(output_path, perm)

        if verbose:
            print(f"Permutation saved to {output_path}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()