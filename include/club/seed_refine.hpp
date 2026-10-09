#pragma once
// seed_refine.hpp — Seed-and-refine reordering (Alg. 1 + Alg. 2).
//
// Seed:  Mask(A, W) -> ClusterLex -> Permute  (existing club::mask/cluster_lex/permute)
// Refine: bounded exact-resolution local refinement over row windows [start, end)
//         of length L, stride s (default s == L => disjoint, fully parallel).
//         Per window: ExactJaccard on TRUE column sets -> GreedyChain ->
//         commit only if strictly better on LocalBlockCount.
// Outer driver alternates row / column passes (column pass via transpose,
// like reorder2) for at most T iterations, keeping the best-seen state and
// stopping on no strict improvement.
//
// Complexity (n rows, L constant, d_avg avg true row degree, mhat = nnz(Ahat)):
//   mask:          O(nnz)            (existing, OpenMP)
//   cluster_lex:   O(mhat log l)     (existing, OpenMP)
//   local_refine:  O((n/L) * L^2 * d_avg) = O(L * nnz) per sweep (OpenMP over
//                  disjoint windows; each window does an LxL exact Jaccard
//                  merge-join + O(L^2) chain + O(L*d) block-count compare).
//   driver:        O(T * (nnz + mhat log l + L*nnz)) — near-linear for
//                  constant W, L, T. Same class as RCM / rabbit / PaToH
//                  coarsening passes; no O(n^2) row-pair step.
//
// Includes reorder.hpp (one-directional, same pattern as chain/jaccard/refine).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "club/reorder.hpp"

namespace club {
    namespace seed {

        struct SeedStats {
            size_t iters_used = 0;          // row+col passes actually executed
            size_t best_blocks = 0;         // best block count seen (at BxB)
            size_t n = 0;                   // rows
            std::vector<size_t> cost_trace; // block count after each full pass
        };

        // ----------------------------------------------------------------
        // exact_jaccard_rows(A, r1, r2): Jaccard distance on TRUE column sets.
        // Rows are sorted (read_from_mtx invariant) so merge-join is exact.
        // O(d1 + d2). Deterministic, no allocation.
        // ----------------------------------------------------------------
        template <typename DataT, typename intT>
        inline double exact_jaccard_rows( const CSR<DataT, intT>& A, size_t r1, size_t r2 ) {
            const size_t s1 = static_cast<size_t>( A.row_ptr[r1] );
            const size_t e1 = static_cast<size_t>( A.row_ptr[r1 + 1] );
            const size_t s2 = static_cast<size_t>( A.row_ptr[r2] );
            const size_t e2 = static_cast<size_t>( A.row_ptr[r2 + 1] );
            const size_t n1 = e1 - s1, n2 = e2 - s2;
            if ( n1 == 0 && n2 == 0 )
                return 0.0;
            if ( n1 == 0 || n2 == 0 )
                return 1.0;
            size_t isect = 0, p1 = s1, p2 = s2;
            while ( p1 < e1 && p2 < e2 ) {
                const auto c1 = A.col_ind[p1], c2 = A.col_ind[p2];
                if ( c1 == c2 ) {
                    ++isect;
                    ++p1;
                    ++p2;
                } else if ( c1 < c2 ) {
                    ++p1;
                } else {
                    ++p2;
                }
            }
            return 1.0 - static_cast<double>( isect ) / static_cast<double>( n1 + n2 - isect );
        }

        // ----------------------------------------------------------------
        // greedy_chain(R, D): nearest-neighbour chain over the LxL distance
        // matrix D (row-major, D[i*L+j] = dist(R[i], R[j])).
        // Start = row with min total distance (centrality proxy), min row-id
        // tie-break; each step picks the nearest unvisited row (tie: min id).
        // Returns the chain as global row ids in new order. O(L^2), serial
        // per window (L constant, no need for inner parallelism).
        // ----------------------------------------------------------------
        inline std::vector<size_t> greedy_chain( const std::vector<size_t>& R,
                                                 const std::vector<double>& D ) {
            const size_t L = R.size();
            std::vector<size_t> chain;
            chain.reserve( L );
            if ( L == 0 )
                return chain;
            if ( L == 1 )
                return R;
            std::vector<char> used( L, 0 );
            size_t cur = 0;
            double best_tot = std::numeric_limits<double>::infinity();
            for ( size_t i = 0; i < L; ++i ) {
                double tot = 0.0;
                for ( size_t j = 0; j < L; ++j )
                    tot += D[i * L + j];
                if ( tot < best_tot - 1e-12 ||
                     ( std::abs( tot - best_tot ) <= 1e-12 && R[i] < R[cur] ) ) {
                    best_tot = tot;
                    cur = i;
                }
            }
            chain.push_back( R[cur] );
            used[cur] = 1;
            for ( size_t step = 1; step < L; ++step ) {
                size_t nxt = L;
                double nxt_d = std::numeric_limits<double>::infinity();
                for ( size_t j = 0; j < L; ++j ) {
                    if ( used[j] )
                        continue;
                    const double d = D[cur * L + j];
                    if ( nxt == L || d < nxt_d - 1e-12 ||
                         ( std::abs( d - nxt_d ) <= 1e-12 && R[j] < R[nxt] ) ) {
                        nxt_d = d;
                        nxt = j;
                    }
                }
                assert( nxt < L );
                used[nxt] = 1;
                chain.push_back( R[nxt] );
                cur = nxt;
            }
            return chain;
        }

        // ----------------------------------------------------------------
        // local_block_count(A, start, order, B): window-local block cost.
        // `order` holds the global row ids covering positions
        // [start, start+order.size()) in the candidate order.
        // Cost = sum over block-rows (br = pos / B) of distinct block-columns
        // (col / B) touched. Matches count_nonzero_blocks(A, B, B) restricted
        // to the window; exact when the window is B-aligned (true for
        // defaults L=256, B=32, disjoint windows with start % B == 0).
        // O(sum of degrees in window) via per-stripe sort+unique.
        // ----------------------------------------------------------------
        template <typename DataT, typename intT>
        inline size_t local_block_count( const CSR<DataT, intT>& A,
                                         size_t start,
                                         const std::vector<size_t>& order,
                                         size_t B ) {
            const size_t L = order.size();
            if ( L == 0 || B == 0 )
                return 0;
            size_t cost = 0;
            size_t p = 0;
            std::vector<size_t> bcols;
            while ( p < L ) {
                const size_t br = ( start + p ) / B;
                size_t q = p;
                while ( q < L && ( start + q ) / B == br )
                    ++q;
                bcols.clear();
                for ( size_t t = p; t < q; ++t ) {
                    const size_t r = order[t];
                    const size_t s = static_cast<size_t>( A.row_ptr[r] );
                    const size_t e = static_cast<size_t>( A.row_ptr[r + 1] );
                    for ( size_t k = s; k < e; ++k )
                        bcols.push_back( static_cast<size_t>( A.col_ind[k] ) / B );
                }
                std::sort( bcols.begin(), bcols.end() );
                bcols.erase( std::unique( bcols.begin(), bcols.end() ), bcols.end() );
                cost += bcols.size();
                p = q;
            }
            return cost;
        }

        // ----------------------------------------------------------------
        // apply_window_order(A, start, order): rewrite rows
        // [start, start+L) of A in `order` (a permutation of those ids).
        // The window's total nnz is preserved, so row_ptr entries after the
        // window keep their offsets; only this window's row_ptr/nzcount and
        // its col_ind/values span change. Disjoint windows are race-free.
        // Caller guarantees disjointness (s == L path).
        // ----------------------------------------------------------------
        template <typename DataT, typename intT>
        inline void
        apply_window_order( CSR<DataT, intT>& A, size_t start, const std::vector<size_t>& order ) {
            const size_t L = order.size();
            if ( L <= 1 )
                return;
            std::vector<size_t> lens( L );
            size_t span = 0;
            for ( size_t t = 0; t < L; ++t ) {
                const size_t r = order[t];
                lens[t] = static_cast<size_t>( A.row_ptr[r + 1] - A.row_ptr[r] );
                span += lens[t];
            }
            const size_t base = static_cast<size_t>( A.row_ptr[start] );
            std::vector<intT> tmp_cols( span );
            std::vector<DataT> tmp_vals;
            if ( !A.pattern_only )
                tmp_vals.resize( span );
            size_t dst = 0;
            for ( size_t t = 0; t < L; ++t ) {
                const size_t r = order[t];
                const size_t s = static_cast<size_t>( A.row_ptr[r] );
                std::copy_n( A.col_ind.begin() + static_cast<ptrdiff_t>( s ),
                             lens[t],
                             tmp_cols.begin() + static_cast<ptrdiff_t>( dst ) );
                if ( !A.pattern_only )
                    std::copy_n( A.values.begin() + static_cast<ptrdiff_t>( s ),
                                 lens[t],
                                 tmp_vals.begin() + static_cast<ptrdiff_t>( dst ) );
                dst += lens[t];
            }
            for ( size_t t = 0; t < L; ++t ) {
                A.nzcount[start + t] = static_cast<intT>( lens[t] );
                A.row_ptr[start + t + 1] = A.row_ptr[start + t] + static_cast<intT>( lens[t] );
            }
            std::copy( tmp_cols.begin(),
                       tmp_cols.end(),
                       A.col_ind.begin() + static_cast<ptrdiff_t>( base ) );
            if ( !A.pattern_only )
                std::copy( tmp_vals.begin(),
                           tmp_vals.end(),
                           A.values.begin() + static_cast<ptrdiff_t>( base ) );
        }

        // ----------------------------------------------------------------
        // local_refine(A, L, s, B, delta_out): Alg. 2 LocalRefine.
        // For each window R = [start, min(start+L, n)):
        //   D  = exact Jaccard on TRUE sets (no windowing)
        //   pi = greedy nearest-neighbour chain
        //   commit pi only if LocalBlockCount(pi) < LocalBlockCount(id).
        // s == L (disjoint, default): OpenMP parallel-for, race-free.
        // s < L (overlap): sequential fallback so commits cannot race.
        // delta_out (optional): identity perm updated with every committed
        // window move, letting the outer driver keep pi_r / pi_c exact.
        // ----------------------------------------------------------------
        template <typename DataT, typename intT>
        inline void local_refine( CSR<DataT, intT>& A,
                                  size_t L,
                                  size_t s,
                                  size_t B,
                                  std::vector<size_t>* delta_out = nullptr ) {
            const size_t n = static_cast<size_t>( A.rows );
            if ( n == 0 )
                return;
            if ( L == 0 )
                L = 256;
            if ( s == 0 || s > L )
                s = L;
            if ( B == 0 )
                B = 32;

            std::vector<size_t> delta;
            if ( delta_out ) {
                delta.resize( n );
                std::iota( delta.begin(), delta.end(), 0 );
            }

            auto process_window = [&]( size_t start ) {
                const size_t end = std::min( start + L, n );
                const size_t wlen = end - start;
                if ( wlen <= 1 )
                    return;
                bool any = false;
                for ( size_t r = start; r < end; ++r ) {
                    if ( A.row_ptr[r + 1] != A.row_ptr[r] ) {
                        any = true;
                        break;
                    }
                }
                if ( !any )
                    return;
                std::vector<size_t> R( wlen );
                std::iota( R.begin(), R.end(), start );
                std::vector<double> D( wlen * wlen, 0.0 );
                for ( size_t i = 0; i < wlen; ++i ) {
                    for ( size_t j = i + 1; j < wlen; ++j ) {
                        const double d = exact_jaccard_rows( A, R[i], R[j] );
                        D[i * wlen + j] = d;
                        D[j * wlen + i] = d;
                    }
                }
                std::vector<size_t> pi = greedy_chain( R, D );
                const size_t c_id = local_block_count( A, start, R, B );
                const size_t c_pi = local_block_count( A, start, pi, B );
                if ( c_pi < c_id ) {
                    apply_window_order( A, start, pi );
                    if ( delta_out ) {
                        for ( size_t t = 0; t < wlen; ++t )
                            delta[start + t] = pi[t];
                    }
                }
            };

            if ( s == L ) {
#pragma omp parallel for schedule( dynamic )
                for ( size_t start = 0; start < n; start += s )
                    process_window( start );
            } else {
                for ( size_t start = 0; start < n; start += s )
                    process_window( start );
            }

            if ( delta_out )
                *delta_out = std::move( delta );
        }

        // compose_perm(cum, P): cumulative permutation update.
        // permute(A, P) moves old row P[i] to new position i, so
        // cum_new[i] = cum_old[P[i]].
        inline std::vector<size_t> compose_perm( const std::vector<size_t>& cum,
                                                 const std::vector<size_t>& P ) {
            std::vector<size_t> out( cum.size() );
            for ( size_t i = 0; i < P.size(); ++i )
                out[i] = cum[P[i]];
            return out;
        }

        // ----------------------------------------------------------------
        // seed_refine(A, W, L, s, T, B, pi_r, pi_c, stats): Alg. 1.
        // Row seed (Mask/ClusterLex/Permute/LocalRefine), then up to T full
        // column+row passes with best-state rollback and early stop on no
        // strict improvement. pi_r / pi_c map new position -> original index
        // and track BOTH ClusterLex moves and LocalRefine window commits.
        // ----------------------------------------------------------------
        template <typename DataT, typename intT>
        inline void seed_refine( CSR<DataT, intT>& A,
                                 size_t W,
                                 size_t L,
                                 size_t s,
                                 size_t T,
                                 size_t B,
                                 std::vector<size_t>& pi_r,
                                 std::vector<size_t>& pi_c,
                                 SeedStats* stats = nullptr ) {
            const size_t n = static_cast<size_t>( A.rows );
            const size_t m = static_cast<size_t>( A.cols );
            if ( W == 0 )
                W = 4;
            if ( L == 0 )
                L = 256;
            if ( s == 0 || s > L )
                s = L;
            if ( B == 0 )
                B = 32;

            pi_r.resize( n );
            std::iota( pi_r.begin(), pi_r.end(), 0 );
            pi_c.resize( m );
            std::iota( pi_c.begin(), pi_c.end(), 0 );
            if ( n == 0 )
                return;

            SeedStats st;
            st.n = n;

            // Baseline BEFORE any move: seed counts as candidate pass 0, so a
            // regressing seed rolls back to natural order (never worse than
            // baseline — same guarantee as reorder2/apply_if_better).
            size_t best_blocks = count_nonzero_blocks( A, B, B );
            CSR<DataT, intT> best = A;
            std::vector<size_t> best_r = pi_r, best_c = pi_c;
            st.best_blocks = best_blocks;
            st.cost_trace.push_back( best_blocks );

            // ---- initial row seed ----
            {
                CSR<size_t, size_t> Ahat;
                std::vector<size_t> P;
                mask( A, W, Ahat );
                cluster_lex( Ahat, P );
                pi_r = compose_perm( pi_r, P );
                permute( A, P );
                std::vector<size_t> d;
                local_refine( A, L, s, B, &d );
                pi_r = compose_perm( pi_r, d );
            }

            {
                const size_t seeded = count_nonzero_blocks( A, B, B );
                st.cost_trace.push_back( seeded );
                LOG_INFO( "msg", "seed-refine: seeded", "nonzero blocks", seeded );
                if ( seeded < best_blocks ) {
                    best_blocks = seeded;
                    best = A;
                    best_r = pi_r;
                    best_c = pi_c;
                    st.best_blocks = best_blocks;
                } else {
                    // Seed regressed: restore natural order, keep identity perms.
                    A = best;
                    pi_r = best_r;
                    pi_c = best_c;
                }
            }

            for ( size_t t = 0; t < T; ++t ) {
                // ---- column pass, via transpose ----
                {
                    CSR<DataT, intT> AT;
                    transpose( A, AT );
                    CSR<size_t, size_t> AhatT;
                    std::vector<size_t> Q;
                    mask( AT, W, AhatT );
                    cluster_lex( AhatT, Q );
                    pi_c = compose_perm( pi_c, Q );
                    permute( AT, Q );
                    std::vector<size_t> d;
                    local_refine( AT, L, s, B, &d );
                    pi_c = compose_perm( pi_c, d );
                    transpose( AT, A );
                }
                // ---- row pass ----
                {
                    CSR<size_t, size_t> Ahat;
                    std::vector<size_t> P;
                    mask( A, W, Ahat );
                    cluster_lex( Ahat, P );
                    pi_r = compose_perm( pi_r, P );
                    permute( A, P );
                    std::vector<size_t> d;
                    local_refine( A, L, s, B, &d );
                    pi_r = compose_perm( pi_r, d );
                }
                const size_t blocks = count_nonzero_blocks( A, B, B );
                st.cost_trace.push_back( blocks );
                LOG_INFO( "msg", "seed-refine: pass done", "t", t, "nonzero blocks", blocks );
                if ( blocks < best_blocks ) {
                    best_blocks = blocks;
                    best = A;
                    best_r = pi_r;
                    best_c = pi_c;
                    st.best_blocks = best_blocks;
                } else {
                    break;
                }
                st.iters_used = t + 1;
            }

            A = std::move( best );
            pi_r = std::move( best_r );
            pi_c = std::move( best_c );
            if ( stats )
                *stats = std::move( st );
            LOG_INFO( "msg", "seed-refine: done", "nonzero blocks", best_blocks );
        }

    } // namespace seed
} // namespace club
