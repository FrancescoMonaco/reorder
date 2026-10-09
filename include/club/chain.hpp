#pragma once
// chain.hpp — block-native sketch sort ("anchor sort") for row reordering.
//
// Goal: order rows so rows within a bs-neighbourhood share windows, i.e.
// minimise  cost(P) = sum over stripes of |union of windows|
// (== number of nonzero bs x bs blocks, the measured blocking metric).
//
// Design (Gray-speed class, block-native key):
//   1. ONE mask() at W == bs (block-native sketch, OpenMP-parallel).
//   2. Per row, one anchor: rarest window a(r) = argmin df(w) — a 1-sample
//      MinHash-style key used as a SORT key, not an LSH index. Rows sharing
//      an anchor sort together the way TCA's LSH buckets group rows,
//      except assignment is by sort (no candidate pairs, no exact Jaccard,
//      no priority queue, no GPU, deterministic).
//   3. Secondary key: cheap salted FNV-style hash of the window set
//      (same "similar sets share order" property as MinHash bands, without
//      storing signatures or querying an index).
//   4. ONE std::sort by (df(anchor) rank, h1, h2). Optional single
//      stripe_refine polish at the same bs.
//
// Complexity (mhat = nnz(Ahat), d = avg sketch degree):
//   mask:  O(nnz) parallel (existing)
//   keys:  O(mhat) parallel
//   sort:  O(n log n)
// Total: O(nnz + n log n) — same class as Gray code, ~5-10x Gray wall time.
// No O(n^2) row pairs, no PQ, no second LSH level, no 6-mask guard.
//
// Novelty vs prior art in this tree:
//   vs TCA/DTC-LSH: MinHash used as ordering, not as candidate-pair index;
//     no exact-Jaccard verification, no global PQ merge, CPU-only.
//   vs Gray: key is block-native (W==bs windows = blocking objective), no
//     dense/sparse split, no alternating asc/desc.
//   vs SPARTA: no group formation, just a sort.
//   vs cluster_lex: no exact-prefix requirement (bridging rows share the
//     anchor key instead of shattering into singletons).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "club/matrices.hpp"
#include "club/refine.hpp"

namespace club {
    namespace chain {

        struct ChainStats {
            size_t n = 0;
            size_t nwin = 0;
            size_t nnz_hat = 0;
            size_t n_empty = 0;    // rows with empty sketch (sorted first, stable)
            double build_ms = 0.0; // key-build phase
            double chain_ms = 0.0; // sort phase
        };

        // anchor_rows: block-native sketch sort, two modes (Gray-speed class).
        //
        // Mode A (exact): when the full window bitmap fits in the memory budget
        // (nwin small, typical: nwin = ceil(ncols/W) for W == bs), the sort key IS
        // the row's window set — lexicographic bitmap sort, i.e. Gray code's key
        // generalized to the blocking objective (W==bs windows) with no chunking
        // loss and no asc/desc zigzag. Rows that share windows sort adjacently.
        //
        // Mode B (fallback, huge nwin): k-lane MinHash signature (k 64-bit lanes,
        // min over the row's windows of splitmix64(w, lane_seed)), sorted
        // lexicographically. Signature proximity ~ Jaccard, so similar rows sort
        // near each other — MinHash used as an ORDERING key, not an LSH index.
        //
        // Empty rows get all-zero keys and sort first. Output P: position -> row.
        // Cost: O(mhat) keys (OpenMP) + O(n log n) compares. Deterministic.
        inline void anchor_rows( const CSR<size_t, size_t>& Ahat,
                                 std::vector<size_t>& P,
                                 size_t lanes = 8,
                                 ChainStats* st = nullptr ) {
            auto t0 = std::chrono::steady_clock::now();
            const size_t n = Ahat.rows, nwin = Ahat.cols;
            P.resize( n );
            if ( n == 0 )
                return;
            if ( lanes == 0 )
                lanes = 8;

            constexpr double BITMAP_BUDGET_MB = 512.0;
            const size_t w64 = ( nwin + 63 ) / 64;
            const bool exact =
                ( nwin > 0 ) &&
                ( ( double( n ) * double( w64 ) * 8.0 / ( 1024.0 * 1024.0 ) ) <= BITMAP_BUDGET_MB );
            const size_t stride = exact ? w64 : lanes;

            std::vector<uint64_t> sigs( n * stride, 0 );
#pragma omp parallel for schedule( static )
            for ( size_t i = 0; i < n; ++i ) {
                const size_t s = Ahat.row_ptr[i], e = Ahat.row_ptr[i + 1];
                if ( s == e )
                    continue; // empty row: key stays all-zero, sorts first
                uint64_t* key = &sigs[i * stride];
                if ( exact ) {
                    for ( size_t k = s; k < e; ++k ) {
                        const size_t w = Ahat.col_ind[k];
                        if ( w < nwin )
                            key[w >> 6] |= 1ULL << ( w & 63 );
                    }
                } else {
                    for ( size_t j = 0; j < lanes; ++j ) {
                        uint64_t mn = ~0ULL;
                        for ( size_t k = s; k < e; ++k ) {
                            // splitmix64(w + lane constant): universal-ish family
                            uint64_t z = static_cast<uint64_t>( Ahat.col_ind[k] ) +
                                         0x9E3779B97F4A7C15ULL * ( j + 1 );
                            z ^= z >> 30;
                            z *= 0xBF58476D1CE4E5B9ULL;
                            z ^= z >> 27;
                            z *= 0x94D049BB133111EBULL;
                            z ^= z >> 31;
                            if ( z < mn )
                                mn = z;
                        }
                        key[j] = mn;
                    }
                }
            }

            auto lex_less = [&]( size_t x, size_t y ) {
                const uint64_t* a = &sigs[x * stride];
                const uint64_t* b = &sigs[y * stride];
                for ( size_t j = 0; j < stride; ++j )
                    if ( a[j] != b[j] )
                        return a[j] < b[j];
                return x < y;
            };
            std::vector<size_t> idx( n );
            std::iota( idx.begin(), idx.end(), 0 );
            std::sort( idx.begin(), idx.end(), lex_less );
            P.assign( idx.begin(), idx.end() );
            auto t1 = std::chrono::steady_clock::now();
            if ( st ) {
                st->n = n;
                st->nwin = nwin;
                st->nnz_hat = Ahat.row_ptr[n];
                size_t ne = 0;
                for ( size_t i = 0; i < n; ++i )
                    if ( Ahat.row_ptr[i + 1] == Ahat.row_ptr[i] )
                        ++ne;
                st->n_empty = ne;
                using ms = std::chrono::duration<double, std::milli>;
                st->build_ms += ms( t1 - t0 ).count();
                st->chain_ms += 0.0;
            }
        }

    } // namespace chain
} // namespace club

// ---- Legacy serial greedy chain (superseded by anchor_rows above).
// Kept for reference/ablation only; NOT on the default path: it is serial
// O(n * |U| * K * d) with per-step scoring, ~50x slower than the sort
// (30-80 ms vs <2 ms on 3elt-class). Do not call from chain_reorder.
namespace club {
    namespace chain {
        namespace legacy {
            struct SketchIndex {
                std::vector<std::vector<size_t>> inv; // inv[w]: row ids, mass-desc, top-K
                std::vector<double> idf;              // per-window IDF weight
                std::vector<double> mass;             // per-row IDF mass (seed order key)
                std::vector<size_t> mass_order;       // rows by decreasing mass
                size_t K = 8;

                void
                build( const CSR<size_t, size_t>& Ahat, size_t k_hits, ChainStats* st = nullptr ) {
                    auto t0 = std::chrono::steady_clock::now();
                    const size_t n = Ahat.rows, nwin = Ahat.cols;
                    K = ( k_hits == 0 ) ? 8 : k_hits;

                    std::vector<size_t> df( nwin, 0 );
                    for ( size_t i = 0; i < n; ++i )
                        for ( size_t k = Ahat.row_ptr[i]; k < Ahat.row_ptr[i + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( w < nwin )
                                ++df[w];
                        }
                    idf.assign( nwin, 0.0 );
                    for ( size_t w = 0; w < nwin; ++w )
                        if ( df[w] > 0 )
                            idf[w] = std::log1p( static_cast<double>( n ) /
                                                 static_cast<double>( df[w] ) );
                    mass.assign( n, 0.0 );
                    for ( size_t i = 0; i < n; ++i ) {
                        double m = 0.0;
                        for ( size_t k = Ahat.row_ptr[i]; k < Ahat.row_ptr[i + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( w < nwin )
                                m += idf[w];
                        }
                        mass[i] = m;
                    }
                    mass_order.resize( n );
                    std::iota( mass_order.begin(), mass_order.end(), 0 );
                    std::sort( mass_order.begin(), mass_order.end(), [&]( size_t a, size_t b ) {
                        if ( mass[a] != mass[b] )
                            return mass[a] > mass[b];
                        return a < b;
                    } );

                    inv.assign( nwin, {} );
                    for ( size_t i = 0; i < n; ++i )
                        for ( size_t k = Ahat.row_ptr[i]; k < Ahat.row_ptr[i + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( w < nwin )
                                inv[w].push_back( i );
                        }
                    // Per-window truncate to top-K by mass (nth_element, not full sort:
                    // popular windows can have df ~ n).
#pragma omp parallel for schedule( dynamic, 64 )
                    for ( size_t w = 0; w < nwin; ++w ) {
                        auto& v = inv[w];
                        if ( v.size() > K ) {
                            std::nth_element(
                                v.begin(), v.begin() + K, v.end(), [&]( size_t a, size_t b ) {
                                    return mass[a] > mass[b];
                                } );
                            v.resize( K );
                        }
                        std::sort( v.begin(), v.end(), [&]( size_t a, size_t b ) {
                            return mass[a] > mass[b];
                        } );
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    if ( st ) {
                        st->n = n;
                        st->nwin = nwin;
                        st->nnz_hat = Ahat.row_ptr[n];
                        st->build_ms +=
                            std::chrono::duration<double, std::milli>( t1 - t0 ).count();
                    }
                }
            };

            // chain_rows: greedy stripe-union chain over a window sketch.
            //
            // P layout: position -> row. Seed = max-mass row. Maintains the union of
            // windows in the current stripe (reset every bs rows); at each step scores
            // only the top-K unplaced rows of each union window's posting list by
            // IDF overlap with the union and takes the best (ties -> smaller row id).
            // Steps with no window-sharing candidate jump to the next max-mass
            // unplaced row (counts as a restart in stats->n_jumps).
            //
            // Epoch-stamped `seen` vector dedups candidates without clearing: O(1) per
            // probe, O(n) memory. No O(n^2) structure anywhere.
            inline void chain_rows( const CSR<size_t, size_t>& Ahat,
                                    const SketchIndex& idx,
                                    std::vector<size_t>& P,
                                    size_t bs,
                                    ChainStats* st = nullptr ) {
                auto t0 = std::chrono::steady_clock::now();
                const size_t n = Ahat.rows, nwin = Ahat.cols;
                if ( bs == 0 )
                    bs = 16;
                P.assign( n, 0 );
                if ( n == 0 )
                    return;
                std::vector<char> placed( n, 0 );
                std::vector<int> seen( n, 0 );
                int epoch = 1;
                std::vector<size_t> win_cnt( nwin, 0 );
                std::vector<size_t> win_epoch( nwin, 0 );
                int wepoch = 1;
                std::vector<size_t> union_list;
                union_list.reserve( bs * 8 );

                size_t mi = 0;
                while ( mi < n && placed[idx.mass_order[mi]] )
                    ++mi;
                P[0] = ( mi < n ) ? idx.mass_order[mi] : 0;
                placed[P[0]] = 1;
                if ( mi < n )
                    ++mi;

                size_t stripe_start = 0;
                size_t jumps = 0;
                const size_t K = idx.K;

                auto union_add = [&]( size_t w ) {
                    if ( w >= nwin )
                        return;
                    if ( win_epoch[w] != (size_t)wepoch ) {
                        win_epoch[w] = (size_t)wepoch;
                        win_cnt[w] = 1;
                        union_list.push_back( w );
                    } else {
                        ++win_cnt[w];
                    }
                };
                // Seed stripe union with row 0's windows.
                for ( size_t k = Ahat.row_ptr[P[0]]; k < Ahat.row_ptr[P[0] + 1]; ++k )
                    union_add( Ahat.col_ind[k] );

                for ( size_t pos = 1; pos < n; ++pos ) {
                    if ( pos - stripe_start >= bs ) {
                        stripe_start = pos;
                        ++wepoch;
                        if ( wepoch == std::numeric_limits<int>::max() ) {
                            std::fill( win_epoch.begin(), win_epoch.end(), 0 );
                            wepoch = 1;
                        }
                        union_list.clear();
                    }
                    size_t prev = P[pos - 1];
                    for ( size_t k = Ahat.row_ptr[prev]; k < Ahat.row_ptr[prev + 1]; ++k )
                        union_add( Ahat.col_ind[k] );

                    ++epoch;
                    if ( epoch == std::numeric_limits<int>::max() ) {
                        std::fill( seen.begin(), seen.end(), 0 );
                        epoch = 1;
                    }
                    size_t best = n;
                    double best_s = -1.0;
                    for ( size_t w : union_list ) {
                        if ( w >= nwin || w >= idx.inv.size() )
                            continue;
                        size_t taken = 0;
                        for ( size_t c : idx.inv[w] ) {
                            if ( c >= n || placed[c] || seen[c] == epoch )
                                continue;
                            seen[c] = epoch;
                            if ( taken++ >= K )
                                break;
                            double s = 0.0;
                            for ( size_t k = Ahat.row_ptr[c]; k < Ahat.row_ptr[c + 1]; ++k ) {
                                size_t w2 = Ahat.col_ind[k];
                                if ( w2 < nwin && win_epoch[w2] == (size_t)wepoch )
                                    s += idx.idf[w2];
                            }
                            if ( s > best_s + 1e-12 || ( s == best_s && c < best ) ) {
                                best_s = s;
                                best = c;
                            }
                        }
                    }
                    if ( best == n ) {
                        ++jumps;
                        while ( mi < n && placed[idx.mass_order[mi]] )
                            ++mi;
                        best = ( mi < n ) ? idx.mass_order[mi++] : 0;
                        // Defensive: linear scan if mass order exhausted (cannot happen
                        // unless n == 0, but keeps the permutation valid regardless).
                        if ( placed[best] ) {
                            for ( size_t r = 0; r < n; ++r )
                                if ( !placed[r] ) {
                                    best = r;
                                    break;
                                }
                        }
                    }
                    P[pos] = best;
                    placed[best] = 1;
                }
                auto t1 = std::chrono::steady_clock::now();
                if ( st ) {
                    st->chain_ms += std::chrono::duration<double, std::milli>( t1 - t0 ).count();
                }
            }

        } // namespace legacy
    } // namespace chain
} // namespace club

// chain_reorder: anchor-sort path used by wrapper.cpp / reorder.py.
// ONE mask at W==bs (block-native), ONE anchor_rows sort, then up to
// max_passes stripe_refine polish passes at the same bs. Semantics of
// max_passes (--passes in bench_anchor.py / reorder.py --max-passes):
//   0 -> pure anchor sort, no refinement (cheap default for A/B timing);
//   k -> run k successive stripe-gain swap passes (each pass is O(stripe
//        window) and stops early when a pass makes no improving swap).
// No 6-mask guard (that cost more than the reorder itself); the caller
// (bench / driver) scores quality.
// k_hits is repurposed as the number of MinHash lanes for the large-nwin
// fallback (exact bitmap mode is used when nwin is small).
namespace club {
    namespace chain {

        // stripe_cost: sum over consecutive bs-stripe ranges of |union of windows|
        // — the exact nonzero bs x bs block count when W == bs. O(mhat + n), cheap
        // guard metric used to decide between candidate permutations.
        inline size_t
        stripe_cost( const CSR<size_t, size_t>& Ahat, size_t bs, const std::vector<size_t>& P ) {
            const size_t n = Ahat.rows, nwin = Ahat.cols;
            if ( n == 0 || P.size() != n )
                return 0;
            std::vector<size_t> epoch( nwin, 0 );
            size_t ep = 0, cost = 0;
            for ( size_t s = 0; s < n; s += bs ) {
                if ( ++ep == 0 ) {
                    std::fill( epoch.begin(), epoch.end(), 0 );
                    ep = 1;
                }
                const size_t end = std::min( s + bs, n );
                for ( size_t pos = s; pos < end; ++pos ) {
                    const size_t r = P[pos];
                    for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k ) {
                        const size_t w = Ahat.col_ind[k];
                        if ( w < nwin && epoch[w] != ep ) {
                            epoch[w] = ep;
                            ++cost;
                        }
                    }
                }
            }
            return cost;
        }

        // true_cost: exact number of nonzero gbs x gbs blocks of the permuted
        // matrix, computed on the sketch (windows are W columns wide, so the column
        // block index at scale gbs is w*W/gbs — exact when gbs divides W).
        // Unlike stripe_cost (window unions), this IS the measured blocking metric.
        inline size_t true_cost( const CSR<size_t, size_t>& Ahat,
                                 size_t gbs,
                                 size_t W,
                                 const std::vector<size_t>& P ) {
            const size_t n = Ahat.rows, nwin = Ahat.cols;
            if ( n == 0 || P.size() != n || gbs == 0 || W % gbs != 0 )
                return 0;
            const size_t scale = W / gbs; // colblock = w * scale
            const size_t ncblk = ( ( nwin ? nwin - 1 : 0 ) * scale + 1 ) + 1;
            std::vector<size_t> epoch( ncblk, 0 );
            size_t ep = 0, cost = 0;
            for ( size_t s = 0; s < n; s += gbs ) {
                if ( ++ep == 0 ) {
                    std::fill( epoch.begin(), epoch.end(), 0 );
                    ep = 1;
                }
                const size_t end = std::min( s + gbs, n );
                for ( size_t pos = s; pos < end; ++pos ) {
                    const size_t r = P[pos];
                    for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k ) {
                        const size_t w = Ahat.col_ind[k];
                        if ( w >= nwin )
                            continue;
                        const size_t cb = w * scale;
                        if ( cb < ncblk && epoch[cb] != ep ) {
                            epoch[cb] = ep; // ep is per row-stripe: (stripe, colblock)
                            ++cost;
                        }
                    }
                }
            }
            return cost;
        }

        template <typename DataT = float, typename intT = int>
        inline std::vector<size_t> chain_reorder( CSR<DataT, intT>& A,
                                                  size_t W,
                                                  size_t bs,
                                                  size_t k_hits,
                                                  size_t max_passes,
                                                  size_t window,
                                                  ChainStats* st = nullptr ) {
            if ( W == 0 )
                W = 16;
            if ( bs == 0 )
                bs = W;
            CSR<size_t, size_t> Ahat;
            mask( A, W, Ahat );
            const size_t n = Ahat.rows;
            std::vector<size_t> P;
            anchor_rows( Ahat, P, k_hits, st );
            if ( max_passes > 0 )
                refine::stripe_refine( Ahat, P, bs, max_passes, window );
            // Identity guard: the sort keys are locality-preserving but can still
            // scramble an order that is already good (e.g. banded natural order).
            // Multi-scale block cost on the sketch is O(mhat) per scale; keep
            // whichever candidate is better on the summed cost.
            std::vector<size_t> I( n );
            std::iota( I.begin(), I.end(), 0 );
            size_t costP = 0, costI = 0;
            for ( size_t gbs : { (size_t)4, (size_t)16, (size_t)64 } ) {
                if ( gbs > n )
                    continue;
                costP += stripe_cost( Ahat, gbs, P );
                costI += stripe_cost( Ahat, gbs, I );
            }
            if ( costP > costI )
                P = I;
            return P;
        }

    } // namespace chain
} // namespace club
