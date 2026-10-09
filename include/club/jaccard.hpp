#pragma once
// jaccard.hpp — fast Jaccard clustering on the 0-1 window sketch.
//
// Back to the original CLUB idea, sub-quadratic:
//   1. mask(W == bs)           -> 0-1 row x window sketch, O(nnz) parallel
//   2. capped inverted index   -> candidate neighbours per row (top-K rows
//                                 per window, IDF-ranked). Never materialises
//                                 row x row pairs: O(mhat) memory.
//   3. exact Jaccard on sketch sets for the (few) candidate pairs,
//      union-find at threshold tau -> clusters. O(mhat * K * d).
//   4. order clusters by IDF mass, chain rows WITHIN a cluster with the
//      stripe-union greedy (bounded by bs so it stays near-linear).
//   5. optional stripe_refine polish (exact local block search).
//   6. alternate rows/columns until the block cost stops improving
//      (bond-energy style, both sides permuted, still no quadratic step).
//
// Complexity per row pass: O(nnz + mhat*K*d + n log n) — no O(n^2) term.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "chain.hpp"   // mask, SketchIndex (legacy), stripe_cost, ChainStats
#include "refine.hpp"  // stripe_refine
#include "reorder.hpp" // permute, permute_cols, transpose

namespace club {
    namespace jacc {

        struct JaccardStats {
            size_t n = 0;
            size_t mhat = 0;
            size_t n_clusters = 0;
            size_t iters_used = 0;
            double index_ms = 0.0;
            double cluster_ms = 0.0;
            double order_ms = 0.0;
            double permute_ms = 0.0;
            std::vector<size_t> cost_trace; // block cost after each row+col pass
        };

        // ---------------------------------------------------------------------------
        // Disjoint-set union with min-root union (deterministic component ids).
        // ---------------------------------------------------------------------------
        struct DSU {
            std::vector<size_t> p;
            void init( size_t n ) {
                p.resize( n );
                std::iota( p.begin(), p.end(), 0 );
            }
            size_t find( size_t x ) {
                while ( p[x] != x ) {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                return x;
            }
            void unite( size_t a, size_t b ) {
                a = find( a );
                b = find( b );
                if ( a == b )
                    return;
                if ( b < a )
                    std::swap( a, b );
                p[b] = a; // root is always the min id of the component
            }
        };

        // ---------------------------------------------------------------------------
        // cluster_rows: candidate gather via the capped inverted index, exact Jaccard
        // on the candidate pairs, union-find at tau. Fills `members` (each list in
        // mass-descending order) and `comp_of` (row -> component id).
        // ---------------------------------------------------------------------------
        inline void cluster_rows( const CSR<size_t, size_t>& Ahat,
                                  const chain::legacy::SketchIndex& idx,
                                  double tau,
                                  size_t cand_cap,
                                  std::vector<size_t>& comp_of,
                                  std::vector<std::vector<size_t>>& members,
                                  JaccardStats* st = nullptr ) {
            auto t0 = std::chrono::steady_clock::now();
            const size_t n = Ahat.rows, nwin = Ahat.cols;
            comp_of.assign( n, 0 );
            members.clear();
            if ( n == 0 )
                return;
            if ( cand_cap == 0 )
                cand_cap = 64;

            // ---- Phase 1 (parallel): gather candidate sets ------------------------
            std::vector<std::vector<size_t>> cand_local( n );
#pragma omp parallel
            {
                std::vector<int> cep( n, 0 );
                int ep = 0;
                std::vector<size_t> cand;
                cand.reserve( cand_cap );
                std::vector<std::vector<size_t>> mine( n );
#pragma omp for schedule( static )
                for ( long long ii = 0; ii < (long long)n; ++ii ) {
                    const size_t i = (size_t)ii;
                    cand.clear();
                    ++ep;
                    bool full = false;
                    for ( size_t k = Ahat.row_ptr[i]; k < Ahat.row_ptr[i + 1] && !full; ++k ) {
                        const size_t w = Ahat.col_ind[k];
                        if ( w >= nwin || w >= idx.inv.size() )
                            continue;
                        for ( size_t c : idx.inv[w] ) {
                            if ( c == i || cep[c] == ep )
                                continue;
                            cep[c] = ep;
                            cand.push_back( c );
                            if ( cand.size() >= cand_cap ) {
                                full = true;
                                break;
                            }
                        }
                    }
                    if ( !cand.empty() )
                        mine[i].swap( cand );
                }
#pragma omp critical
                {
                    for ( size_t i = 0; i < n; ++i )
                        if ( !mine[i].empty() )
                            cand_local[i].swap( mine[i] );
                }
            }

            // ---- Phase 2 (serial): exact Jaccard on candidate pairs, union-find ---
            DSU dsu;
            dsu.init( n );
            std::vector<size_t> wmark( nwin, 0 );
            size_t wep = 0;
            std::vector<size_t> order( n );
            std::iota( order.begin(), order.end(), 0 );
            std::sort( order.begin(), order.end(), [&]( size_t a, size_t b ) {
                if ( cand_local[a].size() != cand_local[b].size() )
                    return cand_local[a].size() > cand_local[b].size();
                return a < b;
            } );
            for ( size_t i : order ) {
                const auto& ci = cand_local[i];
                if ( ci.empty() )
                    continue;
                const size_t ki = Ahat.row_ptr[i + 1] - Ahat.row_ptr[i];
                if ( ++wep == 0 ) {
                    std::fill( wmark.begin(), wmark.end(), 0 );
                    wep = 1;
                }
                for ( size_t k = Ahat.row_ptr[i]; k < Ahat.row_ptr[i + 1]; ++k )
                    if ( Ahat.col_ind[k] < nwin )
                        wmark[Ahat.col_ind[k]] = wep;
                for ( size_t c : ci ) {
                    if ( dsu.find( i ) == dsu.find( c ) )
                        continue;
                    const size_t kc = Ahat.row_ptr[c + 1] - Ahat.row_ptr[c];
                    size_t inter = 0;
                    for ( size_t k = Ahat.row_ptr[c]; k < Ahat.row_ptr[c + 1]; ++k ) {
                        const size_t w = Ahat.col_ind[k];
                        if ( w < nwin && wmark[w] == wep )
                            ++inter;
                    }
                    const size_t uni = ki + kc - inter;
                    const double j = uni ? (double)inter / (double)uni : 0.0;
                    if ( j >= tau )
                        dsu.unite( i, c );
                }
            }

            // ---- Collect components, first-touch in mass order ---------------------
            std::vector<size_t> root2comp( n, std::numeric_limits<size_t>::max() );
            for ( size_t r : idx.mass_order ) {
                const size_t root = dsu.find( r );
                if ( root2comp[root] == std::numeric_limits<size_t>::max() ) {
                    root2comp[root] = members.size();
                    members.push_back( {} );
                }
                members[root2comp[root]].push_back( r );
            }
            for ( size_t i = 0; i < n; ++i )
                comp_of[i] = root2comp[dsu.find( i )];
            auto t1 = std::chrono::steady_clock::now();
            if ( st ) {
                st->n_clusters = members.size();
                st->cluster_ms += std::chrono::duration<double, std::milli>( t1 - t0 ).count();
            }
        }

        // ---------------------------------------------------------------------------
        // ChainScratch: reusable epoch-stamped buffers so we never allocate O(n) /
        // O(nwin) per cluster.
        // ---------------------------------------------------------------------------
        struct ChainScratch {
            std::vector<char> placed;   // n
            std::vector<int> cepoch;    // n  (candidate dedup)
            std::vector<int> mflag;     // n  (membership test)
            std::vector<size_t> wepoch; // nwin
            std::vector<size_t> wcnt;   // nwin
            int cep = 0;
            int mep = 0;
            size_t wep = 0;

            void init( size_t n, size_t nwin ) {
                placed.assign( n, 0 );
                cepoch.assign( n, 0 );
                mflag.assign( n, 0 );
                wepoch.assign( nwin, 0 );
                wcnt.assign( nwin, 0 );
                cep = mep = 0;
                wep = 0;
            }
        };

        // chain_rows over a SUBSET of rows (`members`, mass-descending). Appends the
        // chained order to P. Serial; bounded by cluster size — total cost is
        // O(sum_c m_c * bs * K * d).
        inline void chain_subset( const CSR<size_t, size_t>& Ahat,
                                  const chain::legacy::SketchIndex& idx,
                                  const std::vector<size_t>& members,
                                  size_t bs,
                                  ChainScratch& sc,
                                  std::vector<size_t>& P ) {
            const size_t n = Ahat.rows, nwin = Ahat.cols;
            const size_t m = members.size();
            if ( m == 0 )
                return;
            if ( m == 1 ) {
                P.push_back( members[0] );
                return;
            }
            if ( ++sc.mep == 0 ) {
                std::fill( sc.mflag.begin(), sc.mflag.end(), 0 );
                sc.mep = 1;
            }
            for ( size_t r : members )
                sc.mflag[r] = sc.mep;

            const size_t K = idx.K;
            size_t stripe_start = 0;
            size_t mi = 0;
            const size_t first = members[0];
            sc.placed[first] = 1;
            P.push_back( first );

            // Incremental stripe union (same semantics as chain_rows): win_epoch/wcnt
            // persist across steps and reset every bs rows; ulist lists the union
            // windows in first-touch order and drives candidate generation.
            std::vector<size_t> ulist;
            ulist.reserve( bs * 8 );
            auto union_add = [&]( size_t w ) {
                if ( w >= nwin )
                    return;
                if ( sc.wepoch[w] != sc.wep ) {
                    sc.wepoch[w] = sc.wep;
                    sc.wcnt[w] = 1;
                    ulist.push_back( w );
                } else {
                    ++sc.wcnt[w];
                }
            };
            for ( size_t k = Ahat.row_ptr[first]; k < Ahat.row_ptr[first + 1]; ++k )
                union_add( Ahat.col_ind[k] );

            for ( size_t pos = 1; pos < m; ++pos ) {
                const size_t prev = P.back();
                if ( ( pos - stripe_start ) >= bs ) {
                    stripe_start = pos;
                    if ( ++sc.wep == std::numeric_limits<size_t>::max() ) {
                        std::fill( sc.wepoch.begin(), sc.wepoch.end(), 0 );
                        sc.wep = 1;
                    }
                    ulist.clear();
                }
                for ( size_t k = Ahat.row_ptr[prev]; k < Ahat.row_ptr[prev + 1]; ++k )
                    union_add( Ahat.col_ind[k] );
                const size_t cur_wep = sc.wep;

                if ( ++sc.cep == 0 ) {
                    std::fill( sc.cepoch.begin(), sc.cepoch.end(), 0 );
                    sc.cep = 1;
                }
                size_t best = std::numeric_limits<size_t>::max();
                double best_s = -1.0;
                for ( size_t w : ulist ) {
                    if ( w >= nwin || w >= idx.inv.size() )
                        continue;
                    size_t taken = 0;
                    for ( size_t c : idx.inv[w] ) {
                        if ( c >= n || sc.placed[c] || sc.mflag[c] != sc.mep )
                            continue;
                        if ( sc.cepoch[c] == sc.cep )
                            continue;
                        sc.cepoch[c] = sc.cep;
                        if ( taken++ >= K )
                            break;
                        double s = 0.0;
                        for ( size_t k = Ahat.row_ptr[c]; k < Ahat.row_ptr[c + 1]; ++k ) {
                            const size_t w2 = Ahat.col_ind[k];
                            if ( w2 < nwin && sc.wepoch[w2] == cur_wep )
                                s += idx.idf[w2];
                        }
                        if ( s > best_s + 1e-12 || ( s == best_s && c < best ) ) {
                            best_s = s;
                            best = c;
                        }
                    }
                }
                if ( best == std::numeric_limits<size_t>::max() ) {
                    while ( mi < m && sc.placed[members[mi]] )
                        ++mi;
                    best = members[mi];
                }
                sc.placed[best] = 1;
                P.push_back( best );
            }
            for ( size_t r : members )
                if ( sc.mflag[r] == sc.mep )
                    sc.mflag[r] = 0;
        }

        // ---------------------------------------------------------------------------
        // order_from_clusters: macro order = clusters by leading-row IDF mass; within
        // each cluster the greedy stripe-union chain.
        // ---------------------------------------------------------------------------
        inline void order_from_clusters( const CSR<size_t, size_t>& Ahat,
                                         const chain::legacy::SketchIndex& idx,
                                         const std::vector<std::vector<size_t>>& members,
                                         size_t bs,
                                         ChainScratch& sc,
                                         std::vector<size_t>& P,
                                         JaccardStats* st = nullptr ) {
            auto t0 = std::chrono::steady_clock::now();
            P.clear();
            P.reserve( Ahat.rows );
            sc.init( Ahat.rows, Ahat.cols );
            std::vector<size_t> comp_order( members.size() );
            std::iota( comp_order.begin(), comp_order.end(), 0 );
            std::sort( comp_order.begin(), comp_order.end(), [&]( size_t a, size_t b ) {
                const double ma = members[a].empty() ? 0.0 : idx.mass[members[a][0]];
                const double mb = members[b].empty() ? 0.0 : idx.mass[members[b][0]];
                if ( ma != mb )
                    return ma > mb;
                const size_t ra = members[a].empty() ? 0 : members[a][0];
                const size_t rb = members[b].empty() ? 0 : members[b][0];
                return ra < rb;
            } );
            for ( size_t c : comp_order )
                chain_subset( Ahat, idx, members[c], bs, sc, P );
            auto t1 = std::chrono::steady_clock::now();
            if ( st )
                st->order_ms += std::chrono::duration<double, std::milli>( t1 - t0 ).count();
        }

        // ---------------------------------------------------------------------------
        // jaccard_reorder: the full technique. Alternates row/column passes until the
        // bs-block cost stops strictly improving. levels ladder: empty = {bs};
        // otherwise coarse-to-fine sketches (e.g. {4*bs, bs}).
        // Returns the BEST row permutation seen (positions -> rows). If both_sides,
        // also fills out_col_perm (positions -> original column indices).
        // The matrix A is left in the state of the LAST iteration (use the returned
        // best permutations for output).
        // ---------------------------------------------------------------------------
        template <typename DataT = float, typename intT = int>
        std::vector<size_t> jaccard_reorder( CSR<DataT, intT>& A,
                                             size_t W,
                                             size_t bs,
                                             size_t K,
                                             double tau,
                                             size_t iters,
                                             size_t refine_passes,
                                             bool both_sides,
                                             const std::vector<size_t>& levels,
                                             std::vector<size_t>* out_col_perm = nullptr,
                                             JaccardStats* st = nullptr ) {
            if ( W == 0 )
                W = bs ? bs : 16;
            if ( bs == 0 )
                bs = W;
            if ( K == 0 )
                K = 8;
            const size_t n = static_cast<size_t>( A.rows );
            std::vector<size_t> cum_rows( n );
            std::iota( cum_rows.begin(), cum_rows.end(), 0 );
            std::vector<size_t> cum_cols;
            if ( both_sides ) {
                cum_cols.resize( static_cast<size_t>( A.cols ) );
                std::iota( cum_cols.begin(), cum_cols.end(), 0 );
            }

            auto compose = []( const std::vector<size_t>& cum, const std::vector<size_t>& P ) {
                std::vector<size_t> out( cum.size() );
                for ( size_t i = 0; i < P.size(); ++i )
                    out[i] = cum[P[i]];
                return out;
            };

            // Cost of the current (already permuted) matrix, multi-scale: the
            // reported metric averages bs=4..128, so a single-scale guard can
            // accept a perm that regresses the mean (seen on ash292). Sum of stripe
            // costs at {4,16,64,bs} on one sketch is O(mhat) per scale and tracks
            // the measured metric closely enough for accept/reject decisions.
            auto current_cost = [&]( size_t& cost ) {
                CSR<size_t, size_t> Ah;
                club::mask( A, bs, Ah );
                std::vector<size_t> I( n );
                std::iota( I.begin(), I.end(), 0 );
                cost = 0;
                for ( size_t gbs : { (size_t)4, (size_t)16, (size_t)64, bs } ) {
                    if ( gbs == 0 || gbs > n )
                        continue;
                    cost += chain::stripe_cost( Ah, gbs, I );
                }
            };

            // Never-worse-than-identity: benchmark the original matrix first, so a
            // permutation is only kept if it strictly improves the bs-block cost.
            size_t best_cost = 0;
            current_cost( best_cost );
            std::vector<size_t> best_rows = cum_rows, best_cols = cum_cols;
            (void)levels; // single-level at W == bs; the coarse->fine ladder showed no
                          // benefit in the A/B (kept for API compatibility)

            size_t it = 0;
            for ( ; it < iters; ++it ) {
                // ---- candidate orders at W == bs ----
                CSR<size_t, size_t> Ah;
                club::mask( A, bs, Ah );
                chain::legacy::SketchIndex idx;
                idx.build( Ah, K, nullptr );
                std::vector<size_t> comp_dummy;
                std::vector<std::vector<size_t>> mems;
                cluster_rows( Ah, idx, tau, 0, comp_dummy, mems, st );
                ChainScratch sc;
                std::vector<size_t> Pj;
                order_from_clusters( Ah, idx, mems, bs, sc, Pj, st );
                std::vector<size_t> Ps;
                chain::anchor_rows( Ah, Ps, 8, nullptr );
                // pick the better of {jaccard-cluster-chain, anchor-sort} by exact
                // nonzero-block count (true_cost IS the measured metric)
                const size_t cj = chain::true_cost( Ah, bs, bs, Pj );
                const size_t cs = chain::true_cost( Ah, bs, bs, Ps );
                std::vector<size_t> P = ( cs < cj ) ? Ps : Pj;
                if ( refine_passes > 0 )
                    club::refine::stripe_refine( Ah, P, bs, refine_passes, 3 );
                cum_rows = compose( cum_rows, P );
                auto tp = std::chrono::steady_clock::now();
                club::permute( A, P );
                if ( st )
                    st->permute_ms += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - tp )
                                          .count();
                // ---- column passes ----
                if ( both_sides ) {
                    CSR<DataT, intT> AT;
                    club::transpose( A, AT );
                    CSR<size_t, size_t> Ahc;
                    club::mask( AT, bs, Ahc );
                    chain::legacy::SketchIndex idxc;
                    idxc.build( Ahc, K, nullptr );
                    std::vector<size_t> comp_c;
                    std::vector<std::vector<size_t>> mems_c;
                    cluster_rows( Ahc, idxc, tau, 0, comp_c, mems_c, st );
                    ChainScratch scc;
                    std::vector<size_t> Qj;
                    order_from_clusters( Ahc, idxc, mems_c, bs, scc, Qj, st );
                    std::vector<size_t> Qs;
                    chain::anchor_rows( Ahc, Qs, 8, nullptr );
                    const size_t qj = chain::true_cost( Ahc, bs, bs, Qj );
                    const size_t qs = chain::true_cost( Ahc, bs, bs, Qs );
                    std::vector<size_t> Q = ( qs < qj ) ? Qs : Qj;
                    if ( refine_passes > 0 )
                        club::refine::stripe_refine( Ahc, Q, bs, refine_passes, 3 );
                    cum_cols = compose( cum_cols, Q );
                    club::permute_cols( A, Q );
                }
                // ---- evaluate ----
                size_t cost = 0;
                current_cost( cost );
                if ( st )
                    st->cost_trace.push_back( cost );
                if ( cost < best_cost ) {
                    best_cost = cost;
                    best_rows = cum_rows;
                    best_cols = cum_cols;
                } else {
                    ++it;
                    break; // no strict improvement -> converged
                }
            }
            if ( st ) {
                st->iters_used = it;
                st->n = n;
            }
            if ( out_col_perm )
                *out_col_perm = both_sides ? best_cols : std::vector<size_t>();
            return best_rows;
        }

    } // namespace jacc
} // namespace club
