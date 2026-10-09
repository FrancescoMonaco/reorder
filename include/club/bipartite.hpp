#pragma once
// bipartite.hpp — Track R: bipartite Louvain / label-propagation ordering.
// Includes reorder.hpp for complete club:: types (one-directional include).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include "club/reorder.hpp"

namespace club {
    namespace bipartite {

        // Weighted bipartite scoring context shared by LP + traversal.
        struct Weights {
            std::vector<double> window_w;
            std::vector<double> cluster_n;
            double max_w = 1.0;
        };

        inline Weights build_weights( const MacroGraph& g ) {
            Weights wt;
            const size_t nm = g.n_micro;
            wt.window_w.assign( g.n_windows, 0.0 );
            wt.cluster_n.assign( nm, 1.0 );
            for ( size_t w = 0; w < g.n_windows; ++w ) {
                size_t df = g.window_to_clusters[w].size();
                if ( df == 0 )
                    continue;
                wt.window_w[w] =
                    std::log( 1.0 + static_cast<double>( nm ) / static_cast<double>( df ) );
                wt.max_w = std::max( wt.max_w, wt.window_w[w] );
            }
            for ( size_t c = 0; c < nm; ++c ) {
                size_t deg = g.macro_sig[c].size();
                wt.cluster_n[c] = ( deg == 0 ) ? 1.0 : std::sqrt( static_cast<double>( deg ) );
            }
            return wt;
        }

        // Bipartite label propagation with deterministic tie-breaks.
        inline std::vector<size_t>
        label_propagate( const MacroGraph& g, const Weights& wt, size_t max_sweeps = 8 ) {
            const size_t nm = g.n_micro;
            const size_t nw = g.n_windows;
            std::vector<size_t> clab( nm ), wlab( nw );
            std::iota( clab.begin(), clab.end(), 0 );
            for ( size_t w = 0; w < nw; ++w )
                wlab[w] = nm + w;

            std::unordered_map<size_t, double> acc;
            for ( size_t sweep = 0; sweep < max_sweeps; ++sweep ) {
                bool changed = false;
                for ( size_t w = 0; w < nw; ++w ) {
                    const auto& members = g.window_to_clusters[w];
                    if ( members.empty() )
                        continue;
                    acc.clear();
                    for ( size_t c : members ) {
                        if ( c >= nm )
                            continue;
                        acc[clab[c]] += wt.window_w[w] / wt.cluster_n[c];
                    }
                    size_t best = wlab[w];
                    double best_v = -1.0;
                    for ( const auto& kv : acc ) {
                        if ( kv.second > best_v + 1e-12 ||
                             ( std::abs( kv.second - best_v ) <= 1e-12 && kv.first < best ) ) {
                            best_v = kv.second;
                            best = kv.first;
                        }
                    }
                    if ( best != wlab[w] ) {
                        wlab[w] = best;
                        changed = true;
                    }
                }
                for ( size_t c = 0; c < nm; ++c ) {
                    if ( g.macro_sig[c].empty() )
                        continue;
                    acc.clear();
                    for ( size_t w : g.macro_sig[c] )
                        acc[wlab[w]] += wt.window_w[w] / wt.cluster_n[c];
                    size_t best = clab[c];
                    double best_v = -1.0;
                    for ( const auto& kv : acc ) {
                        if ( kv.second > best_v + 1e-12 ||
                             ( std::abs( kv.second - best_v ) <= 1e-12 && kv.first < best ) ) {
                            best_v = kv.second;
                            best = kv.first;
                        }
                    }
                    if ( best != clab[c] ) {
                        clab[c] = best;
                        changed = true;
                    }
                }
                if ( !changed )
                    break;
            }
            return clab;
        }

        // Weighted overlap: sum of shared window weights, length-normalised.
        inline double
        weighted_overlap( const MacroGraph& g, const Weights& wt, size_t a, size_t b ) {
            const auto& sa = g.macro_sig[a];
            const auto& sb = g.macro_sig[b];
            double num = 0.0;
            size_t i = 0, j = 0;
            while ( i < sa.size() && j < sb.size() ) {
                if ( sa[i] == sb[j] ) {
                    num += wt.window_w[sa[i]];
                    ++i;
                    ++j;
                } else if ( sa[i] < sb[j] ) {
                    ++i;
                } else {
                    ++j;
                }
            }
            double den = wt.cluster_n[a] * wt.cluster_n[b];
            return ( den > 0.0 ) ? num / den : 0.0;
        }

        // Gain-ordered traversal with a LAZY MAX-HEAP frontier.
        //
        // Why: the previous greedy rescan was O(|comm|^2) *and* re-evaluated
        // weighted_overlap(last, c) for every candidate at every step, i.e.
        // O(|comm|^2 * l) overall -- super-linear in practice on million-row
        // graphs (this was the dominant cost of Track R).
        //
        // Two exact/approximate relaxations with negligible quality impact:
        //   1. score[] only ever *increases* (commit() adds weight, never removes),
        //      so a lazy max-heap with stale-entry skipping reproduces the same
        //      greedy max-score pick in O(E log n).
        //   2. The secondary "similarity to last placed" term is now evaluated only
        //      on the TOP_F freshest frontier candidates instead of all of them --
        //      same decision among the strongest-connected candidates, O(TOP_F*l)
        //      per step instead of O(|comm|*l).
        inline std::vector<size_t> order_by_strength( const MacroGraph& g,
                                                      const Weights& wt,
                                                      const std::vector<size_t>& clab ) {
            constexpr size_t TOP_F = 4;
            const size_t nm = g.n_micro;
            const size_t nw = g.n_windows;
            std::unordered_map<size_t, std::vector<size_t>> comm;
            for ( size_t c = 0; c < nm; ++c )
                comm[clab[c]].push_back( c );
            std::vector<std::vector<size_t>> comms;
            comms.reserve( comm.size() );
            for ( auto& kv : comm )
                comms.push_back( std::move( kv.second ) );
            std::sort( comms.begin(), comms.end(), []( const auto& a, const auto& b ) {
                if ( a.size() != b.size() )
                    return a.size() > b.size();
                return a.front() < b.front();
            } );

            std::vector<size_t> cluster_order;
            cluster_order.reserve( nm );
            std::vector<char> placed( nm, 0 );
            std::vector<char> win_seen( nw, 0 );
            std::vector<double> score( nm, 0.0 );

            // Max-heap on (score, smaller id wins ties). Stale entries are skipped
            // by comparing the popped score against the live score[].
            struct Entry {
                double s;
                size_t c;
            };
            auto cmp = []( const Entry& a, const Entry& b ) {
                if ( a.s != b.s )
                    return a.s < b.s; // min-heap by score == max-heap by -score
                return a.c > b.c;     // smaller id = higher priority on ties
            };
            std::priority_queue<Entry, std::vector<Entry>, decltype( cmp )> pq( cmp );

            for ( auto& members : comms ) {
                size_t seed = members[0];
                double seed_deg = std::numeric_limits<double>::infinity();
                for ( size_t c : members ) {
                    double d = 0.0;
                    for ( size_t w : g.macro_sig[c] )
                        d += wt.window_w[w];
                    if ( d < seed_deg || ( d == seed_deg && c < seed ) ) {
                        seed = c;
                        seed_deg = d;
                    }
                }
                std::vector<char> in_comm( nm, 0 );
                for ( size_t c : members )
                    in_comm[c] = 1;

                auto commit = [&]( size_t c ) {
                    placed[c] = 1;
                    cluster_order.push_back( c );
                    for ( size_t w : g.macro_sig[c] ) {
                        if ( win_seen[w] )
                            continue;
                        win_seen[w] = 1;
                        for ( size_t c2 : g.window_to_clusters[w] ) {
                            if ( c2 < nm && in_comm[c2] && !placed[c2] ) {
                                score[c2] +=
                                    wt.window_w[w] / ( wt.cluster_n[c] * wt.cluster_n[c2] );
                                pq.push( { score[c2], c2 } );
                            }
                        }
                    }
                };
                commit( seed );
                size_t last = seed;
                for ( size_t t = 1; t < members.size(); ++t ) {
                    // Collect up to TOP_F fresh (unplaced, up-to-date-score) candidates.
                    Entry cands[TOP_F];
                    size_t ncand = 0;
                    while ( !pq.empty() && ncand < TOP_F ) {
                        Entry e = pq.top();
                        pq.pop();
                        if ( placed[e.c] )
                            continue;
                        if ( e.s < score[e.c] - 1e-12 )
                            continue; // stale
                        cands[ncand++] = e;
                    }
                    // Drain duplicates until a fresh one is available if none found.
                    if ( ncand == 0 ) {
                        // Disconnected remainder of the community: fall back to the
                        // lowest-id unplaced member (score is 0 or accumulated from
                        // cross-community windows, which we intentionally ignore).
                        size_t fallback = nm;
                        for ( size_t c : members ) {
                            if ( !placed[c] ) {
                                fallback = c;
                                break;
                            }
                        }
                        if ( fallback == nm )
                            break; // all placed (defensive; should not happen)
                        commit( fallback );
                        last = fallback;
                        continue;
                    }
                    // Re-evaluate the last-overlap heuristic only on the few best
                    // candidates; keep the losers in the heap (their score is
                    // unchanged, so re-pushing is safe).
                    size_t best = cands[0].c;
                    double best_s = score[cands[0].c] + weighted_overlap( g, wt, last, cands[0].c );
                    for ( size_t f = 1; f < ncand; ++f ) {
                        double s = score[cands[f].c] + weighted_overlap( g, wt, last, cands[f].c );
                        if ( s > best_s + 1e-12 ||
                             ( std::abs( s - best_s ) <= 1e-12 && cands[f].c < best ) ) {
                            best_s = s;
                            best = cands[f].c;
                        }
                    }
                    for ( size_t f = 0; f < ncand; ++f )
                        if ( cands[f].c != best )
                            pq.push( cands[f] );
                    commit( best );
                    last = best;
                }
                std::fill( win_seen.begin(), win_seen.end(), 0 );
            }
            assert( cluster_order.size() == nm );
            return cluster_order;
        }

        // Medoid intra-cluster row ordering + RCM-style chain refinement.
        //
        // Medoid step (unchanged semantics): sampled medoid anchors a distance key;
        // rows sort by distance to medoid. This is star-shaped: every row is close
        // to the medoid, but CONSECUTIVE rows in the sorted order need not share
        // anything. For bs=4/8 stripes that is fatal: a stripe of 4 rows each close
        // to the medoid via DIFFERENT windows has a 4x union.
        //
        // Chain refinement (new, fixes bs=4/8): after medoid sort, run a greedy
        // nearest-neighbour walk restricted to the cluster: start from the medoid,
        // repeatedly pick the unvisited row with max IDF-weighted overlap to the
        // last placed row (ties by id). This is the TSP-path heuristic (GROOT's
        // DFS-on-similarity-graph in miniature, Rabbit's community-chain analogue).
        // Consecutive rows now share windows by construction -> small-stripe unions
        // shrink. Cost O(|c|^2 * l) worst case per cluster; bounded in practice
        // because micro-clusters are <= micro_threshold rows. Only runs when the
        // cluster is small enough (CHAIN_MAX) to keep the coarse stage linear.
        inline void order_rows_by_medoid( const CSR<size_t, size_t>& Ahat,
                                          std::vector<std::vector<size_t>>& micro_groups ) {
            constexpr size_t MEDOID_SAMPLES = 24;

            auto jaccard_rows = [&]( size_t a, size_t b ) -> double {
                size_t sa = Ahat.row_ptr[a], ea = Ahat.row_ptr[a + 1];
                size_t sb = Ahat.row_ptr[b], eb = Ahat.row_ptr[b + 1];
                size_t na = ea - sa, nb = eb - sb;
                if ( na == 0 && nb == 0 )
                    return 0.0;
                if ( na == 0 || nb == 0 )
                    return 1.0;
                size_t isect = 0, pa = sa, pb = sb;
                while ( pa < ea && pb < eb ) {
                    size_t ca = Ahat.col_ind[pa], cb = Ahat.col_ind[pb];
                    if ( ca == cb ) {
                        ++isect;
                        ++pa;
                        ++pb;
                    } else if ( ca < cb ) {
                        ++pa;
                    } else {
                        ++pb;
                    }
                }
                return 1.0 - static_cast<double>( isect ) / static_cast<double>( na + nb - isect );
            };

            std::vector<double> dist;
            for ( auto& grp : micro_groups ) {
                const size_t sz = grp.size();
                if ( sz <= 2 )
                    continue;

                // Deterministic candidate sample: evenly strided members.
                size_t ncand = std::min( MEDOID_SAMPLES, sz );
                size_t stride = sz / ncand;
                size_t medoid = grp[0];
                double best = std::numeric_limits<double>::infinity();
                for ( size_t ci = 0; ci < ncand; ++ci ) {
                    size_t r = grp[ci * stride];
                    // Score candidate against an evenly strided reference sample.
                    size_t nref = std::min( MEDOID_SAMPLES, sz );
                    double acc = 0.0;
                    for ( size_t ri = 0; ri < nref; ++ri ) {
                        size_t o = grp[ri * ( sz / nref )];
                        if ( o != r )
                            acc += jaccard_rows( r, o );
                    }
                    if ( acc < best ) {
                        best = acc;
                        medoid = r;
                    }
                }

                // Precompute distances to the medoid once, then sort by key.
                dist.assign( sz, 0.0 );
                for ( size_t t = 0; t < sz; ++t )
                    dist[t] = ( grp[t] == medoid ) ? -1.0 : jaccard_rows( medoid, grp[t] );
                std::vector<size_t> idx( sz );
                std::iota( idx.begin(), idx.end(), 0 );
                std::sort( idx.begin(), idx.end(), [&]( size_t a, size_t b ) {
                    if ( dist[a] != dist[b] )
                        return dist[a] < dist[b];
                    return grp[a] < grp[b];
                } );
                std::vector<size_t> sorted( sz );
                for ( size_t t = 0; t < sz; ++t )
                    sorted[t] = grp[idx[t]];
                grp.swap( sorted );

                // Chain refinement: greedy NN walk from the medoid using exact
                // shared-window overlap (unweighted here — Ahat is the W==bs sketch
                // for the target scale when called from stripe flow, binary overlap
                // is the right affinity). Only for small clusters to stay linear.
                constexpr size_t CHAIN_MAX = 64;
                if ( sz <= CHAIN_MAX ) {
                    std::vector<char> used( sz, 0 );
                    std::vector<size_t> chain;
                    chain.reserve( sz );
                    size_t cur_idx = 0; // medoid is at sorted[0] (dist -1.0)
                    chain.push_back( sorted[cur_idx] );
                    used[cur_idx] = 1;
                    auto overlap = [&]( size_t a, size_t b ) -> size_t {
                        size_t sa = Ahat.row_ptr[a], ea = Ahat.row_ptr[a + 1];
                        size_t sb = Ahat.row_ptr[b], eb = Ahat.row_ptr[b + 1];
                        size_t is = 0, pa = sa, pb = sb;
                        while ( pa < ea && pb < eb ) {
                            size_t ca = Ahat.col_ind[pa], cb = Ahat.col_ind[pb];
                            if ( ca == cb ) {
                                ++is;
                                ++pa;
                                ++pb;
                            } else if ( ca < cb ) {
                                ++pa;
                            } else {
                                ++pb;
                            }
                        }
                        return is;
                    };
                    for ( size_t step = 1; step < sz; ++step ) {
                        size_t best_i = sz;
                        size_t best_ov = 0;
                        size_t cur_row = chain.back();
                        for ( size_t t = 0; t < sz; ++t ) {
                            if ( used[t] )
                                continue;
                            size_t ov = overlap( cur_row, sorted[t] );
                            if ( best_i == sz || ov > best_ov ||
                                 ( ov == best_ov && sorted[t] < sorted[best_i] ) ) {
                                best_ov = ov;
                                best_i = t;
                            }
                        }
                        used[best_i] = 1;
                        chain.push_back( sorted[best_i] );
                    }
                    grp.swap( chain );
                }
            }
        }

        // Track R entry point (coarse order used by Track S as well).
        // W_target: window size of the sketch's *intended evaluation scale*.
        // The chain refinement inside order_rows_by_medoid uses binary overlap on
        // this sketch, so callers that want bs=4/8 quality should pass a fine sketch
        // (W=4/8) OR rely on stripe_refine afterwards (recommended: coarse W=32 +
        // stripe bs_list={4,8,16,32}).
        inline void cluster_bipartite_lp( const CSR<size_t, size_t>& Ahat,
                                          std::vector<size_t>& P,
                                          size_t micro_threshold = 0,
                                          size_t lp_sweeps = 8,
                                          MicroMacroStats* stats_out = nullptr ) {
            const size_t n = Ahat.rows;
            P.resize( n );
            if ( n == 0 )
                return;
            MacroGraph graph = build_macro_graph( Ahat, micro_threshold, 0.15 );
            Weights wt = build_weights( graph );
            std::vector<size_t> clab = label_propagate( graph, wt, lp_sweeps );
            std::vector<size_t> cluster_order = order_by_strength( graph, wt, clab );
            order_rows_by_medoid( Ahat, graph.micro_groups );
            size_t pos = 0;
            for ( size_t r : graph.exhausted_rows )
                P[pos++] = r;
            for ( size_t c : cluster_order )
                for ( size_t r : graph.micro_groups[c] )
                    P[pos++] = r;
            assert( pos == n );
            if ( stats_out ) {
                stats_out->n_micro = graph.n_micro;
                stats_out->micro_threshold = micro_threshold;
                stats_out->macro_incidences = graph.macro_incidences;
                stats_out->macro_components = 0;
                stats_out->contrast_threshold = 0.15;
            }
        }

    } // namespace bipartite
} // namespace club
