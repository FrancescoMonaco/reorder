#pragma once
// multilevel.hpp — Track P: heavy-edge coarsening + corrected FM.
//
// PaToH-like quality path over micro-clusters:
//   1. IDF window weights (shared vocabulary with Track R).
//   2. Heavy-edge bipartite matching coarsening to a coarse hypergraph.
//   3. Balanced FM with correct gain sign, balance guard, and best-prefix
//      rollback (bucketed max-gain lookup optional).
//   4. Projection + refinement up the hierarchy.
//   5. Strength-ordered expansion + medoid intra-cluster ordering.
//
// keep this header one-directional: it includes reorder.hpp, never reverse.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "club/reorder.hpp"

namespace club {
    namespace multilevel {

        // Coarse hypergraph level produced by heavy-edge matching.
        struct Level {
            std::vector<std::vector<size_t>> members;  // coarse node -> fine nodes
            std::vector<std::vector<size_t>> sig;      // coarse node -> windows
            std::vector<std::vector<size_t>> win2node; // window -> coarse nodes
            std::vector<double> win_w;                 // IDF window weights
            std::vector<size_t> fine_parent;           // fine node -> coarse node
        };

        // IDF weights over an arbitrary node->window incidence.
        inline std::vector<double> idf_weights( const std::vector<std::vector<size_t>>& node_sig,
                                                size_t n_windows ) {
            std::vector<size_t> df( n_windows, 0 );
            for ( const auto& s : node_sig )
                for ( size_t w : s )
                    if ( w < n_windows )
                        ++df[w];
            std::vector<double> wgt( n_windows, 0.0 );
            double n = static_cast<double>( node_sig.size() );
            for ( size_t w = 0; w < n_windows; ++w ) {
                if ( df[w] == 0 )
                    continue;
                wgt[w] = std::log( 1.0 + n / static_cast<double>( df[w] ) );
            }
            return wgt;
        }

        // Heavy-edge matching: greedily merge each unmatched node with the
        // unmatched neighbour maximising shared IDF weight (length-normalised).
        // Deterministic: candidates scanned in increasing node id.
        inline Level coarsen_once( const std::vector<std::vector<size_t>>& node_sig,
                                   size_t n_windows ) {
            const size_t nn = node_sig.size();
            Level lv;
            lv.win_w = idf_weights( node_sig, n_windows );
            if ( nn == 0 )
                return lv;

            std::vector<double> norm( nn, 1.0 );
            for ( size_t i = 0; i < nn; ++i )
                norm[i] = node_sig[i].empty()
                              ? 1.0
                              : std::sqrt( static_cast<double>( node_sig[i].size() ) );

            // Inverted index node windows -> nodes.
            std::vector<std::vector<size_t>> inv( n_windows );
            for ( size_t i = 0; i < nn; ++i )
                for ( size_t w : node_sig[i] )
                    if ( w < n_windows )
                        inv[w].push_back( i );

            std::vector<char> matched( nn, 0 );
            lv.fine_parent.assign( nn, 0 );
            for ( size_t i = 0; i < nn; ++i ) {
                if ( matched[i] ) {
                    lv.fine_parent[i] = lv.members.size() - 1;
                    continue;
                }
                // Score neighbours by shared IDF weight.
                std::unordered_map<size_t, double> score;
                for ( size_t w : node_sig[i] ) {
                    if ( w >= n_windows )
                        continue;
                    for ( size_t j : inv[w] ) {
                        if ( j == i || matched[j] )
                            continue;
                        score[j] += lv.win_w[w] / ( norm[i] * norm[j] );
                    }
                }
                size_t best = nn;
                double best_s = 0.0;
                for ( const auto& kv : score ) {
                    if ( kv.second > best_s + 1e-12 ||
                         ( kv.second == best_s && kv.first < best ) ) {
                        best_s = kv.second;
                        best = kv.first;
                    }
                }
                if ( best == nn || best_s <= 0.0 ) {
                    lv.fine_parent[i] = lv.members.size();
                    lv.members.push_back( { i } );
                    matched[i] = 1;
                } else {
                    lv.fine_parent[i] = lv.members.size();
                    lv.fine_parent[best] = lv.members.size();
                    lv.members.push_back( { i, best } );
                    matched[i] = 1;
                    matched[best] = 1;
                }
            }

            // Union coarse signatures + inverted index.
            lv.sig.resize( lv.members.size() );
            for ( size_t c = 0; c < lv.members.size(); ++c ) {
                std::vector<size_t> u;
                for ( size_t f : lv.members[c] )
                    u.insert( u.end(), node_sig[f].begin(), node_sig[f].end() );
                std::sort( u.begin(), u.end() );
                u.erase( std::unique( u.begin(), u.end() ), u.end() );
                lv.sig[c] = std::move( u );
            }
            lv.win2node.assign( n_windows, {} );
            for ( size_t c = 0; c < lv.sig.size(); ++c )
                for ( size_t w : lv.sig[c] )
                    if ( w < n_windows )
                        lv.win2node[w].push_back( c );
            return lv;
        }

        // Balanced FM on a generic node->window incidence (coarse or fine).
        // Correct gain sign + balance guard + best-prefix rollback.
        inline std::vector<size_t>
        fm_partition( const std::vector<std::vector<size_t>>& node_sig,
                      const std::vector<std::vector<size_t>>& /*win2node*/,
                      size_t n_windows,
                      size_t max_passes = 4,
                      double balance_eps = 0.05 ) {
            const size_t nn = node_sig.size();
            std::vector<size_t> part( nn, 0 );
            if ( nn <= 1 )
                return part;

            std::vector<size_t> order( nn );
            std::iota( order.begin(), order.end(), 0 );
            std::sort( order.begin(), order.end(), [&]( size_t a, size_t b ) {
                if ( node_sig[a].size() != node_sig[b].size() )
                    return node_sig[a].size() > node_sig[b].size();
                return a < b;
            } );
            size_t cnt[2] = { 0, 0 };
            for ( size_t i : order ) {
                part[i] = ( cnt[0] <= cnt[1] ) ? 0 : 1;
                ++cnt[part[i]];
            }

            std::vector<std::array<size_t, 2>> side( n_windows, { 0, 0 } );
            for ( size_t i = 0; i < nn; ++i )
                for ( size_t w : node_sig[i] )
                    if ( w < n_windows )
                        ++side[w][part[i]];

            auto ok_move = [&]( size_t /*i*/, size_t from ) {
                size_t nf = cnt[from] - 1, nt = cnt[1 - from] + 1;
                if ( nf == 0 || nt == 0 )
                    return false;
                double lo = ( 0.5 - balance_eps ) * static_cast<double>( nn );
                double hi = ( 0.5 + balance_eps ) * static_cast<double>( nn );
                return static_cast<double>( nf ) + 1e-9 >= lo &&
                       static_cast<double>( nf ) - 1e-9 <= hi &&
                       static_cast<double>( nt ) + 1e-9 >= lo &&
                       static_cast<double>( nt ) - 1e-9 <= hi;
            };
            auto gain_of = [&]( size_t i ) {
                int g = 0;
                size_t from = part[i], to = 1 - from;
                for ( size_t w : node_sig[i] ) {
                    if ( w >= n_windows )
                        continue;
                    if ( side[w][from] == 1 && side[w][to] > 0 )
                        ++g;
                    else if ( side[w][to] == 0 && side[w][from] > 1 )
                        --g;
                }
                return g;
            };

            std::vector<char> moved( nn, 0 );
            std::vector<size_t> seq;
            seq.reserve( nn );
            for ( size_t pass = 0; pass < max_passes; ++pass ) {
                std::fill( moved.begin(), moved.end(), 0 );
                seq.clear();
                std::vector<size_t> p0 = part;
                std::vector<std::array<size_t, 2>> s0 = side;
                size_t c0[2] = { cnt[0], cnt[1] };
                size_t cur_cut = 0;
                for ( auto& sc : s0 )
                    if ( sc[0] > 0 && sc[1] > 0 )
                        ++cur_cut;
                // Greedy max-gain moves (naive scan: coarse graphs are small).
                for ( size_t t = 0; t < nn; ++t ) {
                    size_t best = nn;
                    int best_g = std::numeric_limits<int>::min();
                    for ( size_t i = 0; i < nn; ++i ) {
                        if ( moved[i] || !ok_move( i, part[i] ) )
                            continue;
                        int g = gain_of( i );
                        if ( best == nn || g > best_g || ( g == best_g && i < best ) ) {
                            best_g = g;
                            best = i;
                        }
                    }
                    if ( best == nn )
                        break;
                    size_t from = part[best], to = 1 - from;
                    for ( size_t w : node_sig[best] ) {
                        if ( w >= n_windows )
                            continue;
                        --side[w][from];
                        ++side[w][to];
                    }
                    --cnt[from];
                    ++cnt[to];
                    part[best] = to;
                    moved[best] = 1;
                    seq.push_back( best );
                }
                if ( seq.empty() )
                    break;
                // Best-prefix rollback.
                std::vector<size_t> rp = p0;
                std::vector<std::array<size_t, 2>> rs = s0;
                size_t cc = cur_cut, best_cut = cur_cut, best_pre = 0;
                for ( size_t t = 0; t < seq.size(); ++t ) {
                    size_t i = seq[t];
                    size_t from = rp[i], to = 1 - from;
                    for ( size_t w : node_sig[i] ) {
                        if ( w >= n_windows )
                            continue;
                        if ( rs[w][from] == 1 && rs[w][to] > 0 )
                            --cc;
                        else if ( rs[w][to] == 0 && rs[w][from] > 1 )
                            ++cc;
                        --rs[w][from];
                        ++rs[w][to];
                    }
                    rp[i] = to;
                    if ( cc < best_cut ) {
                        best_cut = cc;
                        best_pre = t + 1;
                    }
                }
                part = p0;
                side = s0;
                cnt[0] = c0[0];
                cnt[1] = c0[1];
                for ( size_t t = 0; t < best_pre; ++t ) {
                    size_t i = seq[t];
                    size_t from = part[i], to = 1 - from;
                    for ( size_t w : node_sig[i] ) {
                        if ( w >= n_windows )
                            continue;
                        --side[w][from];
                        ++side[w][to];
                    }
                    --cnt[from];
                    ++cnt[to];
                    part[i] = to;
                }
                if ( best_pre == 0 )
                    break;
            }
            return part;
        }

        // FM on a possibly large node set: coarsen with heavy-edge matching down to
        // FM_NODE_CAP nodes first, run FM there (its per-pass cost is O(nn^2 * l),
        // quadratic only in the *coarse* size), then project the partition back up.
        // Quality-neutral for large inputs: the fine-level refinement pass that
        // callers run afterwards still operates at full resolution, and deeper
        // hierarchies are exactly what multilevel partitioning is supposed to do.
        inline std::vector<size_t>
        fm_partition_scalable( const std::vector<std::vector<size_t>>& node_sig,
                               const std::vector<std::vector<size_t>>& win2node,
                               size_t n_windows,
                               size_t max_passes,
                               double balance_eps ) {
            constexpr size_t FM_NODE_CAP = 2048;
            constexpr size_t MAX_LEVELS = 12;
            const size_t nn = node_sig.size();
            if ( nn <= FM_NODE_CAP )
                return fm_partition( node_sig, win2node, n_windows, max_passes, balance_eps );

            std::vector<Level> levels;
            std::vector<std::vector<size_t>> cur = node_sig;
            while ( cur.size() > FM_NODE_CAP && levels.size() < MAX_LEVELS ) {
                Level lv = coarsen_once( cur, n_windows );
                if ( lv.members.size() >= cur.size() )
                    break; // matching stalled; partition what we have
                levels.push_back( std::move( lv ) );
                cur = levels.back().sig;
            }

            std::vector<std::vector<size_t>> inv( n_windows );
            for ( size_t i = 0; i < cur.size(); ++i )
                for ( size_t w : cur[i] )
                    if ( w < n_windows )
                        inv[w].push_back( i );
            std::vector<size_t> part = fm_partition( cur, inv, n_windows, max_passes, balance_eps );

            for ( size_t l = levels.size(); l-- > 0; ) {
                const auto& par = levels[l].fine_parent;
                std::vector<size_t> finer( par.size() );
                for ( size_t i = 0; i < par.size(); ++i )
                    finer[i] = ( par[i] < part.size() ) ? part[par[i]] : 0;
                part.swap( finer );
            }
            return part;
        }

        // Track P entry point: defaults preserve the existing recursive API.
        inline void cluster_multilevel_fm( const CSR<size_t, size_t>& Ahat,
                                           std::vector<size_t>& P,
                                           size_t k = 0,
                                           size_t micro_threshold = 0,
                                           MicroMacroStats* stats_out = nullptr ) {
            const size_t n = Ahat.rows;
            P.resize( n );
            if ( n == 0 )
                return;
            if ( k == 0 )
                k = 8;
            k = std::min( k, n );

            MacroGraph graph = build_macro_graph( Ahat, micro_threshold, 0.15 );
            const size_t nm = graph.n_micro;
            if ( nm == 0 ) {
                P = graph.exhausted_rows;
                return;
            }

            // Initial 2-way split, scalable: coarsens deep internally, FM on the
            // coarsest level, projects back. (Previously: one coarsen pass + FM on
            // the resulting ~nm/2-node graph -- still quadratic for big matrices.)
            std::vector<size_t> fine_part = fm_partition_scalable(
                graph.macro_sig, graph.window_to_clusters, graph.n_windows, 4, 0.05 );
            // One projected refinement round at fine level.
            std::vector<std::array<size_t, 2>> side( graph.n_windows, { 0, 0 } );
            for ( size_t i = 0; i < nm; ++i )
                for ( size_t w : graph.macro_sig[i] )
                    ++side[w][fine_part[i]];
            size_t cnt[2] = { 0, 0 };
            for ( size_t v : fine_part )
                ++cnt[v];
            for ( size_t iter = 0; iter < 3; ++iter ) {
                bool improved = false;
                for ( size_t i = 0; i < nm; ++i ) {
                    size_t from = fine_part[i], to = 1 - from;
                    if ( cnt[from] <= 1 )
                        continue;
                    double lo = 0.45 * static_cast<double>( nm );
                    double hi = 0.55 * static_cast<double>( nm );
                    if ( static_cast<double>( cnt[from] - 1 ) < lo - 1e-9 ||
                         static_cast<double>( cnt[to] + 1 ) > hi + 1e-9 )
                        continue;
                    int g = 0;
                    for ( size_t w : graph.macro_sig[i] ) {
                        if ( side[w][from] == 1 && side[w][to] > 0 )
                            ++g;
                        else if ( side[w][to] == 0 && side[w][from] > 1 )
                            --g;
                    }
                    if ( g > 0 ) {
                        for ( size_t w : graph.macro_sig[i] ) {
                            --side[w][from];
                            ++side[w][to];
                        }
                        --cnt[from];
                        ++cnt[to];
                        fine_part[i] = to;
                        improved = true;
                    }
                }
                if ( !improved )
                    break;
            }

            // k-way: repeatedly split the largest group with restricted FM.
            std::vector<size_t> group_id( nm, 0 );
            for ( size_t i = 0; i < nm; ++i )
                group_id[i] = fine_part[i];
            size_t n_groups = 2;
            while ( n_groups < k ) {
                std::unordered_map<size_t, size_t> gsize;
                for ( size_t v : group_id )
                    ++gsize[v];
                size_t target = 0, target_sz = 0;
                for ( auto& kv : gsize ) {
                    if ( kv.second > target_sz ||
                         ( kv.second == target_sz && kv.first < target ) ) {
                        target_sz = kv.second;
                        target = kv.first;
                    }
                }
                if ( target_sz <= 1 )
                    break;
                std::vector<size_t> members;
                for ( size_t i = 0; i < nm; ++i )
                    if ( group_id[i] == target )
                        members.push_back( i );
                std::vector<std::vector<size_t>> sub_sig( members.size() );
                for ( size_t t = 0; t < members.size(); ++t )
                    sub_sig[t] = graph.macro_sig[members[t]];
                std::vector<std::vector<size_t>> sub_inv( graph.n_windows );
                for ( size_t t = 0; t < members.size(); ++t )
                    for ( size_t w : sub_sig[t] )
                        sub_inv[w].push_back( t );
                std::vector<size_t> sub_part =
                    fm_partition_scalable( sub_sig, sub_inv, graph.n_windows, 3, 0.10 );
                size_t c0 = 0;
                for ( size_t v : sub_part )
                    if ( v == 0 )
                        ++c0;
                if ( c0 == 0 || c0 == sub_part.size() )
                    break;
                for ( size_t t = 0; t < members.size(); ++t )
                    if ( sub_part[t] == 1 )
                        group_id[members[t]] = n_groups;
                ++n_groups;
            }

            // Order groups largest-first; clusters within groups greedily.
            std::vector<double> win_w = idf_weights( graph.macro_sig, graph.n_windows );
            std::unordered_map<size_t, std::vector<size_t>> groups;
            for ( size_t i = 0; i < nm; ++i )
                groups[group_id[i]].push_back( i );
            std::vector<std::vector<size_t>> glist;
            for ( auto& kv : groups )
                glist.push_back( std::move( kv.second ) );
            std::sort( glist.begin(), glist.end(), []( const auto& a, const auto& b ) {
                if ( a.size() != b.size() )
                    return a.size() > b.size();
                return a.front() < b.front();
            } );
            std::vector<size_t> cluster_order;
            cluster_order.reserve( nm );
            // Global buffers with touched-list resets: per-group cost is now
            // O(|members| log |members| + expansion), not O(nm + |members|^2).
            // The lazy max-heap is exact here: score[] only increases (commit adds
            // weight), so stale entries are detected by comparing against score[].
            std::vector<char> placed_all( nm, 0 );
            std::vector<char> in_g( nm, 0 );
            std::vector<double> score( nm, 0.0 );
            std::vector<char> win_seen( graph.n_windows, 0 );
            std::vector<size_t> touched, touched_win;
            struct Entry {
                double s;
                size_t c;
            };
            auto cmp = []( const Entry& a, const Entry& b ) {
                if ( a.s != b.s )
                    return a.s < b.s;
                return a.c > b.c;
            };
            std::priority_queue<Entry, std::vector<Entry>, decltype( cmp )> pq( cmp );
            std::vector<double> wdeg( nm, 0.0 );
            for ( size_t c = 0; c < nm; ++c )
                for ( size_t w : graph.macro_sig[c] )
                    wdeg[c] += win_w[w];
            for ( auto& members : glist ) {
                size_t seed = members[0];
                for ( size_t c : members )
                    if ( wdeg[c] < wdeg[seed] - 1e-12 || ( wdeg[c] == wdeg[seed] && c < seed ) )
                        seed = c;
                touched.clear();
                touched_win.clear();
                for ( size_t c : members ) {
                    in_g[c] = 1;
                    touched.push_back( c );
                }
                auto commit = [&]( size_t c ) {
                    cluster_order.push_back( c );
                    placed_all[c] = 1;
                    for ( size_t w : graph.macro_sig[c] ) {
                        if ( win_seen[w] )
                            continue;
                        win_seen[w] = 1;
                        touched_win.push_back( w );
                        for ( size_t c2 : graph.window_to_clusters[w] ) {
                            if ( c2 < nm && in_g[c2] && !placed_all[c2] ) {
                                score[c2] += win_w[w];
                                touched.push_back( c2 );
                                pq.push( { score[c2], c2 } );
                            }
                        }
                    }
                };
                commit( seed );
                for ( size_t t = 1; t < members.size(); ++t ) {
                    size_t best = nm;
                    while ( !pq.empty() ) {
                        Entry e = pq.top();
                        pq.pop();
                        if ( placed_all[e.c] )
                            continue;
                        if ( e.s < score[e.c] - 1e-12 )
                            continue; // stale
                        best = e.c;
                        break;
                    }
                    if ( best == nm ) {
                        // Frontier exhausted: lowest-id unplaced member (matches the
                        // old all-scores-equal tie-break).
                        for ( size_t c : members ) {
                            if ( !placed_all[c] ) {
                                best = c;
                                break;
                            }
                        }
                        if ( best == nm )
                            break; // defensive; all placed
                    }
                    commit( best );
                }
                // Reset only what this group touched.
                for ( size_t c : touched )
                    score[c] = 0.0;
                for ( size_t c : members )
                    in_g[c] = 0;
                for ( size_t w : touched_win )
                    win_seen[w] = 0;
                // Drop any leftover heap entries for next group.
                while ( !pq.empty() )
                    pq.pop();
            }

            // Medoid intra-cluster ordering on Ahat rows.
            // Sampled medoid candidates + precomputed distance keys: linear in |grp|
            // (was O(|grp|^2 * l) for the search plus an O(l) recomputation inside
            // every sort comparator).
            for ( auto& grp : graph.micro_groups ) {
                const size_t sz = grp.size();
                if ( sz <= 2 )
                    continue;
                auto jd = [&]( size_t a, size_t b ) -> double {
                    size_t sa = Ahat.row_ptr[a], ea = Ahat.row_ptr[a + 1];
                    size_t sb = Ahat.row_ptr[b], eb = Ahat.row_ptr[b + 1];
                    size_t na = ea - sa, nb = eb - sb;
                    if ( na == 0 && nb == 0 )
                        return 0.0;
                    if ( na == 0 || nb == 0 )
                        return 1.0;
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
                    return 1.0 - static_cast<double>( is ) / static_cast<double>( na + nb - is );
                };
                constexpr size_t MEDOID_SAMPLES = 24;
                size_t ncand = std::min( MEDOID_SAMPLES, sz );
                size_t stride = sz / ncand;
                size_t med = grp[0];
                double best = std::numeric_limits<double>::infinity();
                for ( size_t ci = 0; ci < ncand; ++ci ) {
                    size_t r = grp[ci * stride];
                    double acc = 0.0;
                    size_t nref = std::min( MEDOID_SAMPLES, sz );
                    for ( size_t ri = 0; ri < nref; ++ri ) {
                        size_t o = grp[ri * ( sz / nref )];
                        if ( o != r )
                            acc += jd( r, o );
                    }
                    if ( acc < best ) {
                        best = acc;
                        med = r;
                    }
                }
                std::vector<double> dist( sz, 0.0 );
                for ( size_t t = 0; t < sz; ++t )
                    dist[t] = ( grp[t] == med ) ? -1.0 : jd( med, grp[t] );
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
            }

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

    } // namespace multilevel
} // namespace club
