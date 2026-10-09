#pragma once
// refine.hpp — Track S: exact stripe-gain local search (row-only).
// Objective (matches ANALYSIS/analysis_utils.py block_density, nnz invariant
// under row perms):
//   cost_bs(P) = sum over stripes S of bs rows of |union windows|
// Minimizing cost_bs maximizes density(bs). Caller builds Ahat with W == bs
// so the delta matches the measured metric at that scale.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "club/reorder.hpp"

namespace club {
    namespace refine {

        // Stripe occupancy table for one block size.
        //
        // SPARSE representation: per-stripe sorted unique window list with counts.
        // The previous dense layout (nstripes x nwin ints) needed ~39 GB for
        // thermal2 at bs=16 (77k stripes x 126k windows) and was untouchable at
        // bs=4 (~150 GB), which made the fine-grained (W==bs) passes the
        // bottleneck both in time (page-walks during init's cost scan) and memory.
        // The sparse layout costs O(unique (stripe,window) pairs) <= O(nnz(Ahat))
        // memory (~100 MB for thermal2 at bs=4) and builds in O(nnz log u).
        //
        // Lookups during parallel scoring are read-only binary searches
        // (count_at), so the parallel phase keeps its race-free property.
        // Mutations (apply_remove/apply_add) happen only in the sequential
        // commit phase and cost O(u_s) per touched stripe via memmove.
        struct StripeTable {
            size_t bs = 32;
            size_t nstripes = 0;
            size_t nwin = 0;
            // PER-STRIPE storage: each stripe owns its sorted unique window list and
            // parallel counts. Mutations touch only the affected stripe's small
            // vector (O(u_s), u_s <= bs * avg_deg) — never a global tail shift plus
            // an O(nstripes) offset walk, which made the previous flattened layout
            // quadratic on thermal2-class matrices (~nnz/2 element moves per row
            // move). Parallel scoring is unchanged: read-only binary searches.
            std::vector<std::vector<size_t>> win; // win[s]: sorted unique windows of stripe s
            std::vector<std::vector<int>> cnt;    // cnt[s]: parallel occurrence counts
            std::vector<size_t> cost;             // unique windows per stripe == win[s].size()
            size_t total = 0;
            // Scratch buffers for batched mutations (sequential commit phase only).
            std::vector<size_t> tmp_pos;  // delete / insert positions (ascending)
            std::vector<size_t> tmp_miss; // missing windows to insert
            std::vector<size_t> scratch_win;
            std::vector<int> scratch_cnt;

            StripeTable() = default;
            StripeTable( const CSR<size_t, size_t>& Ahat,
                         const std::vector<size_t>& P,
                         size_t bs_ ) {
                init( Ahat, P, bs_ );
            }
            void init( const CSR<size_t, size_t>& Ahat, const std::vector<size_t>& P, size_t bs_ ) {
                bs = bs_;
                const size_t n = P.size();
                nwin = Ahat.cols;
                nstripes = ( n + bs - 1 ) / bs;

                // Group rows by stripe (exact counting sort: every row sits at a
                // known position pos, hence stripe pos/bs).
                std::vector<size_t> srow_off( nstripes + 1, 0 );
                for ( size_t pos = 0; pos < n; ++pos )
                    ++srow_off[pos / bs + 1];
                for ( size_t s = 0; s < nstripes; ++s )
                    srow_off[s + 1] += srow_off[s];
                std::vector<size_t> srow( n );
                {
                    std::vector<size_t> cursor( srow_off.begin(), srow_off.end() - 1 );
                    for ( size_t pos = 0; pos < n; ++pos ) {
                        size_t s = pos / bs;
                        srow[cursor[s]++] = P[pos];
                    }
                }

                // Build each stripe's sorted (window,count) list in parallel.
                std::vector<std::vector<size_t>> tmp_win( nstripes );
                std::vector<std::vector<int>> tmp_cnt( nstripes );
#pragma omp parallel for schedule( dynamic )
                for ( size_t s = 0; s < nstripes; ++s ) {
                    std::vector<size_t> all;
                    size_t deg_sum = 0;
                    for ( size_t t = srow_off[s]; t < srow_off[s + 1]; ++t ) {
                        size_t r = srow[t];
                        deg_sum += Ahat.row_ptr[r + 1] - Ahat.row_ptr[r];
                    }
                    all.reserve( deg_sum );
                    for ( size_t t = srow_off[s]; t < srow_off[s + 1]; ++t ) {
                        size_t r = srow[t];
                        all.insert( all.end(),
                                    &Ahat.col_ind[Ahat.row_ptr[r]],
                                    &Ahat.col_ind[Ahat.row_ptr[r + 1]] );
                    }
                    std::sort( all.begin(), all.end() );
                    auto& wv = tmp_win[s];
                    auto& cv = tmp_cnt[s];
                    size_t i = 0;
                    while ( i < all.size() ) {
                        size_t j = i;
                        while ( j < all.size() && all[j] == all[i] )
                            ++j;
                        wv.push_back( all[i] );
                        cv.push_back( static_cast<int>( j - i ) );
                        i = j;
                    }
                }

                // Store per-stripe lists directly (no flattening, no offsets).
                win.resize( nstripes );
                cnt.resize( nstripes );
                cost.assign( nstripes, 0 );
                total = 0;
#pragma omp parallel for schedule( static ) reduction( + : total )
                for ( size_t s = 0; s < nstripes; ++s ) {
                    win[s] = std::move( tmp_win[s] );
                    cnt[s] = std::move( tmp_cnt[s] );
                    cost[s] = win[s].size();
                    total += cost[s];
                }
            }

            // Occurrence count of window w in stripe s (0 if absent).
            inline int count_at( size_t s, size_t w ) const {
                const std::vector<size_t>& v = win[s];
                const size_t u = v.size();
                size_t a = 0, b = u;
                while ( a < b ) {
                    size_t m = a + ( b - a ) / 2;
                    if ( v[m] < w )
                        a = m + 1;
                    else
                        b = m;
                }
                return ( a < u && v[a] == w ) ? cnt[s][a] : 0;
            }
            int remove_delta( size_t r, size_t s, const CSR<size_t, size_t>& Ahat ) const {
                int d = 0;
                for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k )
                    if ( count_at( s, Ahat.col_ind[k] ) == 1 )
                        --d;
                return d;
            }
            int add_delta( size_t r, size_t t, const CSR<size_t, size_t>& Ahat ) const {
                int d = 0;
                for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k )
                    if ( count_at( t, Ahat.col_ind[k] ) == 0 )
                        ++d;
                return d;
            }
            void apply_remove( size_t r, size_t s, const CSR<size_t, size_t>& Ahat ) {
                std::vector<size_t>& v = win[s];
                std::vector<int>& c = cnt[s];
                const size_t hi = v.size();
                // First pass: decrement counts, collect positions whose count hits 0
                // (row windows are unique within the row, so positions are ascending).
                tmp_pos.clear();
                for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k ) {
                    size_t w = Ahat.col_ind[k];
                    size_t a = 0, b = hi;
                    while ( a < b ) {
                        size_t m = a + ( b - a ) / 2;
                        if ( v[m] < w )
                            a = m + 1;
                        else
                            b = m;
                    }
                    if ( a < hi && v[a] == w ) {
                        if ( --c[a] == 0 )
                            tmp_pos.push_back( a );
                    }
                }
                const size_t ndel = tmp_pos.size();
                if ( ndel == 0 )
                    return;
                // Compact the stripe's own vector in one pass (skip deleted entries).
                // Cost O(u_s): no global tail move, no offset bookkeeping.
                size_t write = 0, p = 0;
                for ( size_t i = 0; i < hi; ++i ) {
                    if ( p < ndel && i == tmp_pos[p] ) {
                        ++p;
                        continue;
                    }
                    v[write] = v[i];
                    c[write] = c[i];
                    ++write;
                }
                v.resize( write );
                c.resize( write );
                cost[s] -= ndel;
            }
            void apply_add( size_t r, size_t t, const CSR<size_t, size_t>& Ahat ) {
                std::vector<size_t>& v = win[t];
                std::vector<int>& c = cnt[t];
                const size_t hi = v.size();
                // First pass: bump counts of present windows, record missing ones with
                // their insert positions (ascending, since row windows are sorted;
                // several missing windows may share one insertion slot — they are
                // consumed in ascending window order, preserving sortedness).
                tmp_pos.clear();
                tmp_miss.clear();
                for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k ) {
                    size_t w = Ahat.col_ind[k];
                    size_t a = 0, b = hi;
                    while ( a < b ) {
                        size_t m = a + ( b - a ) / 2;
                        if ( v[m] < w )
                            a = m + 1;
                        else
                            b = m;
                    }
                    if ( a < hi && v[a] == w ) {
                        ++c[a];
                    } else {
                        tmp_pos.push_back( a );
                        tmp_miss.push_back( w );
                    }
                }
                const size_t nadd = tmp_miss.size();
                if ( nadd == 0 )
                    return;
                // Merge the stripe's own vector with the missing windows into scratch,
                // then swap in. Cost O(u_s + nadd): no global tail move, no offsets.
                scratch_win.resize( hi + nadd );
                scratch_cnt.resize( hi + nadd );
                size_t ip = 0, sp = 0;
                for ( size_t i = 0; i < hi; ++i ) {
                    while ( ip < nadd && tmp_pos[ip] == i ) {
                        scratch_win[sp] = tmp_miss[ip];
                        scratch_cnt[sp] = 1;
                        ++sp;
                        ++ip;
                    }
                    scratch_win[sp] = v[i];
                    scratch_cnt[sp] = c[i];
                    ++sp;
                }
                while ( ip < nadd ) { // windows larger than every segment entry
                    scratch_win[sp] = tmp_miss[ip];
                    scratch_cnt[sp] = 1;
                    ++sp;
                    ++ip;
                }
                v.assign( scratch_win.begin(), scratch_win.begin() + static_cast<long>( sp ) );
                c.assign( scratch_cnt.begin(), scratch_cnt.begin() + static_cast<long>( sp ) );
                cost[t] += nadd;
            }
            void recompute_total() {
                total = 0;
                for ( size_t c : cost )
                    total += c;
            }
        };

        // col_run_table: CSR column-run bookkeeping for the lexicographic
        // tie-breaker. For each column j, rows are stored sorted; consecutive runs
        // (r, r+1 both in column j under current P) are the vertical-adjacency
        // pairs the paper correlates with speedup. Maintaining full run tables
        // incrementally under swaps is O(degree); we only need DELTAS, computed
        // on the fly from position maps (see vertical_delta below).
        struct ColRuns {
            // col_rows[j] = sorted positions (in P-order) of rows touching column j.
            // Built once per refine call from A (scalar CSR) + P. Memory O(nnz).
            std::vector<std::vector<int>> col_rows;
            size_t ncols = 0;

            ColRuns() = default;
            template <typename DataT, typename intT>
            ColRuns( const CSR<DataT, intT>& A, const std::vector<size_t>& P ) {
                init( A, P );
            }
            template <typename DataT, typename intT>
            void init( const CSR<DataT, intT>& A, const std::vector<size_t>& P ) {
                const size_t n = P.size();
                ncols = static_cast<size_t>( A.cols );
                std::vector<size_t> pos_of( n );
                for ( size_t pos = 0; pos < n; ++pos )
                    pos_of[P[pos]] = pos;
                col_rows.assign( ncols, {} );
                for ( size_t r = 0; r < n; ++r ) {
                    int p = static_cast<int>( pos_of[r] );
                    for ( intT k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k )
                        col_rows[A.col_ind[k]].push_back( p );
                }
                for ( auto& v : col_rows )
                    std::sort( v.begin(), v.end() );
            }
            // Count of consecutive-position pairs (p, p+1) both present.
            size_t total_runs() const {
                size_t t = 0;
                for ( const auto& v : col_rows )
                    for ( size_t i = 1; i < v.size(); ++i )
                        if ( v[i] == v[i - 1] + 1 )
                            ++t;
                return t;
            }
        };

        // vertical_delta: change in consecutive-position run count if rows at pos
        // and tp are swapped. Uses scalar CSR A + pos_of map. Only columns touched
        // by r or r2 can change run status, and only at boundaries {pos-1,pos,pos+1,
        // tp-1,tp,tp+1}. O(|A[r]| + |A[r2]|).
        template <typename DataT, typename intT>
        inline int vertical_delta( const CSR<DataT, intT>& A,
                                   const std::vector<size_t>& P,
                                   const std::vector<size_t>& pos_of,
                                   size_t pos,
                                   size_t tp ) {
            if ( pos == tp )
                return 0;
            const size_t n = P.size();
            size_t r = P[pos], r2 = P[tp];
            // Collect affected (column, position-boundary) checks: for each column in
            // r/r2, check adjacency pairs (q, q+1) for q in {pos-1, pos, tp-1, tp}.
            // A pair (q,q+1) is a run iff rows at P-positions q and q+1 both touch j.
            // Build row-sets for O(1) membership: mark columns of r and r2 via
            // small hash (linear scan is fine: degrees are small).
            auto touches = [&]( size_t row, intT j ) -> bool {
                for ( intT k = A.row_ptr[row]; k < A.row_ptr[row + 1]; ++k )
                    if ( A.col_ind[k] == j )
                        return true;
                return false;
            };
            int delta = 0;
            // Union of columns of r and r2.
            std::vector<intT> cols;
            cols.reserve( ( A.row_ptr[r + 1] - A.row_ptr[r] ) +
                          ( A.row_ptr[r2 + 1] - A.row_ptr[r2] ) );
            for ( intT k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k )
                cols.push_back( A.col_ind[k] );
            for ( intT k = A.row_ptr[r2]; k < A.row_ptr[r2 + 1]; ++k )
                cols.push_back( A.col_ind[k] );
            std::sort( cols.begin(), cols.end() );
            cols.erase( std::unique( cols.begin(), cols.end() ), cols.end() );
            size_t qs[4] = { pos > 0 ? pos - 1 : n, pos, tp > 0 ? tp - 1 : n, tp };
            for ( intT j : cols ) {
                for ( int qi = 0; qi < 4; ++qi ) {
                    size_t q = qs[qi];
                    if ( q >= n - 1 )
                        continue;
                    // rows currently at q, q+1
                    size_t rq0 = P[q], rq1 = P[q + 1];
                    bool before = touches( rq0, j ) && touches( rq1, j );
                    // rows after swap of pos<->tp
                    size_t nq0 = ( q == pos ) ? r2 : ( ( q == tp ) ? r : rq0 );
                    size_t nq1 = ( q + 1 == pos ) ? r2 : ( ( q + 1 == tp ) ? r : rq1 );
                    bool after = touches( nq0, j ) && touches( nq1, j );
                    delta += ( after ? 1 : 0 ) - ( before ? 1 : 0 );
                }
            }
            (void)pos_of;
            return delta;
        }

        // swap_pass (legacy sequential): try swapping each row with the boundary
        // slot of neighbouring stripes (+/- window_stripes). Exact net delta via
        // remove/add composition. Returns # accepted swaps. Kept for reference;
        // stripe_refine() below uses the parallel two-phase variant.
        inline size_t swap_pass( const CSR<size_t, size_t>& Ahat,
                                 std::vector<size_t>& P,
                                 StripeTable& tab,
                                 size_t window_stripes = 3 ) {
            const size_t n = P.size();
            size_t accepted = 0;
            for ( size_t pos = 0; pos < n; ++pos ) {
                size_t r = P[pos];
                size_t s = pos / tab.bs;
                size_t lo = ( s >= window_stripes ) ? s - window_stripes : 0;
                size_t hi = std::min( tab.nstripes - 1, s + window_stripes );
                for ( size_t t = lo; t <= hi; ++t ) {
                    if ( t == s )
                        continue;
                    size_t tpos = std::min( t * tab.bs + tab.bs - 1, n - 1 );
                    if ( tpos == pos )
                        continue;
                    size_t r2 = P[tpos];
                    size_t s2 = tpos / tab.bs;
                    if ( s2 == s )
                        continue;
                    int dr_out = tab.remove_delta( r, s, Ahat );
                    int dr2_out = tab.remove_delta( r2, s2, Ahat );
                    tab.apply_remove( r, s, Ahat );
                    tab.apply_remove( r2, s2, Ahat );
                    int dr_in = tab.add_delta( r2, s, Ahat );
                    int dr2_in = tab.add_delta( r, s2, Ahat );
                    int net = dr_out + dr2_out + dr_in + dr2_in;
                    tab.apply_add( r, s, Ahat );
                    tab.apply_add( r2, s2, Ahat );
                    if ( net < 0 ) {
                        tab.apply_remove( r, s, Ahat );
                        tab.apply_remove( r2, s2, Ahat );
                        tab.apply_add( r, s2, Ahat );
                        tab.apply_add( r2, s, Ahat );
                        std::swap( P[pos], P[tpos] );
                        ++accepted;
                        break;
                    }
                }
            }
            tab.recompute_total();
            return accepted;
        }

        // stripe_refine(Ahat with W==bs, P, bs, max_passes, window_stripes).
        // Swap passes to convergence. Per pass O(n * window_stripes * avg_deg).
        // Parallel delta evaluation: candidate swaps for different stripes are
        // evaluated concurrently (read-only on tab), then accepted swaps are applied
        // sequentially in deterministic order. Gives ~cores speedup on the scoring
        // loop, which dominates for large n.
        // stripe_refine with lexicographic (primary, secondary) objective:
        //   primary: stripe-union cost (== nonzero blocks; larger = worse)
        //   secondary: -vertical_runs (more consecutive runs = better), weight 1 ladder
        // Accept a swap iff (net_blocks < 0) OR (net_blocks == 0 AND vert_delta > 0).
        // This keeps density strictly primary (never trades a block for locality)
        // while letting vertical adjacency break plateaus — the exact mechanism by
        // which Rabbit beats pure block-count search on sparse low-degree matrices.
        // A_for_runs: scalar CSR of the (scrambled) matrix for run computation; if
        // nullptr, secondary is disabled (pure block-count search).
        template <typename DataT = float, typename intT = int>
        inline size_t stripe_refine_lex( const CSR<size_t, size_t>& Ahat,
                                         std::vector<size_t>& P,
                                         size_t bs,
                                         const CSR<DataT, intT>* A_for_runs,
                                         size_t max_passes = 6,
                                         size_t window_stripes = 3 ) {
            const size_t n = P.size();
            if ( n == 0 )
                return 0;
            StripeTable tab( Ahat, P, bs );
            std::vector<size_t> pos_of( n );
            for ( size_t pos = 0; pos < n; ++pos )
                pos_of[P[pos]] = pos;
            struct Cand {
                size_t pos = 0;
                size_t tpos = 0;
                int net = 0;
            };
            size_t total_accepted = 0;
            for ( size_t pass = 0; pass < max_passes; ++pass ) {
                std::vector<Cand> best( n );
#pragma omp parallel for schedule( static )
                for ( size_t pos = 0; pos < n; ++pos ) {
                    size_t r = P[pos];
                    size_t s = pos / tab.bs;
                    size_t lo = ( s >= window_stripes ) ? s - window_stripes : 0;
                    size_t hi = std::min( tab.nstripes - 1, s + window_stripes );
                    Cand bc;
                    bc.pos = pos;
                    bc.tpos = pos;
                    bc.net = 0;
                    for ( size_t t = lo; t <= hi; ++t ) {
                        if ( t == s )
                            continue;
                        size_t tp = std::min( t * tab.bs + tab.bs - 1, n - 1 );
                        if ( tp == pos )
                            continue;
                        size_t r2 = P[tp];
                        size_t s2 = tp / tab.bs;
                        if ( s2 == s )
                            continue;
                        int d = 0;
                        for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( tab.count_at( s, w ) == 1 )
                                --d;
                            if ( tab.count_at( s2, w ) == 0 )
                                ++d;
                        }
                        for ( size_t k = Ahat.row_ptr[r2]; k < Ahat.row_ptr[r2 + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( tab.count_at( s2, w ) == 1 )
                                --d;
                            if ( tab.count_at( s, w ) == 0 )
                                ++d;
                        }
                        if ( d < bc.net ) {
                            bc.net = d;
                            bc.tpos = tp;
                        }
                    }
                    best[pos] = bc;
                }
                size_t acc = 0;
                std::vector<char> moved( n, 0 );
                for ( size_t pos = 0; pos < n; ++pos ) {
                    size_t tp = best[pos].tpos;
                    int approx = best[pos].net;
                    // Consider plateau swaps too when runs are enabled: approx==0 may
                    // still win on the secondary. Otherwise only strict improvements.
                    if ( moved[pos] || moved[tp] )
                        continue;
                    if ( approx > 0 )
                        continue;
                    if ( approx == 0 && A_for_runs == nullptr )
                        continue;
                    if ( approx == 0 && best[pos].tpos == pos )
                        continue;
                    size_t r = P[pos];
                    size_t r2 = P[tp];
                    size_t s = pos / tab.bs;
                    size_t s2 = tp / tab.bs;
                    if ( s2 == s )
                        continue;
                    int dr_out = tab.remove_delta( r, s, Ahat );
                    int dr2_out = tab.remove_delta( r2, s2, Ahat );
                    tab.apply_remove( r, s, Ahat );
                    tab.apply_remove( r2, s2, Ahat );
                    int dr_in = tab.add_delta( r2, s, Ahat );
                    int dr2_in = tab.add_delta( r, s2, Ahat );
                    int net = dr_out + dr2_out + dr_in + dr2_in;
                    tab.apply_add( r, s, Ahat );
                    tab.apply_add( r2, s2, Ahat );
                    bool accept = ( net < 0 );
                    if ( !accept && net == 0 && A_for_runs != nullptr ) {
                        int vd = vertical_delta( *A_for_runs, P, pos_of, pos, tp );
                        accept = ( vd > 0 );
                    }
                    if ( accept ) {
                        tab.apply_remove( r, s, Ahat );
                        tab.apply_remove( r2, s2, Ahat );
                        tab.apply_add( r, s2, Ahat );
                        tab.apply_add( r2, s, Ahat );
                        std::swap( P[pos], P[tp] );
                        pos_of[r] = tp;
                        pos_of[r2] = pos;
                        moved[pos] = 1;
                        moved[tp] = 1;
                        ++acc;
                    }
                }
                tab.recompute_total();
                total_accepted += acc;
                if ( acc == 0 )
                    break;
            }
            return total_accepted;
        }

        inline size_t stripe_refine( const CSR<size_t, size_t>& Ahat,
                                     std::vector<size_t>& P,
                                     size_t bs,
                                     size_t max_passes = 6,
                                     size_t window_stripes = 3 ) {
            const size_t n = P.size();
            if ( n == 0 )
                return 0;
            StripeTable tab( Ahat, P, bs );
            size_t total_accepted = 0;
            // Candidate struct for parallel scoring.
            struct Cand {
                size_t pos = 0;
                size_t tpos = 0;
                int net = 0;
            };
            for ( size_t pass = 0; pass < max_passes; ++pass ) {
                // Phase 1: score best swap per row in parallel (read-only on tab).
                std::vector<Cand> best( n );
#pragma omp parallel for schedule( static )
                for ( size_t pos = 0; pos < n; ++pos ) {
                    size_t r = P[pos];
                    size_t s = pos / tab.bs;
                    size_t lo = ( s >= window_stripes ) ? s - window_stripes : 0;
                    size_t hi = std::min( tab.nstripes - 1, s + window_stripes );
                    Cand bc;
                    bc.pos = pos;
                    bc.tpos = pos;
                    bc.net = 0;
                    for ( size_t t = lo; t <= hi; ++t ) {
                        if ( t == s )
                            continue;
                        size_t tp = std::min( t * tab.bs + tab.bs - 1, n - 1 );
                        if ( tp == pos )
                            continue;
                        size_t r2 = P[tp];
                        size_t s2 = tp / tab.bs;
                        if ( s2 == s )
                            continue;
                        // Read-only delta: windows leaving/entering stripe unions.
                        // remove part: window disappears iff count==1 in home stripe.
                        // add part: window appears iff count==0 in target stripe,
                        //   EXCEPT windows of r also present in s2 via r2's twin...
                        // Exactness note: because stripes are disjoint, the composed
                        // remove+add delta below is exact for the swap of r and r2.
                        int d = 0;
                        for ( size_t k = Ahat.row_ptr[r]; k < Ahat.row_ptr[r + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( tab.count_at( s, w ) == 1 )
                                --d;
                            if ( tab.count_at( s2, w ) == 0 )
                                ++d;
                        }
                        for ( size_t k = Ahat.row_ptr[r2]; k < Ahat.row_ptr[r2 + 1]; ++k ) {
                            size_t w = Ahat.col_ind[k];
                            if ( tab.count_at( s2, w ) == 1 )
                                --d;
                            if ( tab.count_at( s, w ) == 0 )
                                ++d;
                        }
                        // Correction for shared windows between r and r2: if both
                        // touch w, the naive sum double-counts. Windows in
                        // intersection(r, r2) that are unique in both stripes
                        // contribute -2/+2 correctly already; windows present
                        // elsewhere in s/s2 are unaffected by either move. The only
                        // inexact case is w in both r and r2 with cnt==1 in s and
                        // cnt==0 in s2 (or vice versa): after swap w still leaves s
                        // (r2 carries it) — but our d counts -1 for r leaving and
                        // +0 for r2 arriving... approximate; re-verified exactly at
                        // commit time below. Keep best by approximate score.
                        if ( d < bc.net ) {
                            bc.net = d;
                            bc.tpos = tp;
                        }
                    }
                    best[pos] = bc;
                }
                // Phase 2: commit in deterministic order with exact re-evaluation.
                size_t acc = 0;
                std::vector<char> moved( n, 0 );
                for ( size_t pos = 0; pos < n; ++pos ) {
                    if ( best[pos].net >= 0 || moved[pos] )
                        continue;
                    size_t tp = best[pos].tpos;
                    if ( moved[tp] )
                        continue;
                    size_t r = P[pos];
                    size_t r2 = P[tp];
                    size_t s = pos / tab.bs;
                    size_t s2 = tp / tab.bs;
                    if ( s2 == s )
                        continue;
                    int dr_out = tab.remove_delta( r, s, Ahat );
                    int dr2_out = tab.remove_delta( r2, s2, Ahat );
                    tab.apply_remove( r, s, Ahat );
                    tab.apply_remove( r2, s2, Ahat );
                    int dr_in = tab.add_delta( r2, s, Ahat );
                    int dr2_in = tab.add_delta( r, s2, Ahat );
                    int net = dr_out + dr2_out + dr_in + dr2_in;
                    tab.apply_add( r, s, Ahat );
                    tab.apply_add( r2, s2, Ahat );
                    if ( net < 0 ) {
                        tab.apply_remove( r, s, Ahat );
                        tab.apply_remove( r2, s2, Ahat );
                        tab.apply_add( r, s2, Ahat );
                        tab.apply_add( r2, s, Ahat );
                        std::swap( P[pos], P[tp] );
                        moved[pos] = 1;
                        moved[tp] = 1;
                        ++acc;
                    }
                }
                tab.recompute_total();
                total_accepted += acc;
                if ( acc == 0 )
                    break;
            }
            return total_accepted;
        }

    } // namespace refine
} // namespace club
