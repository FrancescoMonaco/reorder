#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <icecream.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __AVX512F__
    #include <immintrin.h>
#endif

#include "club/expect.hpp"
#include "club/matrices.hpp"
#include "club/metrics.hpp"
#include "club/simd.hpp"
#include "omp.h"

namespace club {

    // Apply the masking to the input matrix. To build Ahat each row v of A is divided into windows
    // of size W Each masked row vhat has a 1 for every non-zero window in v. Overloaded for CSR and
    // BSR, the output Ahat is always in CSR format.
    template <typename DataT = float, typename intT = int>
    void mask( CSR<DataT, intT>& A, size_t W, CSR<size_t, size_t>& Ahat ) {
        Ahat.rows = static_cast<size_t>( A.rows );
        Ahat.cols = static_cast<size_t>( ( A.cols + W - 1 ) / W );

        // Safely resize vector metadata containers
        Ahat.nzcount.assign( Ahat.rows, 0 );
        Ahat.row_ptr.assign( Ahat.rows + 1, 0 );

        // Thread-local vector collection to hold structural column modifications safely
        std::vector<std::vector<size_t>> local_cols( A.rows );

// Step 1: Gather and deduplicate window indices per row in parallel
#pragma omp parallel for schedule( dynamic )
        for ( size_t i = 0; i < A.rows; ++i ) {
            size_t row_start = A.row_ptr[i];
            size_t row_end = A.row_ptr[i + 1];
            if ( row_start == row_end )
                continue;

            std::vector<size_t> windows;
            windows.reserve( row_end - row_start );

            // Translate flat 1D scalar column positions to window steps
            for ( size_t k = row_start; k < row_end; ++k ) {
                windows.push_back( A.col_ind[k] / W );
            }

            // Sort and eliminate duplicate window strikes
            std::sort( windows.begin(), windows.end() );
            windows.erase( std::unique( windows.begin(), windows.end() ), windows.end() );

            Ahat.nzcount[i] = windows.size();
            local_cols[i] = std::move( windows );
        }

        // Step 2: Compute flat 1D array prefix offsets sequentially
        for ( size_t i = 0; i < Ahat.rows; ++i ) {
            Ahat.row_ptr[i + 1] = Ahat.row_ptr[i] + Ahat.nzcount[i];
        }

        // Allocate flat 1D memory spaces
        size_t total_nnz = Ahat.row_ptr[Ahat.rows];
        Ahat.col_ind.resize( total_nnz );
        Ahat.values.assign( total_nnz, 1 ); // Structural masks populate elements as binary 1 flags
        Ahat.pattern_only = false;

// Step 3: Stream row-segments back into the flat 1D layout in parallel
#pragma omp parallel for schedule( static )
        for ( size_t i = 0; i < Ahat.rows; ++i ) {
            size_t dest_offset = Ahat.row_ptr[i];
            const auto& r_cols = local_cols[i];

            for ( size_t j = 0; j < r_cols.size(); ++j ) {
                Ahat.col_ind[dest_offset + j] = r_cols[j];
            }
        }
    }

    template <typename DataT = float, typename intT = int>
    void mask( const BSR<DataT, intT>& A, size_t W, CSR<size_t, size_t>& Ahat ) {
        Ahat.rows = static_cast<size_t>( A.block_rows );
        Ahat.cols = static_cast<size_t>( ( A.block_cols * A.block_size + W - 1 ) / W );
        Ahat.nzcount.assign( Ahat.rows, 0 );
        Ahat.row_ptr.assign( Ahat.rows + 1, 0 );

        // BSR structural masking logic goes here.
        // NOTE: reorder2() below needs mask/transpose/permute_cols for the matrix
        // type it's instantiated with. None of those exist yet for BSR (this stub
        // included), so reorder2<BSR<...>> will fail to compile until this is
        // filled in and BSR transpose()/permute_cols() are added. CSR is fully
        // supported below.
    }

    // ------------------------------------------------------------------------
    // column_rank(Ahat) -> rank[col]
    //
    // Ranks columns of the masked sketch by descending popularity: rank 0 is
    // the most frequent window, increasing ranks go to rarer windows. Ties are
    // broken by column index for determinism.
    //
    // This is the piece that encodes "common windows should dominate the sort,
    // rare windows should only break ties": cluster_lex() below sorts each
    // row's window list by this rank (ascending), so common windows land first
    // in the signature and act as the primary key in the lexicographic
    // comparison; rare windows trail off as tie-breakers.
    //
    // O(total nnz of Ahat + cols): counts are bounded integers in [0, rows], so
    // we bucket by count value instead of paying for a comparison sort.
    // ------------------------------------------------------------------------
    inline std::vector<size_t> column_rank( const CSR<size_t, size_t>& Ahat ) {
        const size_t cols = Ahat.cols;
        const size_t n = Ahat.rows;

        std::vector<size_t> col_count( cols, 0 );

        // Parallel histogram: each thread accumulates into a private buffer,
        // then folds into the shared one — avoids atomics on the hot loop.
#pragma omp parallel
        {
            std::vector<size_t> local( cols, 0 );
#pragma omp for schedule( static ) nowait
            for ( size_t i = 0; i < n; ++i ) {
                size_t s = Ahat.row_ptr[i], e = Ahat.row_ptr[i + 1];
                for ( size_t k = s; k < e; ++k )
                    ++local[Ahat.col_ind[k]];
            }
#pragma omp critical
            for ( size_t c = 0; c < cols; ++c )
                col_count[c] += local[c];
        }

        std::vector<std::vector<size_t>> by_count( n + 1 );
        for ( size_t c = 0; c < cols; ++c )
            by_count[col_count[c]].push_back( c );

        std::vector<size_t> rank( cols );
        size_t r = 0;
        for ( size_t cnt = n;; --cnt ) {
            for ( size_t c : by_count[cnt] )
                rank[c] = r++;
            if ( cnt == 0 )
                break;
        }
        return rank;
    }

    // ------------------------------------------------------------------------
    // cluster_lex(Ahat, P)
    //
    // Full lexicographic clustering of rows by their (rank-remapped) window
    // sets — the "suffix tree" idea, but implemented as what it actually is: a
    // trie over each row's prefix, built without ever materialising the tree.
    // Each level of recursion below IS one radix-bucket pass; the sorted
    // output order falls out of the recursion directly (this is the same
    // relationship a suffix array + LCP array has to an explicit suffix tree —
    // the array gives you the tree's structure for free).
    //
    // Column visiting order is given by column_rank(): rows are split first on
    // their most common shared window (primary key, large groups), then
    // recursively on rarer windows only within ties. This is what makes the
    // top-level grouping driven by widely-shared structure instead of being
    // scattered by rare columns.
    //
    // Complexity: O(total nnz of Ahat) for the rank lookup, O(total nnz log
    // avg_row_nnz) for the per-row sorts, and O(total nnz) amortised across all
    // recursion levels for the bucketing itself (each nonzero participates in
    // at most one bucket per level of its own row's signature length).
    // ------------------------------------------------------------------------
    inline void cluster_lex( const CSR<size_t, size_t>& Ahat, std::vector<size_t>& P ) {
        const size_t n = Ahat.rows;
        P.resize( n );
        if ( n == 0 )
            return;

        const std::vector<size_t> rank = column_rank( Ahat );

        // Per-row signature: this row's windows, rank-remapped and sorted
        // ascending, so the most common window comes first.
        std::vector<std::vector<size_t>> sig( n );
#pragma omp parallel for schedule( dynamic )
        for ( size_t i = 0; i < n; ++i ) {
            size_t s = Ahat.row_ptr[i], e = Ahat.row_ptr[i + 1];
            sig[i].resize( e - s );
            size_t k = s;
#ifdef __AVX512F__
            // Gather 8 ranks at a time, mirroring simd::fetch_first_cols'
            // gather-based column lookups elsewhere in this file. Unverified
            // against this codebase's actual simd.hpp — check intrinsic
            // signatures compile cleanly before relying on this path.
            for ( ; k + 8 <= e; k += 8 ) {
                __m512i idx =
                    _mm512_loadu_si512( reinterpret_cast<const void*>( &Ahat.col_ind[k] ) );
                __m512i r = _mm512_i64gather_epi64(
                    idx, reinterpret_cast<const long long*>( rank.data() ), 8 );
                _mm512_storeu_si512( reinterpret_cast<void*>( &sig[i][k - s] ), r );
            }
#endif
            for ( ; k < e; ++k )
                sig[i][k - s] = rank[Ahat.col_ind[k]];
            std::sort( sig[i].begin(), sig[i].end() );
        }

        // Sequential iterative radix-bucket sort to prevent stack overflow on deep recursive
        // graphs.
        auto process_bucket =
            [&]( std::vector<size_t> init_group, size_t init_depth, size_t init_lo ) {
                struct Task {
                    std::vector<size_t> group;
                    size_t depth;
                    size_t lo;
                };
                std::vector<Task> stack;
                stack.push_back( { std::move( init_group ), init_depth, init_lo } );

                while ( !stack.empty() ) {
                    Task task = std::move( stack.back() );
                    stack.pop_back();

                    if ( task.group.size() <= 1 ) {
                        for ( size_t r : task.group )
                            P[task.lo++] = r;
                        continue;
                    }

                    std::vector<size_t> exhausted;
                    std::unordered_map<size_t, std::vector<size_t>> buckets;
                    for ( size_t r : task.group ) {
                        if ( sig[r].size() <= task.depth )
                            exhausted.push_back( r );
                        else
                            buckets[sig[r][task.depth]].push_back( r );
                    }
                    for ( size_t r : exhausted )
                        P[task.lo++] = r;

                    std::vector<size_t> keys;
                    keys.reserve( buckets.size() );
                    for ( auto& kv : buckets )
                        keys.push_back( kv.first );
                    std::sort( keys.begin(), keys.end() );

                    size_t total_sz = 0;
                    for ( size_t key : keys )
                        total_sz += buckets[key].size();
                    size_t current_lo = task.lo + total_sz;

                    for ( auto it = keys.rbegin(); it != keys.rend(); ++it ) {
                        auto& b = buckets[*it];
                        size_t sz = b.size();
                        current_lo -= sz;
                        stack.push_back( { std::move( b ), task.depth + 1, current_lo } );
                    }
                }
            };

        // Bucket once at the top level by primary key so the independent
        // buckets can recurse in parallel — each thread writes into its own
        // precomputed slice of P, no synchronisation needed.
        std::vector<size_t> exhausted;
        std::unordered_map<size_t, std::vector<size_t>> top_buckets;
        for ( size_t i = 0; i < n; ++i ) {
            if ( sig[i].empty() )
                exhausted.push_back( i );
            else
                top_buckets[sig[i][0]].push_back( i );
        }

        std::vector<size_t> keys;
        keys.reserve( top_buckets.size() );
        for ( auto& kv : top_buckets )
            keys.push_back( kv.first );
        std::sort( keys.begin(), keys.end() );

        size_t pos = 0;
        for ( size_t r : exhausted )
            P[pos++] = r;

        std::vector<std::vector<size_t>> top_groups( keys.size() );
        for ( size_t b = 0; b < keys.size(); ++b )
            top_groups[b] = std::move( top_buckets[keys[b]] );

        std::vector<size_t> offsets( keys.size() + 1, 0 );
        for ( size_t b = 0; b < keys.size(); ++b )
            offsets[b + 1] = offsets[b] + top_groups[b].size();

#pragma omp parallel for schedule( dynamic )
        for ( size_t b = 0; b < keys.size(); ++b )
            process_bucket( std::move( top_groups[b] ), 1, pos + offsets[b] );
    }

    void cluster_jaccard( CSR<size_t, size_t>& Ahat, size_t K, std::vector<size_t>& P, float tau ) {
        const size_t n = Ahat.rows;
        P.resize( n );
        std::iota( P.begin(), P.end(), 0 );
        if ( n == 0 || K == 0 )
            return;
        K = std::min( K, n );

        // Merge-join Jaccard on sorted row segments — O(|ri| + |rj|)
        auto jaccard_dist = [&]( size_t ri, size_t rj ) -> float {
            size_t s1 = Ahat.row_ptr[ri], e1 = Ahat.row_ptr[ri + 1];
            size_t s2 = Ahat.row_ptr[rj], e2 = Ahat.row_ptr[rj + 1];
            size_t ni = e1 - s1, nj = e2 - s2;
            if ( ni == 0 && nj == 0 )
                return 0.0;
            if ( ni == 0 || nj == 0 )
                return 1.0;
            size_t isect = 0, pi = s1, pj = s2;
            while ( pi < e1 && pj < e2 ) {
                if ( Ahat.col_ind[pi] == Ahat.col_ind[pj] ) {
                    ++isect;
                    ++pi;
                    ++pj;
                } else if ( Ahat.col_ind[pi] < Ahat.col_ind[pj] ) {
                    ++pi;
                } else {
                    ++pj;
                }
            }
            return 1.0 - static_cast<float>( isect ) / ( ni + nj - isect );
        };

        // Rows in natural matrix order — no pre-sorting
        std::vector<size_t> unclustered( n );
        std::iota( unclustered.begin(), unclustered.end(), 0 );

        std::mt19937 rng{ std::random_device{}() };
        std::vector<std::vector<size_t>> result_clusters;

        while ( unclustered.size() >= K ) {
            // Uniform random sample of K centroids without replacement.
            // std::sample runs in O(unclustered.size()) — no full copy needed.
            std::vector<size_t> centroids( K );
            std::sample( unclustered.begin(), unclustered.end(), centroids.begin(), K, rng );

            // Parallel phase: each row independently finds its nearest centroid.
            // Writes go to disjoint per-index slots — no synchronisation needed.
            // Sentinel K means "distance > tau, leave for next iteration".
            std::vector<size_t> assignment( unclustered.size(), K );

#pragma omp parallel for schedule( dynamic )
            for ( size_t idx = 0; idx < unclustered.size(); ++idx ) {
                float best = std::numeric_limits<float>::max();
                size_t best_k = K;
                for ( size_t k = 0; k < K; ++k ) {
#ifdef __AVX512F__
                    float d = simd::jaccard( Ahat, unclustered[idx], centroids[k] );
#else
                    float d = jaccard_dist( unclustered[idx], centroids[k] );
#endif
                    if ( d < best ) {
                        best = d;
                        best_k = k;
                    }
                }
                if ( best <= tau )
                    assignment[idx] = best_k;
            }
            // Sequential scatter: one pass preserves natural row ordering
            // within each cluster and within the leftover set.
            std::vector<std::vector<size_t>> new_clusters( K );
            std::vector<size_t> leftover;
            leftover.reserve( unclustered.size() );

            for ( size_t idx = 0; idx < unclustered.size(); ++idx ) {
                if ( assignment[idx] < K )
                    new_clusters[assignment[idx]].push_back( unclustered[idx] );
                else
                    leftover.push_back( unclustered[idx] );
            }

            for ( auto& c : new_clusters )
                if ( !c.empty() )
                    result_clusters.push_back( std::move( c ) );

            unclustered = std::move( leftover );
        }

        // Fewer than K rows remain — each becomes a singleton cluster
        for ( size_t row : unclustered )
            result_clusters.push_back( { row } );

        // Flatten result_clusters into the output permutation
        size_t pos = 0;
        for ( const auto& c : result_clusters )
            for ( size_t row : c )
                P[pos++] = row;
    }
    // Cluster the masked matrix Ahat into K clusters. Each cluster is a set of rows of Ahat that
    // are similar to each other. Then generate the permutation P that reorders the rows of Ahat
    // according to the clusters.
    void cluster( CSR<size_t, size_t>& Ahat, size_t K, std::vector<size_t>& P, float tau = 0.5 ) {
        const size_t n = Ahat.rows;
        P.resize( n );
        std::iota( P.begin(), P.end(), 0 );
        if ( n == 0 || K == 0 )
            return;

        std::vector<std::vector<size_t>> col_buckets( Ahat.cols );

#ifdef __AVX512F__
        const size_t empty_sentinel = Ahat.cols;
        size_t i = 0;
        for ( ; i + 8 <= n; i += 8 ) {
            alignas( 64 ) size_t first_cols[8];
            simd::fetch_first_cols( Ahat, i, first_cols, empty_sentinel );
            for ( int k = 0; k < 8; ++k ) {
                if ( first_cols[k] != empty_sentinel ) {
                    col_buckets[first_cols[k]].push_back( i + k );
                }
            }
        }
        for ( ; i < n; ++i ) {
            if ( Ahat.nzcount[i] > 0 )
                col_buckets[Ahat.col_ind[Ahat.row_ptr[i]]].push_back( i );
        }
#else
        for ( size_t i = 0; i < n; ++i )
            if ( Ahat.nzcount[i] > 0 )
                col_buckets[Ahat.col_ind[Ahat.row_ptr[i]]].push_back( i );
#endif

        std::vector<size_t> unclustered;
        unclustered.reserve( n );
        for ( size_t j = 0; j < Ahat.cols; ++j )
            for ( size_t r : col_buckets[j] )
                unclustered.push_back( r );
        for ( size_t i = 0; i < n; ++i )
            if ( Ahat.nzcount[i] == 0 )
                unclustered.push_back( i );

        P = std::move( unclustered ); // ← radix-sorted order IS the permutation
    }
    // Apply the permutation P to the input matrix A to obtain the reordered matrix A'.
    // Need to overload for CSR and BSR
    template <typename DataT, typename intT>
    void permute( CSR<DataT, intT>& A, const std::vector<size_t>& P ) {
        assert( P.size() == static_cast<size_t>( A.rows ) );

        std::vector<intT> new_nzcount( A.rows );
        std::vector<intT> new_row_ptr( A.rows + 1, 0 );

        // Step 1: new row lengths from the permuted source rows
        for ( size_t i = 0; i < static_cast<size_t>( A.rows ); ++i )
            new_nzcount[i] = A.nzcount[P[i]];

        // Step 2: prefix sum → new row pointers
        for ( size_t i = 0; i < static_cast<size_t>( A.rows ); ++i )
            new_row_ptr[i + 1] = new_row_ptr[i] + new_nzcount[i];

        intT total_nnz = new_row_ptr[A.rows];
        std::vector<intT> new_col_ind( total_nnz );
        std::vector<DataT> new_values;
        if ( !A.pattern_only )
            new_values.resize( total_nnz );

// Step 3: scatter rows into new layout
#pragma omp parallel for schedule( dynamic )
        for ( size_t i = 0; i < static_cast<size_t>( A.rows ); ++i ) {
            intT src = A.row_ptr[P[i]]; // start of old row in flat array
            intT dst = new_row_ptr[i];  // start of new row in flat array
            intT len = new_nzcount[i];

            std::copy_n( &A.col_ind[src], len, &new_col_ind[dst] );
            if ( !A.pattern_only )
                std::copy_n( &A.values[src], len, &new_values[dst] );
        }
        // Step 4: replace in-place
        A.row_ptr = std::move( new_row_ptr );
        A.col_ind = std::move( new_col_ind );
        A.nzcount = std::move( new_nzcount );
        if ( !A.pattern_only )
            A.values = std::move( new_values );
    }

    // Permute BSR
    template <typename DataT, typename intT>
    void permute( BSR<DataT, intT>& A, const std::vector<size_t>& P ) {
        assert( P.size() == static_cast<size_t>( A.block_rows ) );

        const intT bsq = A.block_size * A.block_size;

        // Step 1: new block-row pointer from permuted source lengths
        std::vector<intT> new_bptr( A.block_rows + 1, 0 );
        for ( size_t I = 0; I < static_cast<size_t>( A.block_rows ); ++I ) {
            intT old_I = static_cast<intT>( P[I] );
            new_bptr[I + 1] = new_bptr[I] + ( A.bptr[old_I + 1] - A.bptr[old_I] );
        }

        intT total_blocks = new_bptr[A.block_rows];
        std::vector<intT> new_bind( total_blocks );
        std::vector<DataT> new_bnz( total_blocks * bsq, static_cast<DataT>( 0 ) );

// Step 2: copy block-column indices + dense block data
#pragma omp parallel for schedule( dynamic )
        for ( size_t I = 0; I < static_cast<size_t>( A.block_rows ); ++I ) {
            intT old_I = static_cast<intT>( P[I] );
            intT src_blk = A.bptr[old_I]; // first block of old block-row
            intT dst_blk = new_bptr[I];   // first block of new block-row
            intT num_blocks = A.bptr[old_I + 1] - src_blk;

            // Block column indices
            std::copy_n( &A.bind[src_blk], num_blocks, &new_bind[dst_blk] );

            // Dense block values — each block is bsq contiguous floats
            std::copy_n( &A.bnz[src_blk * bsq], num_blocks * bsq, &new_bnz[dst_blk * bsq] );
        }

        A.bptr = std::move( new_bptr );
        A.bind = std::move( new_bind );
        A.bnz = std::move( new_bnz );
    }

    // ------------------------------------------------------------------------
    // transpose(A, AT) — CSR -> CSR transpose via counting sort.
    //
    // The scatter pass (Step 2) is inherently sequential per *destination*
    // row: multiple source rows can write into the same AT row, so it isn't
    // parallelised the way permute() is. The histogram pass before it is the
    // expensive part for large matrices and could be parallelised the same
    // way column_rank() does it (private histograms folded under a critical
    // section) if profiling shows it's worth it.
    //
    // Bonus: because source rows are visited in ascending order, each AT row
    // ends up with col_ind already sorted ascending — no extra sort needed.
    // ------------------------------------------------------------------------
    template <typename DataT, typename intT>
    void transpose( const CSR<DataT, intT>& A, CSR<DataT, intT>& AT ) {
        AT.rows = A.cols;
        AT.cols = A.rows;
        AT.pattern_only = A.pattern_only;

        AT.row_ptr.assign( AT.rows + 1, 0 );

        std::vector<intT> col_count( A.cols, 0 );
        intT total_nnz = A.row_ptr[A.rows];
        for ( intT k = 0; k < total_nnz; ++k )
            ++col_count[A.col_ind[k]];

        for ( size_t j = 0; j < AT.rows; ++j )
            AT.row_ptr[j + 1] = AT.row_ptr[j] + col_count[j];

        AT.nzcount.assign( col_count.begin(), col_count.end() );
        AT.col_ind.resize( total_nnz );
        if ( !A.pattern_only )
            AT.values.resize( total_nnz );

        std::vector<intT> cursor( AT.row_ptr.begin(), AT.row_ptr.end() - 1 );

        for ( size_t i = 0; i < static_cast<size_t>( A.rows ); ++i ) {
            intT s = A.row_ptr[i], e = A.row_ptr[i + 1];
            for ( intT k = s; k < e; ++k ) {
                intT j = A.col_ind[k];
                intT dst = cursor[j]++;
                AT.col_ind[dst] = static_cast<intT>( i );
                if ( !A.pattern_only )
                    AT.values[dst] = A.values[k];
            }
        }
    }

    // ------------------------------------------------------------------------
    // permute_cols(A, Q) — apply a column permutation directly, no transpose
    // round-trip needed. Same convention as permute()'s P: Q[j] is the OLD
    // column index that becomes new column j.
    //
    // Each row's columns get remapped through inv(Q) and then re-sorted, since
    // the remap can break the ascending order CSR rows are expected to keep.
    // Each row is independent, so this parallelises the same way permute()'s
    // scatter step does.
    // ------------------------------------------------------------------------
    template <typename DataT, typename intT>
    void permute_cols( CSR<DataT, intT>& A, const std::vector<size_t>& Q ) {
        assert( Q.size() == static_cast<size_t>( A.cols ) );

        std::vector<intT> inv_Q( A.cols );
        for ( size_t j = 0; j < Q.size(); ++j )
            inv_Q[Q[j]] = static_cast<intT>( j );

#pragma omp parallel for schedule( dynamic )
        for ( size_t i = 0; i < static_cast<size_t>( A.rows ); ++i ) {
            intT s = A.row_ptr[i], e = A.row_ptr[i + 1];
            size_t len = e - s;
            if ( len == 0 )
                continue;

            for ( intT k = s; k < e; ++k )
                A.col_ind[k] = inv_Q[A.col_ind[k]];

            std::vector<size_t> order( len );
            std::iota( order.begin(), order.end(), 0 );
            std::sort( order.begin(), order.end(), [&]( size_t a, size_t b ) {
                return A.col_ind[s + a] < A.col_ind[s + b];
            } );

            std::vector<intT> new_cols( len );
            for ( size_t t = 0; t < len; ++t )
                new_cols[t] = A.col_ind[s + order[t]];
            std::copy( new_cols.begin(), new_cols.end(), A.col_ind.begin() + s );

            if ( !A.pattern_only ) {
                std::vector<DataT> new_vals( len );
                for ( size_t t = 0; t < len; ++t )
                    new_vals[t] = A.values[s + order[t]];
                std::copy( new_vals.begin(), new_vals.end(), A.values.begin() + s );
            }
        }
    }

    // Apply the reordering to the input matrix A. The reordering is defined by the parameters W and
    // K.
    template <typename MatrixType>
    void reorder( MatrixType& A, size_t W, size_t K, float tau = 0.9 ) {
        CSR<size_t, size_t> Ahat;
        std::vector<size_t> P;
        size_t W_mes = 32;
        LOG_INFO( "msg",
                  "Starting reordering",
                  "nonzero blocks",
                  count_nonzero_blocks( A, W_mes, W_mes ) );

        mask( A, W, Ahat );
        LOG_INFO( "msg", "Masking completed", "Sketch size", Ahat.cols );
        cluster( Ahat, K, P, tau );
        LOG_INFO( "msg",
                  "Applying permutation to the original matrix",
                  "nonzero blocks",
                  count_nonzero_blocks( A, W_mes, W_mes ) );
        permute( A, P );
        LOG_INFO( "msg",
                  "Permutation applied",
                  "nonzero blocks",
                  count_nonzero_blocks( A, W_mes, W_mes ) );
    }

    // ------------------------------------------------------------------------
    // reorder2(A, W, max_iters) — alternating row/column reordering.
    //
    // Bond-Energy-Algorithm-style co-clustering: cluster & permute rows with
    // cluster_lex(), transpose, cluster & permute columns the same way, and
    // repeat. Each pass is O(nnz) (no approximate graph, no k-NN), so a handful
    // of iterations costs little more than your existing single-pass
    // lexicographic reorder.
    //
    // Stops as soon as a full row+column pass fails to *strictly* improve the
    // block count, and always returns the best permutation state actually
    // seen — a regression on the last iteration can't make the final result
    // worse than an earlier, better iteration.
    //
    // CSR only for now: BSR's mask()/transpose()/permute_cols() aren't
    // implemented (mask<BSR> is a stub above), so this won't compile for
    // BSR<...> until those are filled in.
    // ------------------------------------------------------------------------
    template <typename DataT, typename intT>
    void reorder2( CSR<DataT, intT>& A,
                   size_t W,
                   size_t max_iters = 4,
                   std::vector<size_t>* out_perm = nullptr ) {
        const size_t W_mes = 32;
        size_t best_blocks = count_nonzero_blocks( A, W_mes, W_mes );
        LOG_INFO( "msg", "Starting 2-sided reordering", "nonzero blocks", best_blocks );

        CSR<DataT, intT> best = A;

        // Cumulative row permutation: out_perm[i] = original row index that ends up at position i
        std::vector<size_t> cumulative( static_cast<size_t>( A.rows ) );
        std::iota( cumulative.begin(), cumulative.end(), 0 );
        std::vector<size_t> best_perm = cumulative;

        for ( size_t iter = 0; iter < max_iters; ++iter ) {
            // --- row pass ---
            CSR<size_t, size_t> Ahat;
            std::vector<size_t> P;
            mask( A, W, Ahat );
            cluster_lex( Ahat, P );

            // Compose the new row permutation into the cumulative one:
            //   cumulative_new[i] = cumulative_old[ P[i] ]
            // because permute(A, P) moves old row P[i] to new position i.
            {
                std::vector<size_t> composed( static_cast<size_t>( A.rows ) );
                for ( size_t i = 0; i < static_cast<size_t>( A.rows ); ++i )
                    composed[i] = cumulative[P[i]];
                cumulative.swap( composed );
            }

            permute( A, P );

            // --- column pass, via transpose ---
            CSR<DataT, intT> AT;
            transpose( A, AT );
            CSR<size_t, size_t> AhatT;
            std::vector<size_t> Q;
            mask( AT, W, AhatT );
            cluster_lex( AhatT, Q );
            permute_cols( A, Q );

            size_t blocks = count_nonzero_blocks( A, W_mes, W_mes );
            LOG_INFO( "msg", "Pass completed", "iter", iter, "nonzero blocks", blocks );

            if ( blocks < best_blocks ) {
                best_blocks = blocks;
                best = A;
                best_perm = cumulative;
            } else {
                break;
            }
        }

        A = std::move( best );
        if ( out_perm )
            *out_perm = std::move( best_perm );
        LOG_INFO( "msg", "2-sided reordering done", "nonzero blocks", best_blocks );
    }

    template <typename DataT = float, typename intT = int>
    void mask_multilevel( CSR<DataT, intT>& A,
                          const std::vector<size_t>& Ws,
                          std::vector<size_t>& P,
                          CSR<size_t, size_t>* Ahat_out = nullptr ) {
        expect( !Ws.empty() );
        std::vector<CSR<size_t, size_t>> levels( Ws.size() );
        for ( size_t l = 0; l < Ws.size(); ++l )
            mask( A, Ws[l], levels[l] );

        CSR<size_t, size_t> Ahat;
        Ahat.rows = levels[0].rows;
        std::vector<size_t> offset( Ws.size(), 0 );
        size_t running = 0;
        for ( size_t l = 0; l < Ws.size(); ++l ) {
            offset[l] = running;
            running += static_cast<size_t>( levels[l].cols );
        }
        Ahat.cols = running;

        Ahat.nzcount.assign( Ahat.rows, 0 );
        Ahat.row_ptr.assign( Ahat.rows + 1, 0 );
        std::vector<std::vector<size_t>> local_cols( Ahat.rows );

#pragma omp parallel for schedule( dynamic )
        for ( size_t i = 0; i < Ahat.rows; ++i ) {
            std::vector<size_t>& row = local_cols[i];
            for ( size_t l = 0; l < Ws.size(); ++l ) {
                size_t s = static_cast<size_t>( levels[l].row_ptr[i] );
                size_t e = static_cast<size_t>( levels[l].row_ptr[i + 1] );
                for ( size_t k = s; k < e; ++k )
                    row.push_back( static_cast<size_t>( levels[l].col_ind[k] ) + offset[l] );
            }
            std::sort( row.begin(), row.end() );
            Ahat.nzcount[i] = row.size();
        }
        for ( size_t i = 0; i < Ahat.rows; ++i )
            Ahat.row_ptr[i + 1] = Ahat.row_ptr[i] + Ahat.nzcount[i];

        size_t total = Ahat.row_ptr[Ahat.rows];
        Ahat.col_ind.resize( total );
        Ahat.values.assign( total, 1 );
        Ahat.pattern_only = false;

#pragma omp parallel for schedule( static )
        for ( size_t i = 0; i < Ahat.rows; ++i )
            std::copy( local_cols[i].begin(),
                       local_cols[i].end(),
                       Ahat.col_ind.begin() + Ahat.row_ptr[i] );

        // Cluster the multi-resolution sketch exactly as cluster_lex would cluster
        // a single-level one -- P[i] is the original row index that ends up at
        // position i, ready to hand straight to permute(A, P).
        cluster_lex( Ahat, P );

        if ( Ahat_out )
            *Ahat_out = std::move( Ahat );
    }

    // ------------------------------------------------------------------------
    // HybridStats: diagnostics for the 3-stage hybrid reordering pipeline
    // ------------------------------------------------------------------------
    struct HybridStats {
        size_t n_empty = 0;          // rows with 0 non-zero windows
        size_t n_hub_windows = 0;    // windows filtered as global hubs
        size_t n_micro = 0;          // micro-clusters formed in Stage 1
        size_t macro_incidences = 0; // total (micro-cluster, window) pins in Stage 2
        size_t macro_components = 0; // connected components walked by Macro-RCM
    };

    namespace detail {

        // Fast, high-avalanche 64-bit SplitMix64 bijective hash mixer
        inline uint64_t splitmix64( uint64_t x ) noexcept {
            x += 0x9e3779b97f4a7c15ULL;
            x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
            x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebULL;
            return x ^ ( x >> 31 );
        }

        // Fixed-width 24-byte per-row sketch (zero heap allocations per row)
        struct RowSketch {
            uint64_t h1;          // Primary MinHash key (uniform 64-bit)
            uint64_t h2;          // Secondary MinHash key (uniform 64-bit)
            uint32_t rare_win;    // Rank/ID of rarest window touched by row
            uint32_t row_len;     // Number of windows in row
            size_t   row_id;      // Original matrix row index

            bool operator<( const RowSketch& o ) const noexcept {
                if ( h1 != o.h1 ) return h1 < o.h1;
                if ( h2 != o.h2 ) return h2 < o.h2;
                if ( rare_win != o.rare_win ) return rare_win < o.rare_win;
                if ( row_len != o.row_len ) return row_len < o.row_len;
                return row_id < o.row_id;
            }
        };

        // Scalar fallback for Jaccard distance on sorted CSR row intervals
        inline float scalar_jaccard_dist( const CSR<size_t, size_t>& Ahat,
                                          size_t ri,
                                          size_t rj ) noexcept {
            size_t s1 = Ahat.row_ptr[ri], e1 = Ahat.row_ptr[ri + 1];
            size_t s2 = Ahat.row_ptr[rj], e2 = Ahat.row_ptr[rj + 1];
            size_t ni = e1 - s1, nj = e2 - s2;
            if ( ni == 0 && nj == 0 ) return 0.0f;
            if ( ni == 0 || nj == 0 ) return 1.0f;

            size_t isect = 0, pi = s1, pj = s2;
            while ( pi < e1 && pj < e2 ) {
                size_t c1 = Ahat.col_ind[pi];
                size_t c2 = Ahat.col_ind[pj];
                if ( c1 == c2 ) {
                    ++isect;
                    ++pi;
                    ++pj;
                } else if ( c1 < c2 ) {
                    ++pi;
                } else {
                    ++pj;
                }
            }
            return 1.0f - static_cast<float>( isect ) / static_cast<float>( ni + nj - isect );
        }

        inline float fast_jaccard( const CSR<size_t, size_t>& Ahat,
                                   size_t ri,
                                   size_t rj ) noexcept {
#ifdef __AVX512F__
            return simd::jaccard( const_cast<CSR<size_t, size_t>&>( Ahat ), ri, rj );
#else
            return scalar_jaccard_dist( Ahat, ri, rj );
#endif
        }

    } // namespace detail

    // ------------------------------------------------------------------------
    // cluster_hybrid(Ahat, P, W_block, lookahead_C, hub_frac, stats_out)
    //
    // 3-Stage Near-Linear Reordering Pipeline:
    //   Stage 1: IDF-Filtered 2-Hash MinHash + 256-Bucket Parallel Sort
    //   Stage 2: Weighted Bipartite Macro-RCM (FIFO Wavefront + Component Reversal)
    //   Stage 3: OpenMP Chunked SIMD Jaccard Slab Packing
    // ------------------------------------------------------------------------
    inline void cluster_hybrid( const CSR<size_t, size_t>& Ahat,
                                std::vector<size_t>& P,
                                size_t W_block = 32,
                                size_t lookahead_C = 64,
                                double hub_frac = 0.05,
                                HybridStats* stats_out = nullptr ) {
        const size_t n = Ahat.rows;
        const size_t n_windows = Ahat.cols;
        P.resize( n );
        if ( n == 0 )
            return;
        if ( n_windows == 0 ) {
            std::iota( P.begin(), P.end(), 0 );
            return;
        }

        W_block = std::max<size_t>( 1, W_block );

        // ====================================================================
        // PASSAGE 1: Parallel Window Degree Histogram & Hub Thresholding
        // ====================================================================
        std::vector<size_t> win_deg( n_windows, 0 );
        const size_t total_nnz = Ahat.row_ptr[n];

#pragma omp parallel for schedule( static )
        for ( size_t k = 0; k < total_nnz; ++k ) {
#pragma omp atomic
            ++win_deg[Ahat.col_ind[k]];
        }

        // Hub cutoff: windows touched by > hub_frac * n rows (minimum 64)
        // cannot be localized into a few W_block slabs and would short-circuit
        // both MinHash collisions and the Stage 2 BFS wavefront.
        const size_t hub_cutoff = std::max<size_t>(
            64, static_cast<size_t>( std::ceil( hub_frac * static_cast<double>( n ) ) ) );

        size_t n_hub_windows = 0;
        for ( size_t w = 0; w < n_windows; ++w ) {
            if ( win_deg[w] > hub_cutoff )
                ++n_hub_windows;
        }

        // ====================================================================
        // PASSAGE 2 (Stage 1): IDF-Filtered MinHash + Parallel Bucket Sort
        // ====================================================================
        constexpr uint64_t SEED_1 = 0x243f6a8885a308d3ULL;
        constexpr uint64_t SEED_2 = 0x13198a2e03707344ULL;

        std::vector<detail::RowSketch> sketches( n );
        std::vector<uint8_t> is_empty_row( n, 0 );

#pragma omp parallel for schedule( dynamic, 256 )
        for ( size_t i = 0; i < n; ++i ) {
            const size_t s = Ahat.row_ptr[i];
            const size_t e = Ahat.row_ptr[i + 1];
            const size_t len = e - s;

            if ( len == 0 ) {
                is_empty_row[i] = 1;
                sketches[i] = { UINT64_MAX, UINT64_MAX, UINT32_MAX, 0, i };
                continue;
            }

            uint64_t min_h1 = UINT64_MAX;
            uint64_t min_h2 = UINT64_MAX;
            size_t best_rare_deg = SIZE_MAX;
            size_t best_rare_win = Ahat.col_ind[s];
            bool found_non_hub = false;

            // Pass 1 over row i: evaluate non-hub windows (deg <= hub_cutoff)
            for ( size_t k = s; k < e; ++k ) {
                size_t w = Ahat.col_ind[k];
                size_t deg = win_deg[w];
                if ( deg <= hub_cutoff ) {
                    found_non_hub = true;
                    uint64_t h1 = detail::splitmix64( w ^ SEED_1 );
                    uint64_t h2 = detail::splitmix64( w ^ SEED_2 );
                    if ( h1 < min_h1 ) min_h1 = h1;
                    if ( h2 < min_h2 ) min_h2 = h2;

                    // Prefer rarest shared window (deg >= 2) as tie-breaker
                    size_t eff_deg = ( deg >= 2 ) ? deg : ( n + 1 );
                    if ( eff_deg < best_rare_deg ) {
                        best_rare_deg = eff_deg;
                        best_rare_win = w;
                    }
                }
            }

            // Fallback: if row i ONLY touches hub windows, hash all its windows
            // so it is never falsely classified as an empty row.
            if ( !found_non_hub ) {
                for ( size_t k = s; k < e; ++k ) {
                    size_t w = Ahat.col_ind[k];
                    size_t deg = win_deg[w];
                    uint64_t h1 = detail::splitmix64( w ^ SEED_1 );
                    uint64_t h2 = detail::splitmix64( w ^ SEED_2 );
                    if ( h1 < min_h1 ) min_h1 = h1;
                    if ( h2 < min_h2 ) min_h2 = h2;
                    if ( deg < best_rare_deg ) {
                        best_rare_deg = deg;
                        best_rare_win = w;
                    }
                }
            }

            sketches[i] = {
                min_h1,
                min_h2,
                static_cast<uint32_t>( best_rare_win ),
                static_cast<uint32_t>( len ),
                i
            };
        }

        // Partition non-empty rows into 256 buckets by the top 8 bits of h1.
        // Because h1 is the primary key of RowSketch::operator<, sorting the
        // 256 buckets independently in parallel produces the exact global sort!
        constexpr size_t NUM_BUCKETS = 256;
        std::vector<size_t> exhausted_rows;
        std::vector<std::vector<detail::RowSketch>> radix_buckets( NUM_BUCKETS );

        for ( size_t i = 0; i < n; ++i ) {
            if ( is_empty_row[i] ) {
                exhausted_rows.push_back( i );
            } else {
                size_t b = static_cast<size_t>( sketches[i].h1 >> 56 );
                radix_buckets[b].push_back( sketches[i] );
            }
        }

#pragma omp parallel for schedule( dynamic )
        for ( size_t b = 0; b < NUM_BUCKETS; ++b ) {
            std::sort( radix_buckets[b].begin(), radix_buckets[b].end() );
        }

        // Slice sorted buckets into compact micro-clusters (size <= W_block).
        // Consecutive rows merge into the same micro-cluster if they share
        // their primary MinHash h1 OR share their rarest window rare_win.
        std::vector<std::vector<size_t>> bucket_micro_groups[NUM_BUCKETS];

#pragma omp parallel for schedule( dynamic )
        for ( size_t b = 0; b < NUM_BUCKETS; ++b ) {
            const auto& vec = radix_buckets[b];
            auto& out_mc = bucket_micro_groups[b];
            if ( vec.empty() )
                continue;

            std::vector<size_t> current_group;
            current_group.reserve( W_block );
            current_group.push_back( vec[0].row_id );

            for ( size_t idx = 1; idx < vec.size(); ++idx ) {
                const bool same_primary = ( vec[idx].h1 == vec[idx - 1].h1 );
                const bool same_rare = ( vec[idx].rare_win == vec[idx - 1].rare_win );

                if ( current_group.size() < W_block && ( same_primary || same_rare ) ) {
                    current_group.push_back( vec[idx].row_id );
                } else {
                    out_mc.push_back( std::move( current_group ) );
                    current_group.clear();
                    current_group.reserve( W_block );
                    current_group.push_back( vec[idx].row_id );
                }
            }
            if ( !current_group.empty() )
                out_mc.push_back( std::move( current_group ) );
        }

        // Flatten bucket micro-clusters into global micro_groups
        size_t n_micro = 0;
        for ( size_t b = 0; b < NUM_BUCKETS; ++b )
            n_micro += bucket_micro_groups[b].size();

        std::vector<std::vector<size_t>> micro_groups;
        micro_groups.reserve( n_micro );
        for ( size_t b = 0; b < NUM_BUCKETS; ++b ) {
            for ( auto& g : bucket_micro_groups[b] )
                micro_groups.push_back( std::move( g ) );
        }

        // ====================================================================
        // PASSAGE 3 (Stage 2): Lock-Free Macro-Graph & Weighted Bipartite RCM
        // ====================================================================
        // Build per-micro-cluster window summaries (excluding global hubs
        // unless a micro-cluster touches only hubs).
        std::vector<std::vector<size_t>> macro_sig( n_micro );

#pragma omp parallel for schedule( dynamic, 64 )
        for ( size_t c = 0; c < n_micro; ++c ) {
            size_t total_len = 0;
            for ( size_t r : micro_groups[c] )
                total_len += ( Ahat.row_ptr[r + 1] - Ahat.row_ptr[r] );

            std::vector<size_t> u;
            u.reserve( total_len );
            for ( size_t r : micro_groups[c] ) {
                size_t s = Ahat.row_ptr[r], e = Ahat.row_ptr[r + 1];
                for ( size_t k = s; k < e; ++k ) {
                    size_t w = Ahat.col_ind[k];
                    if ( win_deg[w] <= hub_cutoff )
                        u.push_back( w );
                }
            }

            // Fallback if all windows in this micro-cluster exceeded hub_cutoff
            if ( u.empty() ) {
                for ( size_t r : micro_groups[c] ) {
                    size_t s = Ahat.row_ptr[r], e = Ahat.row_ptr[r + 1];
                    for ( size_t k = s; k < e; ++k )
                        u.push_back( Ahat.col_ind[k] );
                }
            }

            std::sort( u.begin(), u.end() );
            u.erase( std::unique( u.begin(), u.end() ), u.end() );
            macro_sig[c] = std::move( u );
        }

        // Construct flat CSR transpose (window -> micro-clusters) with atomic
        // offsets instead of #pragma omp critical vector push_backs.
        std::vector<size_t> win_mc_count( n_windows, 0 );
#pragma omp parallel for schedule( dynamic, 64 )
        for ( size_t c = 0; c < n_micro; ++c ) {
            for ( size_t w : macro_sig[c] ) {
#pragma omp atomic
                ++win_mc_count[w];
            }
        }

        std::vector<size_t> win_ptr( n_windows + 1, 0 );
        for ( size_t w = 0; w < n_windows; ++w )
            win_ptr[w + 1] = win_ptr[w] + win_mc_count[w];

        const size_t macro_incidences = win_ptr[n_windows];
        std::vector<size_t> win_clusters( macro_incidences );
        std::vector<size_t> win_cursor = win_ptr;

#pragma omp parallel for schedule( dynamic, 64 )
        for ( size_t c = 0; c < n_micro; ++c ) {
            for ( size_t w : macro_sig[c] ) {
                size_t pos;
#pragma omp atomic capture
                pos = win_cursor[w]++;
                win_clusters[pos] = c;
            }
        }

        // True FIFO Level-Order Bipartite RCM with Inverse-Degree Edge Weights:
        //   - Seed each component from its lowest-degree unvisited micro-cluster
        //   - Expand each window at most once (visited_window guard -> O(pins) total)
        //   - Accumulate multi-window connection strength: sum(1.0f / deg(w))
        //   - Reverse each component's slice (Reverse Cuthill-McKee)
        std::vector<size_t> seeds_by_deg( n_micro );
        std::iota( seeds_by_deg.begin(), seeds_by_deg.end(), 0 );
        std::sort( seeds_by_deg.begin(), seeds_by_deg.end(), [&]( size_t a, size_t b ) {
            return macro_sig[a].size() < macro_sig[b].size();
        } );

        std::vector<uint8_t> visited_cluster( n_micro, 0 );
        std::vector<uint8_t> visited_window( n_windows, 0 );
        std::vector<float> conn_score( n_micro, 0.0f );
        std::vector<size_t> local_candidates;
        local_candidates.reserve( 256 );

        std::vector<size_t> macro_order;
        macro_order.reserve( n_micro );
        size_t macro_components = 0;

        for ( size_t seed : seeds_by_deg ) {
            if ( visited_cluster[seed] )
                continue;

            ++macro_components;
            const size_t comp_start = macro_order.size();

            // Mark visited ON ENQUEUE so a node never enters macro_order twice
            visited_cluster[seed] = 1;
            macro_order.push_back( seed );

            // macro_order[head .. macro_order.size()-1] acts directly as the
            // in-place FIFO queue!
            for ( size_t head = comp_start; head < macro_order.size(); ++head ) {
                const size_t c = macro_order[head];
                local_candidates.clear();

                for ( size_t w : macro_sig[c] ) {
                    if ( visited_window[w] )
                        continue;
                    visited_window[w] = 1;

                    const size_t ws = win_ptr[w];
                    const size_t we = win_ptr[w + 1];
                    const float inv_deg = 1.0f / static_cast<float>( std::max<size_t>( 1, we - ws ) );

                    for ( size_t idx = ws; idx < we; ++idx ) {
                        size_t c2 = win_clusters[idx];
                        if ( !visited_cluster[c2] ) {
                            if ( conn_score[c2] == 0.0f )
                                local_candidates.push_back( c2 );
                            conn_score[c2] += inv_deg;
                        }
                    }
                }

                // Order newly discovered 2-hop neighbors:
                // Strongest shared-window coupling first, then ascending degree (RCM)
                std::sort( local_candidates.begin(),
                           local_candidates.end(),
                           [&]( size_t a, size_t b ) {
                               if ( conn_score[a] != conn_score[b] )
                                   return conn_score[a] > conn_score[b];
                               return macro_sig[a].size() < macro_sig[b].size();
                           } );

                for ( size_t c2 : local_candidates ) {
                    visited_cluster[c2] = 1;
                    conn_score[c2] = 0.0f; // Clean up in O(1) without std::fill
                    macro_order.push_back( c2 );
                }
            }

            // Reverse the component slice for true Reverse Cuthill-McKee ordering
            std::reverse( macro_order.begin() + comp_start, macro_order.end() );
        }

        expect( macro_order.size() == n_micro );

        // Expand coarsened macro_order back into the full row permutation P
        size_t pos = 0;
        for ( size_t r : exhausted_rows )
            P[pos++] = r;
        for ( size_t c : macro_order ) {
            for ( size_t r : micro_groups[c] )
                P[pos++] = r;
        }
        expect( pos == n );

        // ====================================================================
        // PASSAGE 4 (Stage 3): OpenMP Chunked SIMD Jaccard Slab Packing
        // ====================================================================
        // Refine the non-empty slice P[n_empty .. n-1] in parallel L2-resident
        // chunks (each chunk is a multiple of W_block rows).
        const size_t n_empty = exhausted_rows.size();
        const size_t n_active = n - n_empty;

        if ( lookahead_C > 1 && n_active > 2 ) {
            const size_t chunk_size = std::max<size_t>( W_block * 32, lookahead_C * 4 );
            const size_t num_chunks = ( n_active + chunk_size - 1 ) / chunk_size;

#pragma omp parallel for schedule( dynamic )
            for ( size_t cid = 0; cid < num_chunks; ++cid ) {
                const size_t start = n_empty + cid * chunk_size;
                const size_t end = std::min( n, start + chunk_size );
                for ( size_t i = start; i + 1 < end; ++i ) {
                    const size_t curr = P[i];
                    if ( Ahat.row_ptr[curr] == Ahat.row_ptr[curr + 1] )
                        continue;

                    size_t best_j = i + 1;
                    float best_dist = detail::fast_jaccard( Ahat, curr, P[i + 1] );

                    if ( best_dist > 0.0f ) {
                        const size_t limit = std::min( end, i + 1 + lookahead_C );
                        for ( size_t j = i + 2; j < limit; ++j ) {
                            float d = detail::fast_jaccard( Ahat, curr, P[j] );
                            if ( d < best_dist ) {
                                best_dist = d;
                                best_j = j;
                                if ( best_dist == 0.0f )
                                    break; // Exact window match found
                            }
                        }
                    }

                    if ( best_j != i + 1 )
                        std::swap( P[i + 1], P[best_j] );
                }
            }
        }

        if ( stats_out ) {
            stats_out->n_empty = n_empty;
            stats_out->n_hub_windows = n_hub_windows;
            stats_out->n_micro = n_micro;
            stats_out->macro_incidences = macro_incidences;
            stats_out->macro_components = macro_components;
        }

        LOG_INFO( "msg",
                  "cluster_hybrid done",
                  "n",
                  n,
                  "n_micro",
                  n_micro,
                  "hub_windows",
                  n_hub_windows,
                  "macro_components",
                  macro_components );
    }

    // ------------------------------------------------------------------------
    // reorder_hybrid(A, W, W_mes, lookahead_C, hub_frac, out_perm)
    //
    // Top-level convenience wrapper matching reorder(): mask + cluster_hybrid,
    // then always apply the resulting permutation to A. The permutation
    // (new position -> original row index) is written to out_perm if provided.
    // ------------------------------------------------------------------------
    template <typename MatrixType>
    void reorder_hybrid( MatrixType& A,
                         size_t W = 32,
                         size_t W_mes = 32,
                         size_t lookahead_C = 64,
                         double hub_frac = 0.05,
                         std::vector<size_t>* out_perm = nullptr ) {
        CSR<size_t, size_t> Ahat;
        std::vector<size_t> P;
        HybridStats stats;

        const size_t before_blocks = count_nonzero_blocks( A, W_mes, W_mes );
        LOG_INFO( "msg", "Starting hybrid reordering", "nonzero blocks", before_blocks );

        mask( A, W, Ahat );
        cluster_hybrid( Ahat, P, W_mes, lookahead_C, hub_frac, &stats );
        permute( A, P );

        const size_t after_blocks = count_nonzero_blocks( A, W_mes, W_mes );
        LOG_INFO( "msg", "Hybrid reordering completed", "nonzero blocks", after_blocks );

        if ( out_perm )
            *out_perm = std::move( P );
    }
} // namespace club