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

        IC( A.rows, A.cols );
        IC( Ahat.rows, Ahat.cols );
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
        for ( size_t cnt = n; ; --cnt ) {
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
                __m512i idx = _mm512_loadu_si512( reinterpret_cast<const void*>( &Ahat.col_ind[k] ) );
                __m512i r = _mm512_i64gather_epi64( idx, reinterpret_cast<const long long*>( rank.data() ), 8 );
                _mm512_storeu_si512( reinterpret_cast<void*>( &sig[i][k - s] ), r );
            }
#endif
            for ( ; k < e; ++k )
                sig[i][k - s] = rank[Ahat.col_ind[k]];
            std::sort( sig[i].begin(), sig[i].end() );
        }

        // Sequential iterative radix-bucket sort to prevent stack overflow on deep recursive graphs.
        auto process_bucket = [&]( std::vector<size_t> init_group, size_t init_depth, size_t init_lo ) {
            struct Task {
                std::vector<size_t> group;
                size_t depth;
                size_t lo;
            };
            std::vector<Task> stack;
            stack.push_back({ std::move( init_group ), init_depth, init_lo });

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
                    stack.push_back({ std::move( b ), task.depth + 1, current_lo });
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
    void reorder2( CSR<DataT, intT>& A, size_t W, size_t max_iters = 4,
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
void mask_multilevel( CSR<DataT, intT>& A, const std::vector<size_t>& Ws, std::vector<size_t>& P,
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
        std::copy( local_cols[i].begin(), local_cols[i].end(), Ahat.col_ind.begin() + Ahat.row_ptr[i] );
 
    // Cluster the multi-resolution sketch exactly as cluster_lex would cluster
    // a single-level one -- P[i] is the original row index that ends up at
    // position i, ready to hand straight to permute(A, P).
    cluster_lex( Ahat, P );
 
    if ( Ahat_out )
        *Ahat_out = std::move( Ahat );
}
 
// Guarded apply: only commits the candidate permutation if it strictly
// improves the block count relative to A's *current* ordering. A is left
// untouched if the candidate is not an improvement -- this is what makes
// "never worse than baseline" structural instead of hoped-for.
template <typename MatrixType>
bool apply_if_better( MatrixType& A, const std::vector<size_t>& P, size_t block_w, size_t block_h,
                       size_t& before_out, size_t& after_out ) {
    before_out = count_nonzero_blocks( A, block_w, block_h );
    MatrixType candidate = A;
    permute( candidate, P );
    after_out = count_nonzero_blocks( candidate, block_w, block_h );
    if ( after_out < before_out ) {
        A = std::move( candidate );
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------
// cluster_lex_micromacro(Ahat, P, micro_threshold, stats_out, Ahat_macro_out)
//
// Fixes cluster_lex's core failure mode -- rows only ever group by EXACT
// prefix match, so a row bridging two clusters (sharing sig[a] with row A
// and sig[b] with row B, with A and B sharing nothing directly) never pulls
// A and B together. Lexicographic bucketing has no notion of "reachable
// through a neighbour"; BFS-based methods (RCM) do, by construction.
//
// The fix is NOT to make row-level matching transitive -- that's a graph
// problem on n nodes, exactly the cost cluster_lex exists to avoid. Instead:
//
//   1. MICRO phase: run the same rank-remapped radix bucketing cluster_lex
//      already does, but stop subdividing a bucket once it's <= micro_threshold
//      rows (or its rows run out of signature). These are the micro-clusters:
//      groups of rows agreeing exactly on their most common shared
//      structure. Cost: identical to cluster_lex's signature pass,
//      O(m log l), m = nnz(Ahat), l = avg row length.
//
//   2. MACRO graph: treat each micro-cluster as one node. Build a bipartite
//      incidence between micro-clusters and the windows they touch -- for
//      each window, a list of the micro-clusters touching it. O(m) total:
//      every micro-cluster's summary window list is visited once, and no
//      cluster-cluster pair is ever materialised directly (avoids the
//      O(n_macro^2) blowup a naive "connect every pair sharing a window"
//      approach hits on a popular window).
//
//   3. MACRO ordering: BFS directly on that bipartite structure, alternating
//      cluster-nodes and window-nodes, starting each connected component
//      from its lowest-degree unvisited cluster (RCM's usual seed
//      heuristic). Each window's incidence list is expanded at most once
//      (visited_window guard), so this is O(n_macro + m). This is the step
//      that recovers transitivity: cluster A reaches cluster C through
//      shared-window neighbour B, exactly what lexicographic bucketing on
//      A and C alone would miss.
//
//   4. EXPAND: concatenate micro-clusters in BFS order, each contributing
//      its member rows. O(n).
//
// Aggregate complexity: O(m log l) + O(m) + O(n_macro + m) + O(n) -- same
// asymptotic class as plain cluster_lex, provided micro_threshold keeps
// n_macro sub-linear in n. Rows with no windows at all are placed first,
// matching cluster_lex's convention for exhausted rows.
//
// micro_threshold == 0 means "auto": defaults to sqrt(n), targeting
// n_macro = O(sqrt(n)). This is a first cut at the adaptive-depth problem
// flagged as risky after the fixed-granularity multilevel masking
// experiment -- it scales with n instead of being a hardcoded constant, but
// it isn't yet responsive to the *actual* bucket-size distribution the way
// a real adaptive rule should be. Benchmark before trusting it.
// ------------------------------------------------------------------------
struct MicroMacroStats {
    size_t n_micro = 0;          // number of micro-clusters formed
    size_t micro_threshold = 0;  // threshold actually used, post auto-resolve
    size_t macro_incidences = 0; // total (cluster, window) incidence pairs indexed
    size_t macro_components = 0; // disconnected components the macro BFS walked
};

inline void cluster_lex_micromacro( const CSR<size_t, size_t>& Ahat, std::vector<size_t>& P,
                                     size_t micro_threshold = 0,
                                     MicroMacroStats* stats_out = nullptr,
                                     CSR<size_t, size_t>* Ahat_macro_out = nullptr ) {
    const size_t n = Ahat.rows;
    P.resize( n );
    if ( n == 0 )
        return;

    if ( micro_threshold == 0 )
        micro_threshold =
            std::max<size_t>( 1, static_cast<size_t>( std::sqrt( static_cast<double>( n ) ) ) );

    const std::vector<size_t> rank = column_rank( Ahat );

    // Per-row signature: same construction as cluster_lex (rank-remapped,
    // sorted ascending). Duplicated rather than factored out of cluster_lex
    // so that function stays untouched while this one is being validated.
    std::vector<std::vector<size_t>> sig( n );
#pragma omp parallel for schedule( dynamic )
    for ( size_t i = 0; i < n; ++i ) {
        size_t s = Ahat.row_ptr[i], e = Ahat.row_ptr[i + 1];
        sig[i].resize( e - s );
        for ( size_t k = s; k < e; ++k )
            sig[i][k - s] = rank[Ahat.col_ind[k]];
        std::sort( sig[i].begin(), sig[i].end() );
    }

    // ---- Phase 1: MICRO clustering -------------------------------------
    // Same iterative radix-bucket recursion as cluster_lex's process_bucket,
    // except a group becomes a leaf (a micro-cluster) once its size drops
    // to micro_threshold, not just when it drops to 1.
    std::vector<size_t> exhausted_rows;
    std::vector<std::vector<size_t>> micro_groups;
    {
        struct Task {
            std::vector<size_t> group;
            size_t depth;
        };

        std::unordered_map<size_t, std::vector<size_t>> top_buckets;
        for ( size_t i = 0; i < n; ++i ) {
            if ( sig[i].empty() )
                exhausted_rows.push_back( i );
            else
                top_buckets[sig[i][0]].push_back( i );
        }

        std::vector<size_t> keys;
        keys.reserve( top_buckets.size() );
        for ( auto& kv : top_buckets )
            keys.push_back( kv.first );

        // Move buckets into a plain vector first so the parallel region
        // below never touches the unordered_map concurrently.
        std::vector<std::vector<size_t>> top_groups( keys.size() );
        for ( size_t b = 0; b < keys.size(); ++b )
            top_groups[b] = std::move( top_buckets[keys[b]] );

        std::vector<std::vector<std::vector<size_t>>> per_key_groups( keys.size() );

#pragma omp parallel for schedule( dynamic )
        for ( size_t ki = 0; ki < keys.size(); ++ki ) {
            std::vector<Task> stack;
            stack.push_back( { std::move( top_groups[ki] ), 1 } );
            auto& out_groups = per_key_groups[ki];

            while ( !stack.empty() ) {
                Task task = std::move( stack.back() );
                stack.pop_back();

                if ( task.group.size() <= micro_threshold ) {
                    out_groups.push_back( std::move( task.group ) );
                    continue;
                }

                std::vector<size_t> local_exhausted;
                std::unordered_map<size_t, std::vector<size_t>> buckets;
                for ( size_t r : task.group ) {
                    if ( sig[r].size() <= task.depth )
                        local_exhausted.push_back( r );
                    else
                        buckets[sig[r][task.depth]].push_back( r );
                }
                if ( !local_exhausted.empty() )
                    out_groups.push_back( std::move( local_exhausted ) );

                for ( auto& kv : buckets )
                    stack.push_back( { std::move( kv.second ), task.depth + 1 } );
            }
        }

        for ( auto& pg : per_key_groups )
            for ( auto& g : pg )
                micro_groups.push_back( std::move( g ) );
    }

    const size_t n_micro = micro_groups.size();

    // ---- Phase 2: MACRO graph, window-indexed --------------------------
    // Summary signature per micro-cluster = union of member rows' windows.
    std::vector<std::vector<size_t>> macro_sig( n_micro );
    for ( size_t c = 0; c < n_micro; ++c ) {
        std::vector<size_t> u;
        for ( size_t r : micro_groups[c] )
            u.insert( u.end(), sig[r].begin(), sig[r].end() );
        std::sort( u.begin(), u.end() );
        u.erase( std::unique( u.begin(), u.end() ), u.end() );
        macro_sig[c] = std::move( u );
    }

    const size_t n_windows = Ahat.cols;
    std::vector<std::vector<size_t>> window_to_clusters( n_windows );
    size_t macro_incidences = 0;
    for ( size_t c = 0; c < n_micro; ++c ) {
        for ( size_t w : macro_sig[c] )
            window_to_clusters[w].push_back( c );
        macro_incidences += macro_sig[c].size();
    }

    // ---- Phase 3: MACRO ordering via bipartite BFS ----------------------
    // Alternates cluster-nodes and window-nodes. Each window's incidence
    // list is expanded at most once (visited_window guard), so total work
    // is bounded by macro_incidences, never by cluster-cluster pairs.
    std::vector<char> visited_cluster( n_micro, 0 );
    std::vector<char> visited_window( n_windows, 0 );
    std::vector<size_t> macro_order;
    macro_order.reserve( n_micro );
    size_t macro_components = 0;

    // RCM-style seeding: start each component from its lowest-degree
    // unvisited node (degree = summary size).
    std::vector<size_t> by_degree( n_micro );
    std::iota( by_degree.begin(), by_degree.end(), 0 );
    std::sort( by_degree.begin(), by_degree.end(), [&]( size_t a, size_t b ) {
        return macro_sig[a].size() < macro_sig[b].size();
    } );

    for ( size_t seed_idx : by_degree ) {
        if ( visited_cluster[seed_idx] )
            continue;

        ++macro_components;
        std::vector<size_t> queue;
        queue.push_back( seed_idx );
        visited_cluster[seed_idx] = 1;
        size_t qh = 0;

        while ( qh < queue.size() ) {
            size_t c = queue[qh++];
            macro_order.push_back( c );

            for ( size_t w : macro_sig[c] ) {
                if ( visited_window[w] )
                    continue;
                visited_window[w] = 1;
                for ( size_t c2 : window_to_clusters[w] ) {
                    if ( !visited_cluster[c2] ) {
                        visited_cluster[c2] = 1;
                        queue.push_back( c2 );
                    }
                }
            }
        }
    }
    expect( macro_order.size() == n_micro );

    // ---- Phase 4: expand back to a row permutation -----------------------
    size_t pos = 0;
    for ( size_t r : exhausted_rows )
        P[pos++] = r;
    for ( size_t c : macro_order )
        for ( size_t r : micro_groups[c] )
            P[pos++] = r;
    expect( pos == n );

    if ( stats_out ) {
        stats_out->n_micro = n_micro;
        stats_out->micro_threshold = micro_threshold;
        stats_out->macro_incidences = macro_incidences;
        stats_out->macro_components = macro_components;
    }

    if ( Ahat_macro_out ) {
        Ahat_macro_out->rows = n_micro;
        Ahat_macro_out->cols = n_windows;
        Ahat_macro_out->nzcount.resize( n_micro );
        Ahat_macro_out->row_ptr.assign( n_micro + 1, 0 );
        for ( size_t c = 0; c < n_micro; ++c ) {
            Ahat_macro_out->nzcount[c] = macro_sig[c].size();
            Ahat_macro_out->row_ptr[c + 1] = Ahat_macro_out->row_ptr[c] + macro_sig[c].size();
        }
        Ahat_macro_out->col_ind.resize( Ahat_macro_out->row_ptr[n_micro] );
        Ahat_macro_out->values.assign( Ahat_macro_out->row_ptr[n_micro], 1 );
        Ahat_macro_out->pattern_only = false;
        for ( size_t c = 0; c < n_micro; ++c )
            std::copy( macro_sig[c].begin(), macro_sig[c].end(),
                       Ahat_macro_out->col_ind.begin() + Ahat_macro_out->row_ptr[c] );
    }

    LOG_INFO( "msg", "cluster_lex_micromacro done", "n", n, "n_micro", n_micro,
              "micro_threshold", micro_threshold, "macro_components", macro_components );
}
} // namespace club