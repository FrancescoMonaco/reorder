#pragma once
#include <algorithm>
#include <array>
#include <cassert>
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
#include <unordered_set>
#include <utility>
#include <vector>

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
    }
    void
    cluster_jaccard( CSR<size_t, size_t>& Ahat, size_t K,  std::vector<size_t>& P, float tau ) {
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

#pragma omp parallel for schedule(dynamic)
            for (size_t idx = 0; idx < unclustered.size(); ++idx) {
                float best = std::numeric_limits<float>::max();
                size_t best_k = K;
                for (size_t k = 0; k < K; ++k) {
            #ifdef __AVX512F__
                    float d = simd::jaccard(Ahat, unclustered[idx], centroids[k]);
            #else
                    float d = jaccard_dist(unclustered[idx], centroids[k]);
            #endif
                    if (d < best) { best = d; best_k = k; }
                }
                if (best <= tau) assignment[idx] = best_k;
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
        for (; i + 8 <= n; i += 8) {
            alignas(64) size_t first_cols[8];
            simd::fetch_first_cols(Ahat, i, first_cols, empty_sentinel);
            for (int k = 0; k < 8; ++k) {
                if (first_cols[k] != empty_sentinel) {
                    col_buckets[first_cols[k]].push_back(i + k);
                }
            }
        }
        for (; i < n; ++i) {
            if (Ahat.nzcount[i] > 0)
                col_buckets[Ahat.col_ind[Ahat.row_ptr[i]]].push_back(i);
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

    // Apply the reordering to the input matrix A. The reordering is defined by the parameters W and
    // K.
    template <typename MatrixType>
    void reorder( MatrixType& A, size_t W, size_t K, float tau = 0.9 ) {
        CSR<size_t, size_t> Ahat;
        std::vector<size_t> P;
        size_t W_mes = 32;
        LOG_INFO( "msg", "Starting reordering", "nonzero blocks", count_nonzero_blocks( A, W_mes, W_mes ) );

        mask( A, W, Ahat );
        LOG_INFO( "msg", "Masking completed", "Sketch size", Ahat.cols );
        cluster( Ahat, K, P, tau );
        LOG_INFO( "msg",
                  "Applying permutation to the original matrix",
                  "nonzero blocks",
                  count_nonzero_blocks( A, W_mes, W_mes ) );
        permute( A, P );
        LOG_INFO( "msg", "Permutation applied", "nonzero blocks", count_nonzero_blocks( A, W_mes, W_mes ) );
    }
} // namespace club