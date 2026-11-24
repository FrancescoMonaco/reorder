#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "club/expect.hpp"
#include "club/logging.hpp"
#include "club/matrices.hpp"
#include "club/reorder.hpp"

namespace club {


    // club_mask(A, W, Ahat) — binary sketch of A via fixed-width windows.
    //
    // Each row of A is divided into consecutive windows of W columns. Ahat is a
    // CSR with the same number of rows as A and ceil(cols / W) columns: row i
    // contains the (sorted, deduplicated) indices of all windows of row i that
    // contain at least one nonzero. Equivalently, Ahat is the binary matrix in
    // which entry (i, w) = 1 iff row i has any nonzero in columns
    // [w*W, (w+1)*W).
    //
    template <typename DataT = float, typename intT = int>
    void club_mask( const CSR<DataT, intT>& A,
                    intT W,
                    CSR<size_t, size_t>& Ahat ) {
        expect( W > 0 );
        const size_t w = static_cast<size_t>( W );

        Ahat.rows = static_cast<size_t>( A.rows );
        Ahat.cols = static_cast<size_t>( ( static_cast<size_t>( A.cols ) + w - 1 ) / w );

        Ahat.nzcount.assign( Ahat.rows, 0 );
        Ahat.row_ptr.assign( Ahat.rows + 1, 0 );

        // Thread-local collection of window columns per row (disjoint writes).
        std::vector<std::vector<size_t>> local_cols( Ahat.rows );

// Step 1: gather & deduplicate the non-zero window indices of each row in
// parallel.
#pragma omp parallel for schedule( dynamic )
        for ( size_t i = 0; i < Ahat.rows; ++i ) {
            const size_t row_start = static_cast<size_t>( A.row_ptr[i] );
            const size_t row_end = static_cast<size_t>( A.row_ptr[i + 1] );
            if ( row_start == row_end )
                continue;

            std::vector<size_t> windows;
            windows.reserve( row_end - row_start );

            // Map each flat scalar column to its window index and dedup
            for ( size_t k = row_start; k < row_end; ++k )
                windows.push_back( static_cast<size_t>( A.col_ind[k] ) / w );

            std::sort( windows.begin(), windows.end() );
            windows.erase( std::unique( windows.begin(), windows.end() ),
                           windows.end() );

            Ahat.nzcount[i] = windows.size();
            local_cols[i] = std::move( windows );
        }

        // Step 2: sequential prefix sum over row lengths.
        for ( size_t i = 0; i < Ahat.rows; ++i )
            Ahat.row_ptr[i + 1] = Ahat.row_ptr[i] + Ahat.nzcount[i];

        const size_t total_nnz = Ahat.row_ptr[Ahat.rows];
        Ahat.col_ind.resize( total_nnz );
        Ahat.values.assign( total_nnz, 1 ); // binary sketch entries are all 1
        Ahat.pattern_only = false;

// Step 3: scatter per-row window lists back into the flat CSR layout in
// parallel 
#pragma omp parallel for schedule( static )
        for ( size_t i = 0; i < Ahat.rows; ++i ) {
            size_t dest = Ahat.row_ptr[i];
            const auto& rcols = local_cols[i];
            for ( size_t j = 0; j < rcols.size(); ++j )
                Ahat.col_ind[dest + j] = rcols[j];
        }
    }

    // club_cluster(Ahat, P) — fast lexicographic clustering of the sketch rows.
    //
    // Rows of Ahat are ordered so that rows with identical / similar window sets
    // become adjacent, which exposes dense block structure after permutation.
    // The ordering is a plain lexicographic sort over each row's sorted window
    // list (shorter rows sort before longer ones that share a prefix), which is
    // deterministic, requires no extra memory beyond the permutation, and is
    // read-only on Ahat so the comparison itself is thread-safe if ever
    // parallelised. P[i] is the original row index that ends up at position i,
    // ready to hand straight to permute(A, P).
    inline void club_cluster( const CSR<size_t, size_t>& Ahat,
                              std::vector<size_t>& P ) {
        const size_t n = Ahat.rows;
        P.resize( n );
        std::iota( P.begin(), P.end(), static_cast<size_t>( 0 ) );
        if ( n <= 1 )
            return;

        // Read-only comparator over the flat CSR slices of Ahat.
        auto row_less = [&Ahat]( size_t a, size_t b ) -> bool {
            const size_t sa = Ahat.row_ptr[a], ea = Ahat.row_ptr[a + 1];
            const size_t sb = Ahat.row_ptr[b], eb = Ahat.row_ptr[b + 1];
            return std::lexicographical_compare( Ahat.col_ind.begin() + sa,
                                                 Ahat.col_ind.begin() + ea,
                                                 Ahat.col_ind.begin() + sb,
                                                 Ahat.col_ind.begin() + eb );
        };

        std::sort( P.begin(), P.end(), row_less );
    }

    // club_original(A, W)
    //
    //   1. club_mask   : build the binary window sketch Ahat of A.
    //   2. club_cluster: lexicographically cluster the rows of Ahat -> P.
    //   3. permute     : reorder A in place according to P.
    //
    // Returns the permutation P (new position -> original row index).
    template <typename DataT = float, typename intT = int>
    std::vector<size_t> club_original( CSR<DataT, intT>& A, intT W = 4 ) {
        CSR<size_t, size_t> Ahat;
        std::vector<size_t> P;

        LOG_INFO( "msg", "club_original: starting", "window", W,
                  "rows", A.rows, "cols", A.cols );

        club_mask( A, W, Ahat );
        LOG_INFO( "msg", "club_original: mask done", "sketch_rows", Ahat.rows,
                  "sketch_cols", Ahat.cols, "sketch_nnz", Ahat.col_ind.size() );

        club_cluster( Ahat, P );
        LOG_INFO( "msg", "club_original: cluster done" );

        permute( A, P );
        LOG_INFO( "msg", "club_original: permutation applied" );

        return P;
    }

} // namespace club
