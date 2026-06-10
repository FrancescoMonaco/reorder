#pragma once
#include "club/logging.hpp"
#include "club/matrices.hpp"

namespace club {

    template <typename DataT, typename intT>
    size_t count_nonzero_blocks( const CSR<DataT, intT>& matrix, size_t width, size_t height ) {
        size_t count = 0;
        size_t num_block_rows = ( static_cast<size_t>( matrix.rows ) + height - 1 ) / height;

        for ( size_t I = 0; I < num_block_rows; ++I ) {
            intT row_start = static_cast<intT>( I * height );
            intT row_end = std::min( static_cast<intT>( ( I + 1 ) * height ), matrix.rows );

            // Collect every block-column touched by any row in this block-row
            std::vector<size_t> block_cols;
            for ( intT i = row_start; i < row_end; ++i ) {
                for ( intT k = matrix.row_ptr[i]; k < matrix.row_ptr[i + 1]; ++k ) {
                    block_cols.push_back( static_cast<size_t>( matrix.col_ind[k] ) / width );
                }
            }

            // Deduplicate: sort and count unique entries
            std::sort( block_cols.begin(), block_cols.end() );
            count += static_cast<size_t>( std::unique( block_cols.begin(), block_cols.end() ) -
                                          block_cols.begin() );
        }
        return count;
    }

    template <typename DataT, typename intT>
    size_t count_nonzero_blocks( const BSR<DataT, intT>& matrix, size_t width, size_t height ) {
        const size_t bsz = static_cast<size_t>( matrix.block_size );

        // Fast path: every stored BSR block is structurally non-zero by construction if the sized
        // match
        if ( width == bsz && height == bsz )
            return static_cast<size_t>( matrix.bptr[matrix.block_rows] );

        // General path: a BSR block at (I,J) spans rows [I*bsz, (I+1)*bsz)
        // and cols [J*bsz, (J+1)*bsz). That rectangle may touch several
        // (width x height) blocks if the grids are not multiples of each other,
        // so we compute the range of wide-blocks it overlaps and record each one.
        const size_t total_cols = static_cast<size_t>( matrix.block_cols ) * bsz;
        const size_t num_wide_cols = ( total_cols + width - 1 ) / width;

        std::unordered_set<size_t> occupied; // encoded as wide_row * num_wide_cols + wide_col

        for ( intT I = 0; I < matrix.block_rows; ++I ) {
            for ( intT b = matrix.bptr[I]; b < matrix.bptr[I + 1]; ++b ) {
                intT J = matrix.bind[b];

                // Scalar row/col extents of this BSR block
                size_t r0 = static_cast<size_t>( I ) * bsz;
                size_t r1 = r0 + bsz - 1;
                size_t c0 = static_cast<size_t>( J ) * bsz;
                size_t c1 = c0 + bsz - 1;

                // Wide-block range that overlaps this BSR block
                size_t wr0 = r0 / height, wr1 = r1 / height;
                size_t wc0 = c0 / width, wc1 = c1 / width;

                for ( size_t wr = wr0; wr <= wr1; ++wr )
                    for ( size_t wc = wc0; wc <= wc1; ++wc )
                        occupied.insert( wr * num_wide_cols + wc );
            }
        }
        return occupied.size();
    }
} // namespace club