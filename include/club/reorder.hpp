#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <icecream.hpp>
#include "omp.h"
#include "club/matrices.hpp"
#include "club/expect.hpp"

namespace club {

    // Apply the masking to the input matrix. To build Ahat each row v of A is divided into windows of size W
    // Each masked row vhat has a 1 for every non-zero window in v.
    // Overloaded for CSR and BSR, the output Ahat is always in CSR format. 
    template <typename DataT = float, typename intT = int>
    void mask(CSR<DataT, intT>& A, size_t W, CSR<size_t, size_t>& Ahat) {
        Ahat.rows = static_cast<size_t>(A.rows);
        Ahat.cols = static_cast<size_t>((A.cols + W - 1) / W); 
        
        // Safely resize vector metadata containers
        Ahat.nzcount.assign(Ahat.rows, 0);
        Ahat.row_ptr.assign(Ahat.rows + 1, 0);

        // Thread-local vector collection to hold structural column modifications safely
        std::vector<std::vector<size_t>> local_cols(A.rows);

        // Step 1: Gather and deduplicate window indices per row in parallel
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < A.rows; ++i) {
            size_t row_start = A.row_ptr[i];
            size_t row_end = A.row_ptr[i + 1];
            if (row_start == row_end) continue;

            std::vector<size_t> windows;
            windows.reserve(row_end - row_start);

            // Translate flat 1D scalar column positions to window steps
            for (size_t k = row_start; k < row_end; ++k) {
                windows.push_back(A.col_ind[k] / W);
            }
            
            // Sort and eliminate duplicate window strikes
            std::sort(windows.begin(), windows.end());
            windows.erase(std::unique(windows.begin(), windows.end()), windows.end());

            Ahat.nzcount[i] = windows.size();
            local_cols[i] = std::move(windows);
        }

        // Step 2: Compute flat 1D array prefix offsets sequentially
        for (size_t i = 0; i < Ahat.rows; ++i) {
            Ahat.row_ptr[i + 1] = Ahat.row_ptr[i] + Ahat.nzcount[i];
        }

        // Allocate flat 1D memory spaces
        size_t total_nnz = Ahat.row_ptr[Ahat.rows];
        Ahat.col_ind.resize(total_nnz);
        Ahat.values.assign(total_nnz, 1); // Structural masks populate elements as binary 1 flags
        Ahat.pattern_only = false;

        // Step 3: Stream row-segments back into the flat 1D layout in parallel
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < Ahat.rows; ++i) {
            size_t dest_offset = Ahat.row_ptr[i];
            const auto& r_cols = local_cols[i];
            
            for (size_t j = 0; j < r_cols.size(); ++j) {
                Ahat.col_ind[dest_offset + j] = r_cols[j];
            }
        }

        IC(A.rows, A.cols);
        IC(Ahat.rows, Ahat.cols);
    }

    template <typename DataT = float, typename intT = int>
    void mask(const BSR<DataT, intT>& A, size_t W, CSR<size_t, size_t>& Ahat) {
        Ahat.rows = static_cast<size_t>(A.block_rows);
        Ahat.cols = static_cast<size_t>((A.block_cols * A.block_size + W - 1) / W);
        Ahat.nzcount.assign(Ahat.rows, 0);
        Ahat.row_ptr.assign(Ahat.rows + 1, 0);

        // BSR structural masking logic goes here.
    }

    // Cluster the masked matrix Ahat into K clusters. Each cluster is a set of rows of Ahat that are similar to each other.
    // Then generate the permutation P that reorders the rows of Ahat according to the clusters.
    void cluster(CSR<size_t, size_t>& Ahat, size_t K, std::vector<size_t>& P) {
        (void)Ahat;
        (void)K;
        (void)P;
        // TODO: Implement clustering logic using the updated vector layouts
    }

    // Apply the permutation P to the input matrix A to obtain the reordered matrix A'.
    // Need to overload for CSR and BSR
    template <typename DataT = float, typename intT = int>
    void permute(CSR<DataT, intT>& A, const std::vector<size_t>& P) {
        (void)A;
        (void)P;
        // TODO: Implement permutation logic using the updated vector layouts
    }

    // Permute BSR
    template <typename DataT = float, typename intT = int>
    void permute(BSR<DataT, intT>& A, const std::vector<size_t>& P) {
        // Shuffles entire BLOCK rows based on P
    }

    // Apply the reordering to the input matrix A. The reordering is defined by the parameters W and K.
    template <typename MatrixType>
    void reorder(MatrixType& A, size_t W, size_t K) {
        CSR<size_t, size_t> Ahat;
        std::vector<size_t> P;

        // Compile-time resolution routes to the correct overload automatically!
        mask(A, W, Ahat);
        cluster(Ahat, K, P);
        permute(A, P);
    }
} // namespace reorder