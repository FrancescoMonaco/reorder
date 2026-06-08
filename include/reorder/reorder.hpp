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
#include "reorder/matrices.hpp"
#include "reorder/expect.hpp"


namespace reorder {

    // Apply the masking tothe input matrix. To build Ahat each row v of A is divided into windows of size W
    // Each masked row vhat has a 1 for every non-zero window in v.
    void mask(CSR<size_t>& A, size_t W, CSR<size_t>& Ahat) {
        Ahat.rows = A.rows;
        Ahat.cols = (A.cols + W - 1) / W; // number
        Ahat.nzcount = new size_t[Ahat.rows];
        Ahat.ja = new size_t*[Ahat.rows];
        Ahat.ma = new size_t*[Ahat.rows];
        #pragma omp parallel for
        for (size_t i = 0; i < A.rows; ++i) {
            std::vector<size_t> window_indices;
            for (size_t j = 0; j < A.nzcount[i]; ++j) {
                size_t col = A.ja[i][j];
                size_t window_index = col / W;
                if (std::find(window_indices.begin(), window_indices.end(), window_index) == window_indices.end()) {
                    window_indices.push_back(window_index);
                    Ahat.nzcount[i]++;
                }
            }
            Ahat.ja[i] = new size_t[Ahat.nzcount[i]];
            Ahat.ma[i] = new size_t[Ahat.nzcount[i]];
            for (size_t k = 0; k < window_indices.size(); ++k) {
                Ahat.ja[i][k] = window_indices[k];
                Ahat.ma[i][k] = 1;
            }
        }

    }

    // Cluster the masked matrix Ahat into K clusters. Each cluster is a set of rows of Ahat that are similar to each other.
    // Then generate the permutation P that reorders the rows of Ahat according to the clusters.
    void cluster( size_t K) {
        (void)K;
    }

    // Apply the permutation P to the input matrix A to obtain the reordered matrix A'.
    void permute() {}

    // Apply the reordering to the input matrix A. The reordering is defined by the parameters W and K.
    void reorder( size_t W, size_t K) {
        std::vector<float> A(1000);

        IC(W, K, A.size());

        for (size_t i = 0; i < A.size(); ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
        }


        #pragma omp parallel
        {
            volatile double sink = 0.0;
            #pragma omp for
            for (std::uint64_t i = 0; i < 500000000ULL; ++i) {
                sink += i * 0.0000001;
            }
        }


        mask(W);
        cluster(K);
        permute();
    }
}