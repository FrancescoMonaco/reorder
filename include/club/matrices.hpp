#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <omp.h> // Include OpenMP header
#include "club/expect.hpp"
#include "club/logging.hpp"

namespace club {

enum MatrixFormat {
    mtx,
    el
};

template <typename DataT = float, typename intT = int>
struct CSR
{
    intT rows = 0;      
    intT cols = 0;      
    
    // Restored for backward compatibility with your external codebase
    std::vector<intT> nzcount; // Size: rows
    
    // Standard CSR 1D layout
    std::vector<intT> row_ptr; // Size: rows + 1
    std::vector<intT> col_ind; // Size: total non-zeros (nnz)
    std::vector<DataT> values; // Size: total non-zeros (nnz)

    bool pattern_only = true;

    void read_from_mtx(std::ifstream& infile) {
        // Check the stream
        if (!infile.is_open() || !infile.good()) {
            LOG_ERROR("msg", "[read_from_mtx] Input stream is not open or in a bad state");
            return;
        }

        std::string line;
        int input_rows = 0, input_cols = 0, input_nnz = 0;
        bool found_dims = false;

        // Bug 2 + 3 fix: getline-based loop skips blank lines,
        // comment lines, and strips \r from Windows-style endings
        while (std::getline(infile, line)) {
            // Strip \r so "\r\n" files work on Linux
            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            // Skip blanks and comment lines (% or #)
            if (line.empty() || line[0] == '%' || line[0] == '#')
                continue;

            std::istringstream iss(line);
            if (iss >> input_rows >> input_cols >> input_nnz) {
                found_dims = true;
                break;
            }
            // Line wasn't blank/comment but also didn't parse — report it
            LOG_ERROR("msg", "[read_from_mtx] Unexpected line before dimensions", "line", line);
        }

        if (!found_dims) {
            LOG_ERROR("msg", "[read_from_mtx] Never found a valid dimension line");
            return;
        }

        rows = static_cast<intT>(input_rows);
        cols = static_cast<intT>(input_cols);

        std::vector<std::vector<intT>> pos_holder(input_rows);
        std::vector<std::vector<DataT>> val_holder(input_rows);

        for (int nz = 0; nz < input_nnz; nz++) {
            if (!std::getline(infile, line)) break;
            if (!line.empty() && line.back() == '\r') line.pop_back();

            std::istringstream ls(line);
            int i_raw, j_raw;
            if (!(ls >> i_raw >> j_raw)) continue;

            intT i = static_cast<intT>(i_raw - 1);
            intT j = static_cast<intT>(j_raw - 1);

            DataT val = static_cast<DataT>(1);
            if (ls >> val)
                this->pattern_only = false;

            // Bug 4 fix: bounds-check both i and j with unsigned-safe comparisons
            if (i < static_cast<intT>(input_rows) && j < static_cast<intT>(input_cols)) {
                pos_holder[i].push_back(j);
                if (!this->pattern_only) val_holder[i].push_back(val);
            }
        }

        // Build CSR arrays (unchanged from your original)
        row_ptr.resize(rows + 1, 0);
        nzcount.resize(rows, 0);

        for (intT i = 0; i < rows; i++) {
            nzcount[i]     = static_cast<intT>(pos_holder[i].size());
            row_ptr[i + 1] = row_ptr[i] + nzcount[i];
        }

        col_ind.resize(row_ptr[rows]);
        if (!this->pattern_only) values.resize(row_ptr[rows]);

        #pragma omp parallel for schedule(static)
        for (intT i = 0; i < rows; i++) {
            intT dest = row_ptr[i];
            for (size_t j = 0; j < pos_holder[i].size(); j++) {
                col_ind[dest + j] = pos_holder[i][j];
                if (!this->pattern_only)
                    values[dest + j] = val_holder[i][j];
            }
        }
    }
    intT nztot() const {
        return static_cast<intT>(col_ind.size());
    }
};

template <typename DataT = float, typename intT = int>
struct BSR {
    intT block_rows = 0;
    intT block_cols = 0;
    intT block_size = 0;
    
    std::vector<intT> bptr; // Block row pointers
    std::vector<intT> bind; // Block column indices
    std::vector<DataT> bnz; // Dense block values (flattened row-major blocks)

    // Constructs the BSR matrix out of an existing CSR matrix layout
    void create_from_csr(const CSR<DataT, intT>& csr, intT b_size) {
        if (csr.rows == 0 || b_size <= 0) return;

        this->block_size = b_size;
        this->block_rows = (csr.rows + b_size - 1) / b_size;
        this->block_cols = (csr.cols + b_size - 1) / b_size;

        bptr.resize(block_rows + 1, 0);
        std::vector<std::vector<intT>> block_cols_per_row(block_rows);

        // Step 1: Identify distinct column blocks per block row in parallel
        // Dynamic scheduling accounts for rows having wildly varying counts of non-zeros
        #pragma omp parallel for schedule(dynamic)
        for (intT I = 0; I < block_rows; I++) {
            std::vector<bool> visited(block_cols, false);
            std::vector<intT> local_blocks;

            intT start_row = I * block_size;
            intT end_row = std::min(start_row + block_size, csr.rows);

            // Scan all scalar elements hitting this block row
            for (intT i = start_row; i < end_row; i++) {
                intT row_start = csr.row_ptr[i];
                intT row_end = csr.row_ptr[i + 1];
                for (intT k = row_start; k < row_end; k++) {
                    intT J = csr.col_ind[k] / block_size;
                    if (!visited[J]) {
                        visited[J] = true;
                        local_blocks.push_back(J);
                    }
                }
            }
            std::sort(local_blocks.begin(), local_blocks.end());
            block_cols_per_row[I] = std::move(local_blocks);
        }

        // Step 2: Establish block offsets via sequential prefix sum
        for (intT I = 0; I < block_rows; I++) {
            bptr[I + 1] = bptr[I] + block_cols_per_row[I].size();
        }

        intT total_blocks = bptr[block_rows];
        bind.resize(total_blocks);
        bnz.resize(total_blocks * block_size * block_size, static_cast<DataT>(0));

        // Step 3: Populate bind and map scalar elements into dense blocks in parallel
        #pragma omp parallel for schedule(static)
        for (intT I = 0; I < block_rows; I++) {
            intT block_offset = bptr[I];
            const auto& cols_in_row = block_cols_per_row[I];
            
            // Fill block column indexes
            for (size_t c = 0; c < cols_in_row.size(); c++) {
                bind[block_offset + c] = cols_in_row[c];
            }

            intT start_row = I * block_size;
            intT end_row = std::min(start_row + block_size, csr.rows);

            // Map scalar values into their exact flat structural blocks
            for (intT i = start_row; i < end_row; i++) {
                intT local_i = i - start_row;
                intT row_start = csr.row_ptr[i];
                intT row_end = csr.row_ptr[i + 1];

                for (intT k = row_start; k < row_end; k++) {
                    intT j = csr.col_ind[k];
                    intT J = j / block_size;
                    intT local_j = j % block_size;

                    // Locate where block block column J is positioned inside this row
                    auto it = std::lower_bound(cols_in_row.begin(), cols_in_row.end(), J);
                    intT block_idx = block_offset + std::distance(cols_in_row.begin(), it);

                    // Calculate index in flat 1D dense data storage array
                    intT bnz_idx = block_idx * (block_size * block_size) + (local_i * block_size) + local_j;
                    
                    if (!csr.pattern_only && !csr.values.empty()) {
                        bnz[bnz_idx] = csr.values[k];
                    } else {
                        bnz[bnz_idx] = static_cast<DataT>(1); 
                    }
                }
            }
        }
    }
};

} // namespace reorder