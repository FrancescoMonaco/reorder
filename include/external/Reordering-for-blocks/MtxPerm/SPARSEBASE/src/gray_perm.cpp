/**
 * Gray Permutation Generator
 * 
 * Interface: ./gray_perm matrix.mtx output.perm
 * Generates a Gray code reordering for a Matrix Market sparse matrix.
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: Uses Gray code ordering to generate a permutation that minimizes
 *            Hamming distance between consecutive vertices in the ordering.
 *            No parameters required.
 */

#include "reorder_utils.h"
#include "sparsebase/reorder/gray_reorder.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <iostream>
#include <algorithm>  // for std::max

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[Gray] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[Gray]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm>\n";
        std::cerr << "Generates Gray permutation.\n";
        return 1;
    }

    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);
    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    std::cerr << "[Gray] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        int nnz = csr->get_num_nnz();
        std::cerr << "[Gray] Matrix loaded: " << n << " vertices, " << nnz << " nnz\n";
        
        if (!check_square_matrix(csr, "Gray")) {
            delete csr;
            return 1;
        }

        context::CPUContext cpu_context;
        std::cerr << "[Gray] Computing reordering...\n";
        
        // GrayReorderParams(BitMapSize resolution, int nnz_threshold, int sparse_density_group_size)
        // - resolution: bitmap size (16, 32, or 64 bits)
        // - nnz_threshold: rows with nnz <= threshold are considered "sparse"
        // - sparse_density_group_size: group size for reordering within sparse section
        int avg_nnz_per_row = (n > 0) ? (nnz / n) : 1;
        int group_size = std::max(1, avg_nnz_per_row / 16);  // Reasonable default
        reorder::GrayReorderParams params(
            reorder::BitMapSize::BitSize32,  // 32-bit resolution
            32,                               // nnz threshold for sparse vs dense
            group_size                        // group size based on matrix density
        );

        CPU_TIMER_START(reordering);
        int* perm = bases::ReorderBase::Reorder<reorder::GrayReorder>(
            params, csr, {&cpu_context}, true);
        CPU_TIMER_STOP(reordering);
        
        if (perm == nullptr) {
            std::cerr << "[Gray] Error: Reordering returned null permutation\n";
            delete csr;
            return 1;
        }

        // Invert permutation (SparseBase returns Old-to-New, we need New-to-Old)
        int* inv_perm = new int[n];
        for (int i = 0; i < n; i++) {
            if (perm[i] < 0 || perm[i] >= n) {
                std::cerr << "[Gray] Error: Invalid permutation value at index " << i << ": " << perm[i] << "\n";
                delete[] perm;
                delete[] inv_perm;
                delete csr;
                return 1;
            }
            inv_perm[perm[i]] = i;
        }
        delete[] perm;
        perm = inv_perm;
        
        std::cerr << "[Gray] Saving permutation to: " << perm_file << "\n";
        bool success = save_permutation(perm_file, perm, n);
        std::cerr << "[Gray] " << (success ? "Success" : "Failed") << "\n";
        
        TIMER_PRINT(loading);
        TIMER_PRINT(reordering);

        delete[] perm;
        delete csr;
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
