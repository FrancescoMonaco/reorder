/**
 * AMD (Approximate Minimum Degree) Permutation Generator
 * 
 * Interface: ./amd_perm matrix.mtx output.perm
 * Generates an AMD reordering for a Matrix Market sparse matrix.
 * Orders vertices to minimize fill-in during sparse factorization.
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: Uses approximate minimum degree heuristic to produce orderings
 *            that minimize fill-in during Cholesky or LU factorization.
 *            Commonly used as preprocessing for direct sparse solvers.
 *            No parameters required (uses defaults).
 */

#include "reorder_utils.h"
#include "sparsebase/reorder/amd_reorder.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <iostream>

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[AMD] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[AMD]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm> [dense] [aggressive]\n";
        std::cerr << "Generates AMD (Approximate Minimum Degree) permutation.\n";
        std::cerr << "Optional parameters:\n";
        std::cerr << "  dense: threshold for dense rows (default: 10.0)\n";
        std::cerr << "  aggressive: aggressive absorption (default: 1.0)\n";
        return 1;
    }

    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);
    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    std::cerr << "[AMD] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        std::cerr << "[AMD] Matrix loaded: " << n << " x " << csr->get_dimensions()[1] << "\n";
        
        if (!check_square_matrix(csr, "AMD reordering")) {
            delete csr;
            return 1;
        }
        
        context::CPUContext cpu_context;
        std::cerr << "[AMD] Computing reordering...\n";
        // AMDReorderParams has two optional parameters:
        //   - dense: threshold for treating rows as dense (default: AMD_DEFAULT_DENSE, see sparsebase/reorder/amd_reorder.h)
        //   - aggressive: aggressiveness level (default: AMD_DEFAULT_AGGRESSIVE, see sparsebase/reorder/amd_reorder.h)
        reorder::AMDReorderParams params;
        
        // Parse optional parameters from command line
        if (argc > 3) {
            params.dense = std::stod(argv[3]);
            std::cerr << "[AMD] Using dense = " << params.dense << "\n";
        }
        if (argc > 4) {
            params.aggressive = std::stod(argv[4]);
            std::cerr << "[AMD] Using aggressive = " << params.aggressive << "\n";
        }

        CPU_TIMER_START(reordering);
        int* perm = bases::ReorderBase::Reorder<reorder::AMDReorder>(
            params, csr, {&cpu_context}, true);
        CPU_TIMER_STOP(reordering);
        
        // Invert permutation (SparseBase returns Old-to-New, we need New-to-Old)
        int* inv_perm = new int[n];
        for (int i = 0; i < n; i++) {
            inv_perm[perm[i]] = i;
        }
        delete[] perm;
        perm = inv_perm;
        
        std::cerr << "[AMD] Saving permutation to: " << perm_file << "\n";
        bool success = save_permutation(perm_file, perm, n);
        std::cerr << "[AMD] " << (success ? "Success" : "Failed") << "\n";
        
        TIMER_PRINT(loading);
        TIMER_PRINT(reordering);

        delete[] perm;
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
