/**
 * RCM (Reverse Cuthill-McKee) Permutation Generator
 * 
 * Interface: ./rcm_perm matrix.mtx output.perm
 * Generates an RCM reordering for a Matrix Market sparse matrix.
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: Reduces matrix bandwidth by ordering vertices in a breadth-first
 *            manner, then reversing the order. Commonly used to improve cache
 *            locality and reduce fill-in for sparse linear solvers.
 *            No parameters required.
 */

#include "reorder_utils.h"
#include "sparsebase/reorder/rcm_reorder.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <iostream>

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[RCM] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[RCM]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm>\n";
        std::cerr << "Generates RCM (Reverse Cuthill-McKee) permutation.\n";
        return 1;
    }

    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);
    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    std::cerr << "[RCM] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        std::cerr << "[RCM] Matrix loaded: " << n << " x " << csr->get_dimensions()[1] << "\n";
        
        if (!check_square_matrix(csr, "RCM reordering")) {
            delete csr;
            return 1;
        }
        
        context::CPUContext cpu_context;
        std::cerr << "[RCM] Computing reordering...\n";
        reorder::RCMReorderParams params;

        CPU_TIMER_START(reordering);
        int* perm = bases::ReorderBase::Reorder<reorder::RCMReorder>(
            params, csr, {&cpu_context}, true);
        CPU_TIMER_STOP(reordering);
        
        // Invert permutation (SparseBase returns Old-to-New, we need New-to-Old)
        int* inv_perm = new int[n];
        for (int i = 0; i < n; i++) {
            inv_perm[perm[i]] = i;
        }
        delete[] perm;
        perm = inv_perm;
        
        std::cerr << "[RCM] Saving permutation to: " << perm_file << "\n";
        bool success = save_permutation(perm_file, perm, n);
        std::cerr << "[RCM] " << (success ? "Success" : "Failed") << "\n";
        

        printf("TESTING TIMERS\n");
        TIMER_PRINT(loading);
        TIMER_PRINT(reordering);
        printf("END TIMERS\n");

        delete[] perm;
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
