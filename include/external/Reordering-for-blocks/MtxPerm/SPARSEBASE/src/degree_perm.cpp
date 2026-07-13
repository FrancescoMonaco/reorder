/**
 * Degree Permutation Generator
 * 
 * Interface: ./degree_perm matrix.mtx output.perm
 * Generates a degree-based reordering for a Matrix Market sparse matrix.
 * Orders vertices by their degree in ascending order (lowest degree first).
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: Sorts vertices by their degree. Low-degree vertices
 *            (fewer connections) appear first in the permutation.
 */

#include "reorder_utils.h"
#include "sparsebase/reorder/degree_reorder.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[Degree] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[Degree]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm>\n";
        std::cerr << "Generates Degree permutation.\n";
        return 1;
    }


    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);

    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    std::cerr << "[Degree] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        std::cerr << "[Degree] Matrix loaded: " << n << " vertices\n";
        
        context::CPUContext cpu_context;
        // DegreeReorderParams(bool ascending)
        //   - ascending=true:  vertices ordered low-degree to high-degree
        //   - ascending=false: vertices ordered high-degree to low-degree
        std::cerr << "[Degree] Computing reordering...\n";
        reorder::DegreeReorderParams params(true);

        CPU_TIMER_START(reordering);
        int* perm = bases::ReorderBase::Reorder<reorder::DegreeReorder>(
            params, csr, {&cpu_context}, true);
        CPU_TIMER_STOP(reordering);
        
        // Invert permutation (SparseBase returns Old-to-New, we need New-to-Old)
        int* inv_perm = new int[n];
        for (int i = 0; i < n; i++) {
            inv_perm[perm[i]] = i;
        }
        free(perm);  // SparseBase allocates internally, don't use delete[]
        perm = inv_perm;
        
        std::cerr << "[Degree] Saving permutation to: " << perm_file << "\n";
        bool success = save_permutation(perm_file, perm, n);
        std::cerr << "[Degree] " << (success ? "Success" : "Failed") << "\n";
        
        
        TIMER_PRINT(loading);
        TIMER_PRINT(reordering);


        delete[] perm;
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
