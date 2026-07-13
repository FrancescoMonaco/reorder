/**
 * SlashBurn Permutation Generator
 * 
 * Interface: ./slashburn_perm matrix.mtx output.perm [--k K]
 * Generates a SlashBurn reordering for a Matrix Market sparse matrix.
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: SlashBurn is a reordering algorithm designed to improve graph compression
 *            and community detection by identifying "hub" vertices and moving them
 *            to the front (or back) of the ordering.
 */

#include "reorder_utils.h"
#include "sparsebase/reorder/slashburn_reorder.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <iostream>

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[SlashBurn] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[SlashBurn]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm> [--k K]\n";
        std::cerr << "Generates SlashBurn permutation.\n";
        std::cerr << "Optional flags:\n";
        std::cerr << "  --k K      number of hubs to remove per iteration (default: 1)\n";
        return 1;
    }

    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);
    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    // Parse k and greedy
    int k = 0;
    bool greedy = true;
    std::string k_str = get_flag_value(argc, argv, "--k");
    std::string greedy_str = get_flag_value(argc, argv, "--greedy");
    if (!k_str.empty()) {
        k = std::stoi(k_str);
    }
    if (!greedy_str.empty()) {
        greedy = (greedy_str == "1" || greedy_str == "true");
    }

    std::cerr << "[SlashBurn] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        std::cerr << "[SlashBurn] Matrix loaded: " << n << " x " << csr->get_dimensions()[1] << "\n";
        if (!check_square_matrix(csr, "SlashBurn")) {
            delete csr;
            return 1;
        }

        // Default k = 0.1% of n (matches BEAR reference implementation by Lim/Kang/Faloutsos)
        if (k <= 0) {
            k = std::max(1, n / 1000);
        }
        std::cerr << "[SlashBurn] Reordering with k=" << k << ", greedy=" << (greedy ? "true" : "false") << "...\n";
        context::CPUContext cpu_context;
        CPU_TIMER_START(reordering);
        reorder::SlashburnReorder<int, int, float> reorderer(k, greedy, false);
        auto perm = reorderer.GetReorder(csr, {&cpu_context}, true);
        CPU_TIMER_STOP(reordering);
        if (perm == nullptr) {
            std::cerr << "[SlashBurn] Error: Reordering returned null permutation\n";
            delete csr;
            return 1;
        }
        std::cerr << "[SlashBurn] Reordering complete.\n";
        // Invert permutation (SparseBase returns Old-to-New, we need New-to-Old)
        int* inv_perm = new int[n];
        for (int i = 0; i < n; i++) {
            if (perm[i] < 0 || perm[i] >= n) {
                std::cerr << "[SlashBurn] Error: Invalid permutation value at index " << i << ": " << perm[i] << "\n";
                delete[] perm;
                delete[] inv_perm;
                delete csr;
                return 1;
            }
            inv_perm[perm[i]] = i;
        }
        delete[] perm;
        perm = inv_perm;
        if (save_permutation(perm_file, perm, n)) {
            std::cerr << "[SlashBurn] Permutation saved to " << perm_file << "\n";
        } else {
            std::cerr << "[SlashBurn] Failed to save permutation.\n";
        }
        delete[] perm;
        delete csr;

    } catch (const std::exception& e) {
        std::cerr << "[SlashBurn] Error: " << e.what() << "\n";
        return 1;
    }

    TIMER_PRINT(loading);
    TIMER_PRINT(reordering);

    return 0;
}
