/**
 * METIS Partitioning-based Permutation Generator
 * 
 * Interface: ./metis_part_perm matrix.mtx output.perm [--nparts N]
 * Generates a reordering based on METIS graph partitioning.
 * Unlike nested dissection, this uses k-way partitioning and orders
 * vertices by their partition assignment.
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: Uses METIS k-way partitioning to divide the graph into partitions,
 *            then orders vertices by partition ID. This groups vertices in the
 *            same partition together, which can improve cache locality and
 *            enable block-based parallel processing.
 */

#include "reorder_utils.h"
#include "sparsebase/partition/metis_partition.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[METIS-Part] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[METIS-Part]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm> [--nparts N] [--objtype TYPE]\n";
        std::cerr << "Generates METIS partitioning-based permutation.\n";
        std::cerr << "Optional flags:\n";
        std::cerr << "  --nparts N      number of partitions (default: 128)\n";
        std::cerr << "  --objtype TYPE  optimization objective: cut or vol (default: cut)\n";
        return 1;
    }

    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);
    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    std::cerr << "[METIS-Part] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        std::cerr << "[METIS-Part] Matrix loaded: " << n << " x " << csr->get_dimensions()[1] << "\n";
        
        if (!check_square_matrix(csr, "METIS partitioning")) {
            delete csr;
            return 1;
        }
        
        // Parse number of partitions
        int nparts = 128;  // default: enough partitions for fine-grained row clustering
        std::string nparts_str = get_flag_value(argc, argv, "--nparts");
        if (!nparts_str.empty()) {
            nparts = std::stoi(nparts_str);
        }
        std::cerr << "[METIS-Part] Number of partitions: " << nparts << "\n";
        
        // Parse objective type
        int objtype = metis::METIS_OBJTYPE_CUT;  // default
        std::string objtype_str = get_flag_value(argc, argv, "--objtype");
        if (objtype_str == "vol" || objtype_str == "VOL") {
            objtype = metis::METIS_OBJTYPE_VOL;
        } else if (!objtype_str.empty() && objtype_str != "cut" && objtype_str != "CUT") {
            std::cerr << "[METIS-Part] Warning: unknown objtype '" << objtype_str << "', using 'cut'\n";
        }
        std::cerr << "[METIS-Part] Objective: " << (objtype == metis::METIS_OBJTYPE_VOL ? "volume" : "cut") << "\n";
        
        context::CPUContext cpu_context;
        std::cerr << "[METIS-Part] Computing partitioning...\n";
        
        // Create partitioning parameters
        partition::MetisPartitionParams params;
        params.num_partitions = nparts;
        params.objtype = objtype;
        
        // Create METIS partitioner
        partition::MetisPartition<int, int, float> partitioner;
        
        CPU_TIMER_START(reordering);
        // Get partition assignment for each vertex
        auto partition_result = partitioner.Partition(
            csr, &params, {&cpu_context}, true);
                
        // Create (vertex_id, partition_id) pairs
        std::vector<std::pair<int, int>> vertex_partition(n);
        for (int i = 0; i < n; i++) {
            vertex_partition[i] = {i, partition_result[i]};
        }
        
        // Sort by partition ID (stable sort to maintain relative order within partitions)
        std::stable_sort(vertex_partition.begin(), vertex_partition.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second < b.second;
            });
        
        // Extract permutation (maps new position -> old vertex id)
        int* perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = vertex_partition[i].first;
        }
        
        CPU_TIMER_STOP(reordering);
        
        std::cerr << "[METIS-Part] Saving permutation to: " << perm_file << "\n";
        bool success = save_permutation(perm_file, perm, n);
        std::cerr << "[METIS-Part] " << (success ? "Success" : "Failed") << "\n";
        
        TIMER_PRINT(loading);
        TIMER_PRINT(reordering);

        delete[] perm;        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
