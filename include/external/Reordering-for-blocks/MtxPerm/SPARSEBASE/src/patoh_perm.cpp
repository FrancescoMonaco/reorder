/**
 * PaToH (Hypergraph Partitioning) Permutation Generator
 * 
 * Interface: ./patoh_perm matrix.mtx output.perm [--nparts N] [--objective OBJ]
 * Generates a PaToH hypergraph partitioning-based reordering for a Matrix Market sparse matrix.
 * Uses hypergraph partitioning to group related rows/columns together.
 * Output is a 1-based permutation file compatible with the reordering pipeline.
 * 
 * Algorithm: Uses PaToH hypergraph partitioner to partition the matrix,
 *            then orders vertices by partition ID to improve locality.
 *            CONNECTIVITY objective minimizes the number of partitions each
 *            column (hyperedge) touches, directly optimizing for x-vector
 *            reuse in SpMV/SpMM.
 *            Default: 128 partitions with CONNECTIVITY objective.
 */

#include "reorder_utils.h"
#include "sparsebase/partition/patoh_partition.h"
#include <ccutils/timers.h>
#include <ccutils/macros.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace sparsebase;
using namespace reorder_utils;

int main(int argc, char* argv[]) {
    std::cerr << "[PaToH] Started with " << argc << " arguments\n";
    for (int i = 0; i < argc; i++) {
        std::cerr << "[PaToH]   argv[" << i << "] = " << argv[i] << "\n";
    }
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <output.perm> [--nparts N] [--objective OBJ]\n";
        std::cerr << "Generates PaToH hypergraph partitioning-based permutation.\n";
        std::cerr << "Optional flags:\n";
        std::cerr << "  --nparts N       number of partitions (default: 128)\n";
        std::cerr << "  --objective OBJ  cut or connectivity (default: connectivity)\n";
        return 1;
    }

    CPU_TIMER_DEF(loading);
    CPU_TIMER_DEF(reordering);
    
    std::string mtx_file = argv[1];
    std::string perm_file = argv[2];
    
    // Parse number of partitions
    int num_parts = 128;  // default: enough partitions for fine-grained row clustering
    std::string nparts_str = get_flag_value(argc, argv, "--nparts");
    if (!nparts_str.empty()) {
        num_parts = std::stoi(nparts_str);
    }
    
    // Parse objective type
    // CONNECTIVITY: minimize number of partitions each hyperedge (column) touches
    //   → directly optimizes x-vector reuse for SpMV/SpMM
    // CUT: minimize total cut size (number of cut hyperedges)
    bool use_connectivity = true;  // default
    std::string obj_str = get_flag_value(argc, argv, "--objective");
    if (obj_str == "cut" || obj_str == "CUT") {
        use_connectivity = false;
    } else if (!obj_str.empty() && obj_str != "connectivity" && obj_str != "CONNECTIVITY") {
        std::cerr << "[PaToH] Warning: unknown objective '" << obj_str << "', using 'connectivity'\n";
    }
    
    std::cerr << "[PaToH] Loading matrix: " << mtx_file << "\n";
    try {
        CPU_TIMER_START(loading);
        auto csr = load_matrix(mtx_file);
        CPU_TIMER_STOP(loading);

        int n = csr->get_dimensions()[0];
        std::cerr << "[PaToH] Matrix loaded: " << n << " vertices\n";
        
        context::CPUContext cpu_context;
        std::cerr << "[PaToH] Computing hypergraph partitioning with " << num_parts << " partitions"
                  << " (objective: " << (use_connectivity ? "connectivity" : "cut") << ")...\n";
        
        // PaToH partition parameters
        partition::PatohPartitionParams params;
        params.num_partitions = num_parts;
        params.objective = use_connectivity ? partition::patoh::CON : partition::patoh::CUT;
        params.param_init = partition::patoh::QUALITY;  // Quality partitioning
        
        // Create PaToH partitioner
        partition::PatohPartition<int, int, float> partitioner;
        
        CPU_TIMER_START(reordering);
        int* partition = partitioner.Partition(
            csr, &params, {&cpu_context}, true);
                
        std::vector<std::pair<int, int>> part_vertex;
        part_vertex.reserve(n);
        for (int i = 0; i < n; i++) {
            part_vertex.push_back({partition[i], i});
        }
        
        // Sort by partition ID (and by vertex ID within partition for stability)
        std::sort(part_vertex.begin(), part_vertex.end());
        
        // Create permutation: perm[i] = old position of vertex at new position i
        int* perm = new int[n];
        for (int i = 0; i < n; i++) {
            perm[i] = part_vertex[i].second;
        }
        CPU_TIMER_STOP(reordering);
        
        std::cerr << "[PaToH] Saving permutation to: " << perm_file << "\n";
        bool success = save_permutation(perm_file, perm, n);
        std::cerr << "[PaToH] " << (success ? "Success" : "Failed") << "\n";
        
        TIMER_PRINT(loading);
        TIMER_PRINT(reordering);

        delete[] perm;
        
        return success ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
