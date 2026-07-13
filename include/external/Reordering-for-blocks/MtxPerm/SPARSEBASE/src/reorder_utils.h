/**
 * Common utilities for reordering tools
 * Handles I/O, permutation conversion, and error handling
 */

#ifndef REORDER_UTILS_H
#define REORDER_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include "sparsebase/bases/iobase.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/cpu_context.h"

namespace reorder_utils {

using namespace sparsebase;

/**
 * Loads a Matrix Market file and returns CSR representation
 */
inline format::CSR<int, int, float>* load_matrix(const std::string& mtx_file) {
    return bases::IOBase::ReadMTXToCSR<int, int, float>(mtx_file, true);
}

/**
 * Saves a 0-based permutation array to file in 1-based format
 * Format: single line with space-separated entries
 */
inline bool save_permutation(const std::string& perm_file, int* perm, int n) {
    std::ofstream out(perm_file);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open output file " << perm_file << "\n";
        return false;
    }
    
    for (int i = 0; i < n; i++) {
        if (i > 0) out << " ";
        out << (perm[i] + 1);
    }
    out << "\n";
    out.close();
    return true;
}

/**
 * Simple command-line argument parser
 * Supports: --flag=value or --flag value
 * Returns empty string if flag not found
 */
inline std::string get_flag_value(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Check for --flag=value format
        if (arg.find(flag + "=") == 0) {
            return arg.substr(flag.length() + 1);
        }
        
        // Check for --flag value format
        if (arg == flag && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return "";
}

/**
 * Check if a flag is present (for boolean flags)
 */
inline bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

/**
 * Check if matrix is square. If not, print error and return false.
 * Use for reorderings that require square matrices (AMD, RCM, Rabbit, METIS).
 */
inline bool check_square_matrix(format::CSR<int, int, float>* csr, const std::string& algo_name) {
    int n_rows = csr->get_dimensions()[0];
    int n_cols = csr->get_dimensions()[1];
    if (n_rows != n_cols) {
        std::cerr << "Error: " << algo_name << " requires a square matrix. "
                  << "Got " << n_rows << " x " << n_cols << "\n";
        return false;
    }
    return true;
}

} // namespace reorder_utils

#endif // REORDER_UTILS_H
