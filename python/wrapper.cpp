#include <chrono>
#include <fstream>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "club/reorder.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// reorder2_from_mtx(path, W, max_iters) -> list[size_t]
//
// Load a Matrix Market file into club::CSR<float, int>, run the 2-sided
// club::reorder2() with the given parameters, and return the cumulative row
// permutation (0-based) that maps new position -> original row index.
// ---------------------------------------------------------------------------
static std::vector<size_t>
reorder2_from_mtx( const std::string& mtx_path, size_t W, size_t max_iters ) {
    // Open .mtx file
    std::ifstream infile( mtx_path );
    if ( !infile.is_open() ) {
        throw std::runtime_error( "Failed to open file: " + mtx_path );
    }

    // Read matrix into CSR
    club::CSR<float, int> A;
    A.read_from_mtx( infile );
    infile.close();

    if ( A.rows == 0 ) {
        throw std::runtime_error( "Matrix has zero rows or failed to parse: " + mtx_path );
    }

    // Run reorder2 and capture the cumulative row permutation
    std::vector<size_t> perm;
    club::reorder2( A, W, max_iters, &perm );

    return perm;
}

static std::vector<size_t> mask_multilevel_from_mtx( const std::string& mtx_path,
                                                     const std::vector<size_t>& Ws ) {
    // Open .mtx file
    std::ifstream infile( mtx_path );
    if ( !infile.is_open() ) {
        throw std::runtime_error( "Failed to open file: " + mtx_path );
    }

    // Read matrix into CSR
    club::CSR<float, int> A;
    std::vector<size_t> perm;
    A.read_from_mtx( infile );
    infile.close();

    if ( A.rows == 0 ) {
        throw std::runtime_error( "Matrix has zero rows or failed to parse: " + mtx_path );
    }

    // Run mask_multilevel and return the result in Ahat
    club::mask_multilevel( A, Ws, perm );

    return perm;
}

// ---------------------------------------------------------------------------
// hybrid_from_mtx(path, W, W_block, lookahead_C, hub_frac) -> list[size_t]
//
// Load a Matrix Market file into club::CSR<float, int>, run the 3-stage hybrid
// reordering (club::reorder_hybrid: IDF-filtered MinHash micro-clustering +
// weighted bipartite macro-RCM + SIMD Jaccard slab packing), and return the
// row permutation (0-based) that maps new position -> original row index.
// ---------------------------------------------------------------------------
static std::vector<size_t> hybrid_from_mtx( const std::string& mtx_path,
                                            size_t W,
                                            size_t W_block,
                                            size_t lookahead_C,
                                            double hub_frac ) {
    // Open .mtx file
    std::ifstream infile( mtx_path );
    if ( !infile.is_open() ) {
        throw std::runtime_error( "Failed to open file: " + mtx_path );
    }

    // Read matrix into CSR
    club::CSR<float, int> A;
    A.read_from_mtx( infile );
    infile.close();

    if ( A.rows == 0 ) {
        throw std::runtime_error( "Matrix has zero rows or failed to parse: " + mtx_path );
    }

    // Run the hybrid reordering and capture the row permutation.
    std::vector<size_t> perm;
    club::reorder_hybrid( A, W, W_block, lookahead_C, hub_frac, &perm );

    return perm;
}

// ---------------------------------------------------------------------------
// Python module definition
// ---------------------------------------------------------------------------
NB_MODULE( _reorder_impl, m ) {

    m.doc() = "CLUB reordering bindings for the Reordering-for-blocks pipeline";

    m.def( "reorder2",
           &reorder2_from_mtx,
           "mtx_path"_a,
           "W"_a = 4,
           "max_iters"_a = 4,
           "Load a Matrix Market file, run the 2-sided CLUB reordering "
           "(reorder2) with the given parameters, and return the cumulative "
           "row permutation as a list of 0-based indices." );
    m.def( "multilevel_mask",
           &mask_multilevel_from_mtx,
           "mtx_path"_a,
           "Ws"_a = std::vector<size_t>{ 16, 8, 4, 1 },
           "Load a Matrix Market file, run the multi-level CLUB masking "
           "(mask_multilevel) over the given window sizes and return the row "
           "permutation as a list of 0-based indices." );
    m.def( "hybrid",
           &hybrid_from_mtx,
           "mtx_path"_a,
           "W"_a = 32,
           "W_block"_a = 32,
           "lookahead_C"_a = 64,
           "hub_frac"_a = 0.05,
           "Load a Matrix Market file, run the 3-stage hybrid CLUB reordering "
           "(reorder_hybrid: MinHash micro-clustering + bipartite macro-RCM + "
           "SIMD Jaccard slab packing), and return the row permutation as a "
           "list of 0-based indices." );
}
