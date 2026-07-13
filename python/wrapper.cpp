#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <chrono>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
static std::vector<size_t> reorder2_from_mtx(
    const std::string& mtx_path,
    size_t W,
    size_t max_iters
) {
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

// ---------------------------------------------------------------------------
// Python module definition
// ---------------------------------------------------------------------------
NB_MODULE( _reorder_impl, m ) {

    m.doc() = "CLUB reordering bindings for the Reordering-for-blocks pipeline";

    m.def(
        "reorder2",
        &reorder2_from_mtx,
        "mtx_path"_a,
        "W"_a = 32,
        "max_iters"_a = 4,
        "Load a Matrix Market file, run the 2-sided CLUB reordering "
        "(reorder2) with the given parameters, and return the cumulative "
        "row permutation as a list of 0-based indices."
    );
}
