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

#include "club/halo.hpp"
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
// halo_from_mtx(path, b, K, W, lambda, hub_frac) -> dict
//
// Load a Matrix Market file, run the full rectangular HALO pipeline
// (adaptive hub peel -> OPMH sketch -> parallel radix -> guarded SPA pack,
// columns first then rows), and return {"row_perm": [...], "col_perm": [...]}
// mapping new position -> original index.
// ---------------------------------------------------------------------------
static nb::dict halo_from_mtx( const std::string& mtx_path,
                               size_t b,
                               size_t K,
                               size_t W,
                               double lambda,
                               double hub_frac ) {
    std::ifstream infile( mtx_path );
    if ( !infile.is_open() ) {
        throw std::runtime_error( "Failed to open file: " + mtx_path );
    }

    club::CSR<float, int> A;
    A.read_from_mtx( infile );
    infile.close();

    if ( A.rows == 0 ) {
        throw std::runtime_error( "Matrix has zero rows or failed to parse: " + mtx_path );
    }

    club::halo::Params p;
    p.b = b;
    p.K = K;
    p.W = W;
    p.lambda = lambda;
    p.hub_frac = hub_frac;

    std::vector<size_t> rp, cp;
    club::halo::Stats st;
    if ( A.rows == A.cols ) {
        // Symmetric PAP^T: col perm == row perm. Mirror it into cp so the
        // .perm file carries both lines and ASYMMETRIC evaluation applies
        // both sides (identical to SYMMETRIC for this case).
        club::halo::reorder_symmetric( A, p, &rp, &st );
        cp = rp;
    } else {
        club::halo::reorder_rectangular( A, p, &rp, &cp, &st );
    }

    nb::dict out;
    out["row_perm"] = rp;
    out["col_perm"] = cp;
    out["tau_row"] = st.tau_row;
    out["tau_col"] = st.tau_col;
    out["n_core_rows"] = st.n_core_rows;
    out["n_core_cols"] = st.n_core_cols;
    out["ntiles"] = st.ntiles;
    return out;
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
    m.def( "halo",
           &halo_from_mtx,
           "mtx_path"_a,
           "b"_a = 8,
           "K"_a = 16,
           "W"_a = 16,
           "lambda"_a = 0.5,
           "hub_frac"_a = 0.005,
           "Load a Matrix Market file, run the full rectangular HALO pipeline "
           "(adaptive hub peel + OPMH sketch + parallel radix + guarded SPA "
           "pack, columns then rows), and return row/col perms + stats." );
}
