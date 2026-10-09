#include <chrono>
#include <fstream>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "club/fingerprint.hpp"
#include "club/refine.hpp"
#include "club/reorder.hpp"
#include "club/seed_refine.hpp"

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
// micromacro_mask_from_mtx(path, W, micro_threshold) -> list[size_t]
//
// Load a Matrix Market file into club::CSR<float, int>, build the windowed
// sketch Ahat with club::mask(), then cluster it with the micro-macro
// strategy (club::cluster_lex_micromacro) and return the row permutation
// (0-based) that maps new position -> original row index.
// ---------------------------------------------------------------------------
static std::vector<size_t>
micromacro_mask_from_mtx( const std::string& mtx_path, size_t W, size_t micro_threshold ) {
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

    // Build the windowed sketch, then cluster it with the micro-macro
    // (micro radix buckets + macro bipartite BFS) strategy.
    club::CSR<size_t, size_t> Ahat;
    club::mask( A, W, Ahat );

    // micro_threshold == 0 means "auto" (sqrt(n)), per the header's semantics.
    std::vector<size_t> perm;
    club::cluster_lex_micromacro( Ahat, perm, micro_threshold );

    return perm;
}

// ---------------------------------------------------------------------------
// seed_refine_from_mtx(path, W, L, s, T, B) -> (row_perm, col_perm, blocks)
//
// Seed-and-refine reordering: Mask/ClusterLex seed + bounded exact local
// refinement (ExactJaccard + greedy chain, commit-if-better), alternating
// row/column passes up to T iterations with best-state rollback.
// Returns 0-based perms (new position -> original index) plus best BxB count.
// ---------------------------------------------------------------------------
static std::tuple<std::vector<size_t>, std::vector<size_t>, size_t> seed_refine_from_mtx(
    const std::string& mtx_path, size_t W, size_t L, size_t s, size_t T, size_t B ) {
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

    std::vector<size_t> pi_r, pi_c;
    club::seed::SeedStats st;
    club::seed::seed_refine( A, W, L, s, T, B, pi_r, pi_c, &st );

    return std::make_tuple( std::move( pi_r ), std::move( pi_c ), st.best_blocks );
}

// ---------------------------------------------------------------------------
// fingerprint_from_mtx(path, W, k, tau, Kmax, K0, bs, refine_passes, window)
// Row-only Bloom-archetype reordering: mask(W in {4,8}) -> Bloom fingerprints
// -> leader clustering (overlap>=tau joins, else new archetype, cap Kmax) ->
// mass-order + in-cluster exact-overlap chain -> stripe_refine polish at bs.
// NO identity guard: the algorithm output is returned as-is (row-only).
// Returns 0-based perm (new position -> original row index).
// ---------------------------------------------------------------------------
static std::vector<size_t> fingerprint_from_mtx( const std::string& mtx_path,
                                                 size_t W,
                                                 size_t k_words,
                                                 size_t tau_bits,
                                                 size_t Kmax,
                                                 size_t K0,
                                                 size_t bs,
                                                 size_t refine_passes,
                                                 size_t window ) {
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

    club::fp::FingerprintParams P;
    P.W = W;
    P.k_words = k_words;
    P.tau_bits = tau_bits;
    P.Kmax = Kmax;
    P.K0 = K0;

    std::vector<size_t> perm;
    club::fp::FingerprintStats st;
    club::fp::fingerprint_reorder( A, P, perm, &st );

    if ( refine_passes > 0 ) {
        club::CSR<size_t, size_t> Ahat;
        club::mask( A, bs, Ahat );
        club::refine::stripe_refine( Ahat, perm, bs, refine_passes, window );
    }

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
           "A"_a,
           "Ws"_a = std::vector<size_t>{ 16, 8, 4, 1 },
           "Load a Matrix Market file, run the multi-level CLUB masking " );
    m.def( "micro_macro_mask",
           &micromacro_mask_from_mtx,
           "mtx_path"_a,
           "W"_a = 4,
           "micro_threshold"_a = 0,
           "Load a Matrix Market file, build the CLUB windowed sketch of size W, "
           "and run the micro-macro CLUB clustering (micro_threshold == 0 means "
           "auto: sqrt(n)), returning the row permutation." );
    m.def( "seed_refine",
           &seed_refine_from_mtx,
           "mtx_path"_a,
           "W"_a = 4,
           "L"_a = 256,
           "s"_a = 256,
           "T"_a = 4,
           "B"_a = 32,
           "Seed-and-refine reordering (Mask/ClusterLex seed + bounded exact "
           "LocalRefine, alternating row/column passes). Returns "
           "(row_perm, col_perm, best_blocks), 0-based new->old indices." );
    m.def( "fingerprint",
           &fingerprint_from_mtx,
           "mtx_path"_a,
           "W"_a = 8,
           "k_words"_a = 4,
           "tau_bits"_a = 8,
           "Kmax"_a = 256,
           "K0"_a = 16,
           "bs"_a = 16,
           "refine_passes"_a = 1,
           "window"_a = 3,
           "Row-only Bloom-archetype reordering (mask W -> Bloom fingerprints "
           "-> leader clustering on overlap -> stripe_refine polish). "
           "No identity guard: output returned as-is." );
}
