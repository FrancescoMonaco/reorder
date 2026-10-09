#pragma once
// fingerprint.hpp — Bloom-style sparsity archetypes + leader clustering.
//
// New CLUB assignment path (row-only):
//   S0. Ahat = mask(A, W) with W in {4,8}          (block-native sketch)
//   S1. Fingerprint per row: k Bloom words          (splitmix64 hashed windows,
//       OR-accumulated into k uint64 words; fixed bytes per row)
//   S2. Dictionary of archetype fingerprints, leader-style:
//         rows visit in mass-desc order; score = overlap popcount against
//         each archetype; best >= tau -> assign; else new archetype
//         (capped at Kmax; spillover assigns anyway).
//   S3. Order: archetypes by mass desc; rows inside a cluster chained
//       greedily on EXACT sketch overlap (bounded window, like jaccard's
//       in-cluster chain but standalone).
//   S4. Polish: caller runs refine::stripe_refine (kept outside).
//       NO identity guard: the algorithm output is returned as-is.
//
// Complexity (n rows, mhat sketch nnz, K archetypes, d avg degree):
//   fingerprints: O(mhat) parallel
//   assignment:   O(n * K * k) popcounts, OpenMP over rows within a
//                 sequential-dictionary pass per batch (see below).
//   Parallelism: dictionary grows sequentially, so assignment runs in
//   small batches: score a batch against the frozen dict in parallel,
//   then commit (assign-or-new-archetype) serially in row order.
//   Deterministic given the same visit order and batch size.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "club/matrices.hpp"
#include "club/reorder.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace club {
namespace fp {

struct FingerprintParams {
    size_t W = 8;            // sketch width (mask windows of W cols)
    size_t k_words = 4;      // Bloom words per fingerprint (256 bits)
    size_t tau_bits = 8;     // min shared Bloom bits to join an archetype
    size_t Kmax = 256;       // dictionary cap; beyond: assign best anyway
    size_t K0 = 16;          // seed archetypes: first K0 non-empty rows by mass
    size_t batch = 4096;     // parallel scoring batch size
};

struct FingerprintStats {
    size_t n = 0;
    size_t nwin = 0;
    size_t mhat = 0;
    size_t n_archetypes = 0;
    size_t n_singletons = 0;
    size_t n_empty = 0;
    double fp_ms = 0.0;
    double assign_ms = 0.0;
    double order_ms = 0.0;
};

// splitmix64: cheap universal-ish hash for window -> bit position.
inline uint64_t splitmix64( uint64_t x ) {
    x += 0x9E3779B97F4A7C15ULL;
    x = ( x ^ ( x >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
    x = ( x ^ ( x >> 27 ) ) * 0x94D049BB133111EBULL;
    return x ^ ( x >> 31 );
}

// Build k-word Bloom fingerprints.
// Lane j samples the row's window set with an independent hash and sets
// ONE bit per window: bit splitmix64(w * C + j) % 64 in word j.
// Overlap popcount(a & b) then behaves like a MinHash collision count:
// similar sets share bits, dissimilar sets share ~d/64 by chance.
// (Previous version OR-ed the same window into every word, which inflated
// all scores and washed out tau on dense rows.)
inline void build_fingerprints( const CSR<size_t, size_t>& Ahat,
                                size_t k_words,
                                std::vector<uint64_t>& fps,
                                FingerprintStats* st = nullptr ) {
    const size_t n = Ahat.rows;
    fps.assign( n * ( k_words ? k_words : 1 ), 0 );
    const size_t k = k_words ? k_words : 1;
#pragma omp parallel for schedule( static )
    for ( size_t r = 0; r < n; ++r ) {
        uint64_t* f = &fps[r * k];
        for ( size_t t = Ahat.row_ptr[r]; t < Ahat.row_ptr[r + 1]; ++t ) {
            const uint64_t w = static_cast<uint64_t>( Ahat.col_ind[t] );
            for ( size_t lane = 0; lane < k; ++lane ) {
                const uint64_t h = splitmix64( w * 0x9E3779B97F4A7C15ULL +
                                               0xBF58476D1CE4E5B9ULL * ( lane + 1 ) );
                f[lane] |= 1ULL << ( h & 63 );
            }
        }
    }
    if ( st ) {
        st->n = n;
        st->nwin = Ahat.cols;
        st->mhat = Ahat.rows ? Ahat.row_ptr[n] : 0;
    }
}

inline size_t overlap_bits( const uint64_t* a, const uint64_t* b, size_t k ) {
    size_t s = 0;
    for ( size_t j = 0; j < k; ++j )
        s += static_cast<size_t>( __builtin_popcountll( a[j] & b[j] ) );
    return s;
}

// Exact sketch overlap |Ahat[r] cap Ahat[q]| on sorted window lists.
inline size_t exact_overlap( const CSR<size_t, size_t>& Ahat, size_t r, size_t q ) {
    size_t i = Ahat.row_ptr[r], ie = Ahat.row_ptr[r + 1];
    size_t j = Ahat.row_ptr[q], je = Ahat.row_ptr[q + 1];
    size_t s = 0;
    const auto* ci = Ahat.col_ind.data();
    while ( i < ie && j < je ) {
        if ( ci[i] == ci[j] ) {
            ++s;
            ++i;
            ++j;
        } else if ( ci[i] < ci[j] ) {
            ++i;
        } else {
            ++j;
        }
    }
    return s;
}

// Cluster rows to archetypes (leader-style). Visit order: mass-descending
// (sketch degree), ties by row id — deterministic. Batching: score a batch
// vs the frozen dict in parallel, commit serially so dictionary growth is
// deterministic. Empty rows -> cluster SIZE_MAX (caller emits them first).
inline void cluster_rows( const CSR<size_t, size_t>& Ahat,
                          const std::vector<uint64_t>& fps,
                          const FingerprintParams& P,
                          std::vector<std::vector<size_t>>& members,
                          std::vector<size_t>& archetype_of,
                          FingerprintStats* st = nullptr ) {
    const size_t n = Ahat.rows;
    const size_t k = P.k_words ? P.k_words : 1;
    archetype_of.assign( n, SIZE_MAX );
    members.clear();
    std::vector<size_t> order( n );
    std::iota( order.begin(), order.end(), 0 );
    std::vector<size_t> deg( n );
    for ( size_t r = 0; r < n; ++r )
        deg[r] = Ahat.row_ptr[r + 1] - Ahat.row_ptr[r];
    std::sort( order.begin(), order.end(), [&]( size_t a, size_t b ) {
        if ( deg[a] != deg[b] )
            return deg[a] > deg[b];
        return a < b;
    } );
    std::vector<std::vector<uint64_t>> dict;
    dict.reserve( P.Kmax );
    members.reserve( P.Kmax );
    size_t n_empty = 0;
    for ( size_t r : order ) { // K0 seeds: first K0 non-empty rows
        if ( deg[r] == 0 ) {
            ++n_empty;
            continue;
        }
        if ( dict.size() >= P.K0 || dict.size() >= P.Kmax )
            break;
        dict.emplace_back( &fps[r * k], &fps[r * k + k] );
        members.emplace_back();
        members.back().push_back( r );
        archetype_of[r] = dict.size() - 1;
    }
    auto commit_one = [&]( size_t r2, size_t ba, size_t bs ) {
        if ( bs >= P.tau_bits && ba != SIZE_MAX ) {
            archetype_of[r2] = ba;
            members[ba].push_back( r2 );
        } else if ( dict.size() < P.Kmax ) {
            dict.emplace_back( &fps[r2 * k], &fps[r2 * k + k] );
            members.emplace_back();
            members.back().push_back( r2 );
            archetype_of[r2] = dict.size() - 1;
        } else if ( !members.empty() ) { // spillover: assign best anyway
            const size_t a = ba != SIZE_MAX ? ba : 0;
            archetype_of[r2] = a;
            members[a].push_back( r2 );
        }
    };
    const size_t B = P.batch ? P.batch : 4096;
    std::vector<size_t> pending;
    pending.reserve( B );
    std::vector<size_t> best_a, best_s;
    auto flush = [&]() {
        best_a.assign( pending.size(), SIZE_MAX );
        best_s.assign( pending.size(), 0 );
#pragma omp parallel for schedule( static )
        for ( size_t b = 0; b < pending.size(); ++b ) {
            const uint64_t* f = &fps[pending[b] * k];
            size_t bs = 0, ba = SIZE_MAX;
            for ( size_t a = 0; a < dict.size(); ++a ) {
                const size_t s = overlap_bits( f, dict[a].data(), k );
                if ( s > bs ) {
                    bs = s;
                    ba = a;
                }
            }
            best_a[b] = ba;
            best_s[b] = bs;
        }
        for ( size_t b = 0; b < pending.size(); ++b )
            commit_one( pending[b], best_a[b], best_s[b] );
        pending.clear();
    };
    for ( size_t r : order ) {
        if ( archetype_of[r] != SIZE_MAX || deg[r] == 0 )
            continue;
        pending.push_back( r );
        if ( pending.size() >= B )
            flush();
    }
    if ( !pending.empty() )
        flush();
    if ( st ) {
        st->n_archetypes = dict.size();
        st->n_empty = n_empty;
        size_t ns = 0;
        for ( auto& m : members )
            if ( m.size() == 1 )
                ++ns;
        st->n_singletons = ns;
    }
}

// Order clusters by mass desc; chain rows inside each cluster greedily on
// EXACT sketch overlap (bounded NN walk from the first member).
// Empty rows are emitted first (stable). No identity guard by design.
inline void order_from_archetypes( const CSR<size_t, size_t>& Ahat,
                                   const std::vector<std::vector<size_t>>& members,
                                   std::vector<size_t>& Pout ) {
    Pout.clear();
    const size_t n = Ahat.rows;
    Pout.reserve( n );
    for ( size_t r = 0; r < n; ++r )
        if ( Ahat.row_ptr[r + 1] == Ahat.row_ptr[r] )
            Pout.push_back( r );
    std::vector<size_t> cidx( members.size() );
    std::iota( cidx.begin(), cidx.end(), 0 );
    std::vector<size_t> mass( members.size(), 0 ), mn( members.size(), SIZE_MAX );
    for ( size_t c = 0; c < members.size(); ++c )
        for ( size_t r : members[c] ) {
            mass[c] += Ahat.row_ptr[r + 1] - Ahat.row_ptr[r];
            mn[c] = std::min( mn[c], r );
        }
    std::sort( cidx.begin(), cidx.end(), [&]( size_t a, size_t b ) {
        if ( mass[a] != mass[b] )
            return mass[a] > mass[b];
        return mn[a] < mn[b];
    } );
    for ( size_t c : cidx ) {
        const auto& mem = members[c];
        if ( mem.empty() )
            continue;
        size_t cur = mem[0];
        std::vector<char> done( n, 0 );
        for ( size_t r : mem )
            done[r] = 1;
        Pout.push_back( cur );
        done[cur] = 2;
        size_t emitted = 1;
        while ( emitted < mem.size() ) {
            size_t best = SIZE_MAX, best_s = 0;
            bool first = true;
            for ( size_t r : mem ) {
                if ( done[r] == 2 )
                    continue;
                const size_t s = exact_overlap( Ahat, cur, r );
                if ( first || s > best_s || ( s == best_s && r < best ) ) {
                    best_s = s;
                    best = r;
                    first = false;
                }
            }
            if ( best == SIZE_MAX )
                break;
            Pout.push_back( best );
            done[best] = 2;
            cur = best;
            ++emitted;
        }
    }
}

// Full driver: mask -> fingerprints -> cluster -> order.
// Returns P (position -> row). NO identity guard: output is final as-is.
// Caller runs stripe_refine afterwards if wanted.
template <typename DataT, typename intT>
inline void fingerprint_reorder( CSR<DataT, intT>& A,
                                const FingerprintParams& P,
                                std::vector<size_t>& Pout,
                                FingerprintStats* st = nullptr ) {
    CSR<size_t, size_t> Ahat;
    mask( A, P.W, Ahat );
    std::vector<uint64_t> fps;
    build_fingerprints( Ahat, P.k_words, fps, st );
    std::vector<std::vector<size_t>> members;
    std::vector<size_t> comp;
    cluster_rows( Ahat, fps, P, members, comp, st );
    order_from_archetypes( Ahat, members, Pout );
}

} // namespace fp
} // namespace club
