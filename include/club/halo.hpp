#pragma once
// halo.hpp — HALO: Hub-Aware LSH-Predicted Ordering.
//
// Zero-Dynamic-Graph-Mutation / Zero-Atomic-Contention / Zero-2Hop-Scan /
// Zero-Global-TileCSR-Materialization:
//   Phase 1: work-balanced hub peeling, privatized coalescing, bitset masking
//   Phase 2: register-resident One-Permutation MinHash (OPMH) + densification
//            with Lexicographic MinHash Prefix Keys (streaming row & col passes)
//   Phase 3: lock-free parallel 8-bit LSD radix sort (L1-resident histograms)
//   Phase 4: L1 window-cached SPA guarded packing + W-lookahead window swap

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "club/expect.hpp"
#include "club/logging.hpp"
#include "club/matrices.hpp"
#include "club/metrics.hpp"
#include "omp.h"

namespace club::halo {

    struct Params {
        size_t b = 8;                // hardware tile dim, power of two (4 or 8)
        size_t K = 16;               // OPMH bins, power of two (8 or 16)
        double hub_frac = 0.005;     // top-fraction fallback quantile (0.5%)
        double lambda = 0.5;         // guard slack in [0,1]
        size_t W = 16;               // L1 lookahead window (16..32)
        uint64_t seed = 0x243f6a8885a308d3ULL;
        bool precompress_tiles = true;
    };

    struct Stats {
        size_t n_hub_rows = 0;
        size_t n_hub_cols = 0;
        size_t n_core_rows = 0;
        size_t n_core_cols = 0;
        size_t tau_row = 0;
        size_t tau_col = 0;
        size_t ntiles = 0;
    };

    struct Bitset {
        std::vector<uint64_t> w;
        Bitset() = default;
        explicit Bitset( size_t n ) : w( ( n + 63 ) / 64, 0 ) {}
        void resize( size_t n ) { w.assign( ( n + 63 ) / 64, 0 ); }
        inline void set_atomic( size_t i ) {
            __atomic_fetch_or( &w[i >> 6], 1ULL << ( i & 63 ), __ATOMIC_RELAXED );
        }
        inline void set( size_t i ) { w[i >> 6] |= 1ULL << ( i & 63 ); }
        inline bool test( size_t i ) const { return ( w[i >> 6] >> ( i & 63 ) ) & 1ULL; }
        size_t count() const {
            size_t c = 0;
            for ( uint64_t x : w )
                c += static_cast<size_t>( __builtin_popcountll( x ) );
            return c;
        }
    };

    namespace detail {

        inline uint64_t splitmix64( uint64_t x ) noexcept {
            x += 0x9e3779b97f4a7c15ULL;
            x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
            x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebULL;
            return x ^ ( x >> 31 );
        }

        inline uint32_t mix32( uint32_t x ) noexcept {
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            x *= 0x846ca68bU;
            x ^= x >> 16;
            return x;
        }

        struct RowSketchItem {
            uint64_t sort_key = 0;
            uint32_t row_id = 0;
            uint32_t eff_deg = 0;
        };

        inline unsigned ilog2nz( size_t x ) noexcept {
            return 63U - static_cast<unsigned>( __builtin_clzll( x ) );
        }

        // Pre-checked relaxed atomic min: only issues CAS when val < *ptr.
        inline void atomic_min_u32( uint32_t* ptr, uint32_t val ) noexcept {
            uint32_t cur = __atomic_load_n( ptr, __ATOMIC_RELAXED );
            while ( val < cur ) {
                if ( __atomic_compare_exchange_n(
                         ptr, &cur, val, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED ) )
                    break;
            }
        }

        // Partition CSR rows into P equal-nnz chunks via binary search.
        template <typename intT>
        inline void equal_nnz_range( const std::vector<intT>& row_ptr,
                                     size_t n,
                                     size_t nnz,
                                     int tid,
                                     int P,
                                     size_t& r_start,
                                     size_t& r_end ) {
            size_t lo_t = ( nnz * static_cast<size_t>( tid ) ) / static_cast<size_t>( P );
            size_t hi_t =
                ( nnz * static_cast<size_t>( tid + 1 ) ) / static_cast<size_t>( P );
            size_t lo = 0, hi = n;
            while ( lo < hi ) {
                size_t mid = lo + ( hi - lo ) / 2;
                if ( static_cast<size_t>( row_ptr[mid] ) < lo_t )
                    lo = mid + 1;
                else
                    hi = mid;
            }
            r_start = lo;
            lo = 0;
            hi = n;
            while ( lo < hi ) {
                size_t mid = lo + ( hi - lo ) / 2;
                if ( static_cast<size_t>( row_ptr[mid] ) < hi_t )
                    lo = mid + 1;
                else
                    hi = mid;
            }
            r_end = lo;
            if ( tid == 0 )
                r_start = 0;
            if ( tid == P - 1 )
                r_end = n;
            if ( r_end < r_start )
                r_end = r_start;
        }

        // Adaptive hub threshold: max(min(mean + 4*std, quantile), min_c).
        inline size_t adaptive_tau( double mean,
                                    double stdv,
                                    size_t q,
                                    size_t min_c ) {
            double stat = mean + 4.0 * stdv;
            size_t t_stat =
                ( stat <= 0.0 ) ? min_c : static_cast<size_t>( std::ceil( stat ) );
            return std::max( std::min( t_stat, q ), min_c );
        }

        // (1-hub_frac) quantile computed relative to samp.size().
        inline size_t quantile_tau( const std::vector<size_t>& deg, double hub_frac ) {
            const size_t n = deg.size();
            if ( n == 0 )
                return std::numeric_limits<size_t>::max();
            const double f = std::min( 0.5, std::max( 0.0, hub_frac ) );
            std::vector<size_t> samp;
            if ( n <= 131072 ) {
                samp = deg;
            } else {
                samp.reserve( 65536 );
                const size_t step = n / 65536;
                for ( size_t i = 0; i < n; i += step )
                    samp.push_back( deg[i] );
            }
            const size_t m = samp.size();
            const size_t k = std::clamp<size_t>(
                static_cast<size_t>( std::ceil( f * static_cast<double>( m ) ) ), 1, m );
            const size_t pos = m - k;
            std::nth_element( samp.begin(), samp.begin() + pos, samp.end() );
            return samp[pos];
        }

        // 64-entry L1 coalescing cache for streaming column degree accumulation.
        struct Coalesce64 {
            static constexpr size_t CAP = 64;
            uint32_t keys[CAP];
            uint32_t cnts[CAP];
            uint8_t used[CAP];
            size_t n = 0;
            Coalesce64() {
                for ( size_t i = 0; i < CAP; ++i )
                    used[i] = 0;
            }
            inline void flush_into( std::vector<uint32_t>& g ) {
                for ( size_t i = 0; i < CAP; ++i ) {
                    if ( used[i] ) {
                        __atomic_fetch_add( &g[keys[i]], cnts[i], __ATOMIC_RELAXED );
                        used[i] = 0;
                    }
                }
                n = 0;
            }
            inline void bump( uint32_t c, std::vector<uint32_t>& g ) {
                size_t slot = static_cast<size_t>( mix32( c * 0x9e3779b1U ) ) & 63U;
                for ( size_t t = 0; t < 4; ++t ) {
                    size_t i = ( slot + t ) & 63U;
                    if ( used[i] && keys[i] == c ) {
                        ++cnts[i];
                        return;
                    }
                    if ( !used[i] ) {
                        used[i] = 1;
                        keys[i] = c;
                        cnts[i] = 1;
                        if ( ++n >= CAP )
                            flush_into( g );
                        return;
                    }
                }
                __atomic_fetch_add( &g[keys[slot]], cnts[slot], __ATOMIC_RELAXED );
                keys[slot] = c;
                cnts[slot] = 1;
            }
        };

        template <typename DataT, typename intT>
        inline void row_degrees( const CSR<DataT, intT>& A, std::vector<size_t>& out ) {
            const size_t N = static_cast<size_t>( A.rows );
            out.resize( N );
#pragma omp parallel for schedule( static )
            for ( size_t r = 0; r < N; ++r )
                out[r] = static_cast<size_t>( A.row_ptr[r + 1] - A.row_ptr[r] );
        }

        template <typename DataT, typename intT>
        inline void col_degrees_streaming( const CSR<DataT, intT>& A,
                                           std::vector<size_t>& out ) {
            const size_t N = static_cast<size_t>( A.rows );
            const size_t M = static_cast<size_t>( A.cols );
            const size_t nnz = A.col_ind.size();
            out.assign( M, 0 );
            if ( M == 0 || nnz == 0 )
                return;
            std::vector<uint32_t> cdeg32( M, 0 );
            const int P = omp_get_max_threads();
#pragma omp parallel num_threads( P )
            {
                const int tid = omp_get_thread_num();
                size_t rs, re;
                equal_nnz_range( A.row_ptr, N, nnz, tid, P, rs, re );
                Coalesce64 cache;
                for ( size_t r = rs; r < re; ++r ) {
                    for ( intT k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k )
                        cache.bump( static_cast<uint32_t>( A.col_ind[k] ), cdeg32 );
                }
                cache.flush_into( cdeg32 );
            }
#pragma omp parallel for schedule( static )
            for ( size_t c = 0; c < M; ++c )
                out[c] = static_cast<size_t>( cdeg32[c] );
        }

        inline void degree_stats( const std::vector<size_t>& d,
                                  double& mean,
                                  double& stdv ) {
            const size_t n = d.size();
            if ( n == 0 ) {
                mean = 0.0;
                stdv = 0.0;
                return;
            }
            double s = 0.0;
#pragma omp parallel for reduction( + : s ) schedule( static )
            for ( size_t i = 0; i < n; ++i )
                s += static_cast<double>( d[i] );
            mean = s / static_cast<double>( n );
            double v = 0.0;
#pragma omp parallel for reduction( + : v ) schedule( static )
            for ( size_t i = 0; i < n; ++i ) {
                double dd = static_cast<double>( d[i] ) - mean;
                v += dd * dd;
            }
            stdv = std::sqrt( v / static_cast<double>( n ) );
        }

        // Hierarchical Lexicographic LSH Prefix Key:
        // Packs individual densified MinHash bins from MSB to LSB so radix sort
        // forms a prefix tree where rows sharing 1, 2, 3, 4, or 5 bins cluster
        // naturally without multi-bin avalanche destruction.
        inline uint64_t densify_and_pack( uint32_t* bins,
                                          size_t K,
                                          uint32_t eff_deg ) noexcept {
            for ( size_t k = 0; k < K; ++k ) {
                if ( bins[k] != UINT32_MAX )
                    continue;
                uint32_t att = 1;
                while ( bins[k] == UINT32_MAX && att <= K ) {
                    size_t nk = ( k + att ) & ( K - 1 );
                    if ( bins[nk] != UINT32_MAX )
                        bins[k] = mix32(
                            bins[nk] ^ ( ( static_cast<uint32_t>( k ) << 16 ) | att ) );
                    ++att;
                }
                if ( bins[k] == UINT32_MAX )
                    bins[k] = mix32( static_cast<uint32_t>( k ) ^ 0x85ebca6bU );
            }
            const uint64_t p0 = ( bins[0] >> 18 ) & 0x3FFFULL; // 14 bits (63..50)
            const uint64_t p1 = ( bins[1] >> 20 ) & 0xFFFULL;  // 12 bits (49..38)
            const uint64_t p2 = ( bins[2] >> 20 ) & 0xFFFULL;  // 12 bits (37..26)
            const uint64_t p3 = ( bins[3] >> 22 ) & 0x3FFULL;  // 10 bits (25..16)
            const uint64_t p4 = ( bins[4] >> 22 ) & 0x3FFULL;  // 10 bits (15..6)
            const uint64_t db = std::min<uint64_t>( 61, ilog2nz( eff_deg ) * 4 );

            uint64_t key = ( p0 << 50 ) | ( p1 << 38 ) | ( p2 << 26 ) |
                           ( p3 << 16 ) | ( p4 << 6 ) | db;
            if ( key >= UINT64_MAX - 1 )
                key = UINT64_MAX - 2;
            return key;
        }

        // Phase 2 (Rows): register-resident OPMH over A.
        template <typename DataT, typename intT>
        inline void opmh_sketches( const CSR<DataT, intT>& A,
                                   const Bitset& is_hub_row,
                                   const Bitset& is_hub_col,
                                   size_t K,
                                   uint64_t seed,
                                   std::vector<RowSketchItem>& out ) {
            const size_t N = static_cast<size_t>( A.rows );
            const size_t nnz = A.col_ind.size();
            out.resize( N );
            if ( N == 0 )
                return;
            expect( K >= 1 && ( K & ( K - 1 ) ) == 0 );
            const unsigned logK = ilog2nz( K );
            const int P = omp_get_max_threads();
#pragma omp parallel num_threads( P )
            {
                const int tid = omp_get_thread_num();
                size_t rs, re;
                equal_nnz_range( A.row_ptr, N, nnz, tid, P, rs, re );
                uint32_t bins[16];
                for ( size_t r = rs; r < re; ++r ) {
                    if ( is_hub_row.test( r ) ) {
                        out[r] = { UINT64_MAX, static_cast<uint32_t>( r ), UINT32_MAX };
                        continue;
                    }
                    for ( size_t k = 0; k < K; ++k )
                        bins[k] = UINT32_MAX;
                    uint32_t eff = 0;
                    for ( intT kk = A.row_ptr[r]; kk < A.row_ptr[r + 1]; ++kk ) {
                        size_t c = static_cast<size_t>( A.col_ind[kk] );
                        if ( is_hub_col.test( c ) )
                            continue;
                        ++eff;
                        uint64_t h = splitmix64( c ^ seed );
                        uint32_t b = ( logK >= 64 )
                                         ? 0
                                         : static_cast<uint32_t>( h >> ( 64 - logK ) );
                        uint32_t hv = static_cast<uint32_t>( h );
                        uint32_t idx = b & static_cast<uint32_t>( K - 1 );
                        if ( hv < bins[idx] )
                            bins[idx] = hv;
                    }
                    if ( eff == 0 ) {
                        out[r] = { UINT64_MAX - 1, static_cast<uint32_t>( r ), 0 };
                        continue;
                    }
                    uint64_t key = densify_and_pack( bins, K, eff );
                    out[r] = { key, static_cast<uint32_t>( r ), eff };
                }
            }
        }

        // Phase 2 (Columns, Zero AT Materialization): stream directly over A.
        template <typename DataT, typename intT>
        inline void opmh_col_sketches_streaming( const CSR<DataT, intT>& A,
                                                 const Bitset& is_hub_row,
                                                 const Bitset& is_hub_col,
                                                 const std::vector<size_t>& cdeg,
                                                 size_t K,
                                                 uint64_t seed,
                                                 std::vector<RowSketchItem>& out ) {
            const size_t N = static_cast<size_t>( A.rows );
            const size_t M = static_cast<size_t>( A.cols );
            const size_t nnz = A.col_ind.size();
            out.resize( M );
            if ( M == 0 )
                return;
            expect( K >= 1 && ( K & ( K - 1 ) ) == 0 );
            const unsigned logK = ilog2nz( K );
            std::vector<uint32_t> col_bins( M * K, UINT32_MAX );
            const int P = omp_get_max_threads();
#pragma omp parallel num_threads( P )
            {
                const int tid = omp_get_thread_num();
                size_t rs, re;
                equal_nnz_range( A.row_ptr, N, nnz, tid, P, rs, re );
                for ( size_t r = rs; r < re; ++r ) {
                    if ( is_hub_row.test( r ) )
                        continue;
                    uint64_t h = splitmix64( r ^ seed );
                    size_t b = ( logK >= 64 )
                                   ? 0
                                   : ( static_cast<size_t>( h >> ( 64 - logK ) ) &
                                       ( K - 1 ) );
                    uint32_t hv = static_cast<uint32_t>( h );
                    for ( intT kk = A.row_ptr[r]; kk < A.row_ptr[r + 1]; ++kk ) {
                        size_t c = static_cast<size_t>( A.col_ind[kk] );
                        if ( is_hub_col.test( c ) )
                            continue;
                        atomic_min_u32( &col_bins[( c << logK ) | b], hv );
                    }
                }
            }
#pragma omp parallel for schedule( static ) num_threads( P )
            for ( size_t c = 0; c < M; ++c ) {
                if ( is_hub_col.test( c ) ) {
                    out[c] = { UINT64_MAX, static_cast<uint32_t>( c ), UINT32_MAX };
                    continue;
                }
                uint32_t* bins = &col_bins[c << logK];
                bool any_pin = false;
                for ( size_t k = 0; k < K; ++k ) {
                    if ( bins[k] != UINT32_MAX ) {
                        any_pin = true;
                        break;
                    }
                }
                if ( !any_pin || cdeg[c] == 0 ) {
                    out[c] = { UINT64_MAX - 1, static_cast<uint32_t>( c ), 0 };
                    continue;
                }
                uint32_t eff = static_cast<uint32_t>(
                    std::min<size_t>( cdeg[c], UINT32_MAX - 1 ) );
                uint64_t key = densify_and_pack( bins, K, eff );
                out[c] = { key, static_cast<uint32_t>( c ), eff };
            }
        }

        // Phase 3: 8-bit LSD radix sort (256 buckets, L1-resident histograms).
        inline void radix_sort64( std::vector<RowSketchItem>& a ) {
            const size_t n = a.size();
            if ( n < 2 )
                return;
            if ( n < 2048 ) {
                std::stable_sort( a.begin(), a.end(), []( const auto& x, const auto& y ) {
                    return x.sort_key < y.sort_key;
                } );
                return;
            }
            std::vector<RowSketchItem> b( n );
            RowSketchItem* src = a.data();
            RowSketchItem* dst = b.data();
            const int P = omp_get_max_threads();
            std::vector<size_t> counts( static_cast<size_t>( P ) * 256, 0 );
            std::vector<size_t> off( static_cast<size_t>( P ) * 256, 0 );

            for ( int pass = 0; pass < 8; ++pass ) {
                const unsigned sh = static_cast<unsigned>( pass * 8 );
                std::fill( counts.begin(), counts.end(), 0 );
#pragma omp parallel num_threads( P )
                {
                    const int tid = omp_get_thread_num();
                    size_t* lc = counts.data() + static_cast<size_t>( tid ) * 256;
#pragma omp for schedule( static )
                    for ( size_t i = 0; i < n; ++i )
                        ++lc[( src[i].sort_key >> sh ) & 0xFFULL];
                }
                bool trivial = false;
                for ( size_t d = 0; d < 256; ++d ) {
                    size_t tot = 0;
                    for ( int t = 0; t < P; ++t )
                        tot += counts[static_cast<size_t>( t ) * 256 + d];
                    if ( tot == n ) {
                        trivial = true;
                        break;
                    }
                    if ( tot > 0 )
                        break;
                }
                if ( trivial )
                    continue;

                size_t run = 0;
                for ( size_t d = 0; d < 256; ++d ) {
                    for ( int t = 0; t < P; ++t ) {
                        off[static_cast<size_t>( t ) * 256 + d] = run;
                        run += counts[static_cast<size_t>( t ) * 256 + d];
                    }
                }
#pragma omp parallel num_threads( P )
                {
                    const int tid = omp_get_thread_num();
                    size_t* lo = off.data() + static_cast<size_t>( tid ) * 256;
#pragma omp for schedule( static )
                    for ( size_t i = 0; i < n; ++i ) {
                        size_t d = ( src[i].sort_key >> sh ) & 0xFFULL;
                        dst[lo[d]++] = src[i];
                    }
                }
                std::swap( src, dst );
            }
            if ( src != a.data() )
                std::copy( src, src + n, a.data() );
        }

        // Thread workspace: persistent SPA stamps + L1 window-local tile cache
        // (holds deduplicated tile IDs only for the active W*b window rows).
        struct alignas( 64 ) Workspace {
            std::vector<uint32_t> stamp;
            uint32_t cur = 1;
            std::vector<RowSketchItem> buf;
            std::vector<uint32_t> win_ptr;
            std::vector<uint32_t> win_tids;

            void init( size_t ntiles, size_t buf_n ) {
                stamp.assign( std::max<size_t>( 1, ntiles ), 0 );
                cur = 1;
                buf.resize( buf_n );
                win_ptr.resize( buf_n + 1, 0 );
                win_tids.reserve( buf_n * 64 );
            }

            inline void next_stripe() {
                if ( ++cur == 0 ) {
                    std::fill( stamp.begin(), stamp.end(), 0 );
                    cur = 1;
                }
            }
        };

        // Phase 4: Window-cached SPA guarded stripe packing.
        // Deduplicates tile IDs once per L-row window in L1/L2 cache, avoiding
        // both global TileCSR allocation and repeated CSR scans during W-lookahead.
        template <typename DataT, typename intT>
        inline void guarded_pack_direct( const CSR<DataT, intT>& A,
                                         const Bitset& is_hub_col,
                                         const std::vector<uint32_t>& col_perm_inv,
                                         unsigned log2_b,
                                         std::vector<RowSketchItem>& sk,
                                         size_t n_core,
                                         size_t b,
                                         double lambda,
                                         size_t W,
                                         std::vector<size_t>& row_perm ) {
            const size_t n_all = sk.size();
            row_perm.resize( n_all );
            if ( n_core == 0 ) {
                for ( size_t i = 0; i < n_all; ++i )
                    row_perm[i] = sk[i].row_id;
                return;
            }
            const size_t M = static_cast<size_t>( A.cols );
            const size_t ntiles = ( M + b - 1 ) / b;
            const uint32_t* cpinv = col_perm_inv.data();
            const size_t win_stripes = ( W > 0 ) ? W : 1;
            const size_t win_rows = win_stripes * b;
            const size_t n_windows = ( n_core + win_rows - 1 ) / win_rows;
            const int P = omp_get_max_threads();

#pragma omp parallel num_threads( P )
            {
                Workspace ws;
                ws.init( ntiles, win_rows );
                const double keep = ( 1.0 - lambda ) * 0.5;
                RowSketchItem* buf = ws.buf.data();

#pragma omp for schedule( dynamic, 1 )
                for ( size_t w_idx = 0; w_idx < n_windows; ++w_idx ) {
                    const size_t base = w_idx * win_rows;
                    const size_t lim = std::min( n_core, base + win_rows );
                    const size_t L = lim - base;

                    // 1. Gather & deduplicate tile IDs once for the L rows in this window
                    ws.win_tids.clear();
                    ws.win_ptr[0] = 0;
                    for ( size_t i = 0; i < L; ++i ) {
                        buf[i] = sk[base + i];
                        uint32_t r = buf[i].row_id;
                        size_t start_off = ws.win_tids.size();
                        for ( intT k = A.row_ptr[r]; k < A.row_ptr[r + 1]; ++k ) {
                            size_t c = static_cast<size_t>( A.col_ind[k] );
                            if ( is_hub_col.test( c ) )
                                continue;
                            ws.win_tids.push_back( cpinv[c] >> log2_b );
                        }
                        auto it_beg = ws.win_tids.begin() + static_cast<ptrdiff_t>( start_off );
                        std::sort( it_beg, ws.win_tids.end() );
                        ws.win_tids.erase( std::unique( it_beg, ws.win_tids.end() ),
                                           ws.win_tids.end() );
                        buf[i].eff_deg = static_cast<uint32_t>( i );
                        ws.win_ptr[i + 1] = static_cast<uint32_t>( ws.win_tids.size() );
                    }

                    // 2. Pack L rows into b-row stripes using L1 window tile lists
                    const uint32_t* tids = ws.win_tids.data();
                    uint32_t* st = ws.stamp.data();

                    for ( size_t pos = 0; pos < L; ++pos ) {
                        if ( ( ( base + pos ) % b ) == 0 ) {
                            ws.next_stripe();
                            uint32_t wi = buf[pos].eff_deg;
                            uint32_t s = ws.win_ptr[wi], e = ws.win_ptr[wi + 1];
                            uint32_t cur_st = ws.cur;
                            for ( uint32_t k = s; k < e; ++k )
                                st[tids[k]] = cur_st;
                            row_perm[base + pos] = buf[pos].row_id;
                            continue;
                        }

                        auto eval_slot = [&]( size_t slot ) noexcept {
                            uint32_t wi = buf[slot].eff_deg;
                            uint32_t s = ws.win_ptr[wi], e = ws.win_ptr[wi + 1];
                            uint32_t nw = 0, cur_st = ws.cur;
                            for ( uint32_t k = s; k < e; ++k )
                                if ( st[tids[k]] != cur_st )
                                    ++nw;
                            return std::pair<uint32_t, uint32_t>{ nw, e - s };
                        };

                        auto [nw, tot] = eval_slot( pos );
                        size_t best = pos;
                        if ( tot > 0 &&
                             static_cast<double>( tot - nw ) <
                                 keep * static_cast<double>( tot ) ) {
                            uint32_t best_cost = nw;
                            const size_t le = std::min( L - 1, pos + W );
                            for ( size_t pr = pos + 1; pr <= le; ++pr ) {
                                auto [pn, pt] = eval_slot( pr );
                                if ( pn < best_cost ) {
                                    best_cost = pn;
                                    best = pr;
                                    if ( static_cast<double>( pt - pn ) >=
                                         keep * static_cast<double>( pt ) )
                                        break;
                                }
                            }
                            if ( best != pos )
                                std::swap( buf[pos], buf[best] );
                        }

                        uint32_t wi = buf[pos].eff_deg;
                        uint32_t s = ws.win_ptr[wi], e = ws.win_ptr[wi + 1];
                        uint32_t cur_st = ws.cur;
                        for ( uint32_t k = s; k < e; ++k )
                            st[tids[k]] = cur_st;
                        row_perm[base + pos] = buf[pos].row_id;
                    }
                }
            }
#pragma omp parallel for schedule( static )
            for ( size_t i = n_core; i < n_all; ++i )
                row_perm[i] = sk[i].row_id;
        }

        // Fast sampled check for structural symmetry on square matrices.
        template <typename DataT, typename intT>
        inline bool is_structurally_symmetric_sample( const CSR<DataT, intT>& A ) {
            if ( A.rows != A.cols )
                return false;
            const size_t N = static_cast<size_t>( A.rows );
            if ( N == 0 || A.col_ind.empty() )
                return true;
            const size_t step = std::max<size_t>( 1, N / 512 );
            size_t checked = 0;
            size_t matched = 0;
            for ( size_t r = 0; r < N; r += step ) {
                intT s = A.row_ptr[r], e = A.row_ptr[r + 1];
                if ( s == e )
                    continue;
                size_t c = static_cast<size_t>( A.col_ind[s + ( e - s ) / 2] );
                if ( c >= N )
                    return false;
                ++checked;
                intT cs = A.row_ptr[c], ce = A.row_ptr[c + 1];
                if ( std::binary_search(
                         A.col_ind.begin() + cs, A.col_ind.begin() + ce, static_cast<intT>( r ) ) ||
                     std::find( A.col_ind.begin() + cs,
                                A.col_ind.begin() + ce,
                                static_cast<intT>( r ) ) != A.col_ind.begin() + ce ) {
                    ++matched;
                }
            }
            return checked == 0 || ( matched * 10 >= checked * 9 );
        }

    } // namespace detail

    inline void check_params( const Params& p ) {
        expect( p.b == 4 || p.b == 8 );
        expect( p.K == 8 || p.K == 16 );
        expect( p.lambda >= 0.0 && p.lambda <= 1.0 );
        expect( p.W >= 1 && p.W <= 128 );
        expect( p.hub_frac > 0.0 && p.hub_frac <= 0.5 );
    }

    inline unsigned log2_pow2( size_t b ) {
        expect( b == 4 || b == 8 );
        return ( b == 4 ) ? 2U : 3U;
    }

    // Work-balanced row permutation apply.
    template <typename DataT, typename intT>
    inline void apply_row_perm( CSR<DataT, intT>& A, const std::vector<size_t>& P ) {
        const size_t N = static_cast<size_t>( A.rows );
        expect( P.size() == N );
        std::vector<intT> nrp( N + 1, 0 );
        for ( size_t i = 0; i < N; ++i )
            nrp[i + 1] = nrp[i] + ( A.row_ptr[P[i] + 1] - A.row_ptr[P[i]] );
        const size_t nnz = static_cast<size_t>( nrp[N] );
        std::vector<intT> nci( nnz );
        std::vector<DataT> nv;
        if ( !A.pattern_only )
            nv.resize( nnz );
        const int nthreads = omp_get_max_threads();
#pragma omp parallel num_threads( nthreads )
        {
            const int tid = omp_get_thread_num();
            size_t rs, re;
            detail::equal_nnz_range( nrp, N, nnz, tid, nthreads, rs, re );
            for ( size_t i = rs; i < re; ++i ) {
                intT s = A.row_ptr[P[i]], e = A.row_ptr[P[i] + 1], d = nrp[i];
                for ( intT k = s; k < e; ++k ) {
                    nci[d + ( k - s )] = A.col_ind[k];
                    if ( !A.pattern_only )
                        nv[d + ( k - s )] = A.values[k];
                }
            }
        }
        A.row_ptr = std::move( nrp );
        A.col_ind = std::move( nci );
        if ( !A.pattern_only )
            A.values = std::move( nv );
        A.nzcount.resize( N );
#pragma omp parallel for schedule( static )
        for ( size_t i = 0; i < N; ++i )
            A.nzcount[i] = A.row_ptr[i + 1] - A.row_ptr[i];
    }

    // Work-balanced column permutation apply with thread-privatized scratch buffer.
    template <typename DataT, typename intT>
    inline void apply_col_perm( CSR<DataT, intT>& A, const std::vector<size_t>& Q ) {
        const size_t M = static_cast<size_t>( A.cols );
        const size_t N = static_cast<size_t>( A.rows );
        const size_t nnz = A.col_ind.size();
        expect( Q.size() == M );
        std::vector<intT> inv( M );
#pragma omp parallel for schedule( static )
        for ( size_t j = 0; j < M; ++j )
            inv[Q[j]] = static_cast<intT>( j );

        const int nthreads = omp_get_max_threads();
#pragma omp parallel num_threads( nthreads )
        {
            const int tid = omp_get_thread_num();
            size_t rs, re;
            detail::equal_nnz_range( A.row_ptr, N, nnz, tid, nthreads, rs, re );
            std::vector<std::pair<intT, DataT>> tmp;
            for ( size_t i = rs; i < re; ++i ) {
                intT s = A.row_ptr[i], e = A.row_ptr[i + 1];
                if ( e == s )
                    continue;
                if ( A.pattern_only ) {
                    for ( intT k = s; k < e; ++k )
                        A.col_ind[k] = inv[static_cast<size_t>( A.col_ind[k] )];
                    std::sort( A.col_ind.begin() + s, A.col_ind.begin() + e );
                } else {
                    tmp.clear();
                    tmp.reserve( static_cast<size_t>( e - s ) );
                    for ( intT k = s; k < e; ++k )
                        tmp.emplace_back( inv[static_cast<size_t>( A.col_ind[k] )],
                                           A.values[k] );
                    std::sort(
                        tmp.begin(), tmp.end(), []( const auto& a, const auto& b ) {
                            return a.first < b.first;
                        } );
                    for ( size_t t = 0; t < tmp.size(); ++t ) {
                        A.col_ind[s + static_cast<intT>( t )] = tmp[t].first;
                        A.values[s + static_cast<intT>( t )] = tmp[t].second;
                    }
                }
            }
        }
    }

    // Row pass (rectangular): uses precomputed hub_row, is_hub_col, and col_perm_inv.
    template <typename DataT, typename intT>
    inline void row_pass( const CSR<DataT, intT>& A,
                          const Params& p,
                          const Bitset& is_hub_col,
                          const std::vector<uint32_t>& col_perm_inv,
                          std::vector<size_t>& row_perm,
                          Stats* st,
                          const Bitset* pre_hub_row = nullptr,
                          size_t pre_tau_row = 0 ) {
        const size_t N = static_cast<size_t>( A.rows );
        row_perm.assign( N, 0 );
        if ( N == 0 )
            return;

        Bitset local_hub_row;
        size_t tau_row = pre_tau_row;
        const Bitset* hub_row_ptr = pre_hub_row;
        if ( !hub_row_ptr ) {
            std::vector<size_t> rdeg;
            detail::row_degrees( A, rdeg );
            double mn, sd;
            detail::degree_stats( rdeg, mn, sd );
            tau_row = detail::adaptive_tau(
                mn, sd, detail::quantile_tau( rdeg, p.hub_frac ), 64 );
            local_hub_row.resize( N );
            const int P = omp_get_max_threads();
#pragma omp parallel for schedule( static ) num_threads( P )
            for ( size_t r = 0; r < N; ++r )
                if ( rdeg[r] >= tau_row )
                    local_hub_row.set_atomic( r );
            hub_row_ptr = &local_hub_row;
        }

        std::vector<detail::RowSketchItem> sk;
        detail::opmh_sketches( A, *hub_row_ptr, is_hub_col, p.K, p.seed, sk );
        detail::radix_sort64( sk );
        size_t n_core = 0;
        while ( n_core < N && sk[n_core].sort_key < UINT64_MAX - 1 )
            ++n_core;

        detail::guarded_pack_direct( A,
                                     is_hub_col,
                                     col_perm_inv,
                                     log2_pow2( p.b ),
                                     sk,
                                     n_core,
                                     p.b,
                                     p.lambda,
                                     p.W,
                                     row_perm );
        if ( st ) {
            st->tau_row = tau_row;
            st->n_hub_rows = N - n_core;
            st->n_core_rows = n_core;
        }
    }

    // Full rectangular flow:
    //   1. Compute row & column hub bitsets in parallel.
    //   2. Stream over A to compute column OPMH sketches -> radix -> col_perm.
    //   3. Stream over A to compute row OPMH sketches -> radix -> window-cached SPA pack.
    template <typename DataT, typename intT>
    void reorder_rectangular( CSR<DataT, intT>& A,
                              Params p = Params{},
                              std::vector<size_t>* out_row_perm = nullptr,
                              std::vector<size_t>* out_col_perm = nullptr,
                              Stats* st = nullptr ) {
        check_params( p );
        const size_t N = static_cast<size_t>( A.rows );
        const size_t M = static_cast<size_t>( A.cols );
        LOG_INFO( "msg", "HALO rectangular start", "b", p.b );

        Stats loc;
        std::vector<size_t> rdeg, cdeg;
        detail::row_degrees( A, rdeg );
        detail::col_degrees_streaming( A, cdeg );

        double r_mn, r_sd, c_mn, c_sd;
        detail::degree_stats( rdeg, r_mn, r_sd );
        detail::degree_stats( cdeg, c_mn, c_sd );

        const size_t tau_row = detail::adaptive_tau(
            r_mn, r_sd, detail::quantile_tau( rdeg, p.hub_frac ), 64 );
        const size_t tau_col = detail::adaptive_tau(
            c_mn, c_sd, detail::quantile_tau( cdeg, p.hub_frac ), 64 );

        Bitset is_hub_row( N ), is_hub_col( M );
#pragma omp parallel for schedule( static )
        for ( size_t r = 0; r < N; ++r )
            if ( rdeg[r] >= tau_row )
                is_hub_row.set_atomic( r );
#pragma omp parallel for schedule( static )
        for ( size_t c = 0; c < M; ++c )
            if ( cdeg[c] >= tau_col )
                is_hub_col.set_atomic( c );

        std::vector<uint32_t> col_perm( M ), col_perm_inv( M );
        size_t n_core_cols = 0;
        if ( M > 0 ) {
            std::vector<detail::RowSketchItem> sk_col;
            detail::opmh_col_sketches_streaming(
                A, is_hub_row, is_hub_col, cdeg, p.K, p.seed ^ 0x9e3779b9ULL, sk_col );
            detail::radix_sort64( sk_col );
            while ( n_core_cols < M && sk_col[n_core_cols].sort_key < UINT64_MAX - 1 )
                ++n_core_cols;
#pragma omp parallel for schedule( static )
            for ( size_t j = 0; j < M; ++j ) {
                uint32_t cj = sk_col[j].row_id;
                col_perm[j] = cj;
                col_perm_inv[cj] = static_cast<uint32_t>( j );
            }
        }
        loc.tau_col = tau_col;
        loc.n_hub_cols = M - n_core_cols;
        loc.n_core_cols = n_core_cols;

        std::vector<size_t> row_perm;
        row_pass( A, p, is_hub_col, col_perm_inv, row_perm, &loc, &is_hub_row, tau_row );

        {
            std::vector<char> seen( N, 0 );
            for ( size_t x : row_perm ) {
                expect( x < N && !seen[x] );
                seen[x] = 1;
            }
            seen.assign( M, 0 );
            for ( uint32_t x : col_perm ) {
                expect( static_cast<size_t>( x ) < M && !seen[x] );
                seen[x] = 1;
            }
        }

        club::halo::apply_row_perm( A, row_perm );
        std::vector<size_t> Q( M );
#pragma omp parallel for schedule( static )
        for ( size_t j = 0; j < M; ++j )
            Q[j] = col_perm[j];
        club::halo::apply_col_perm( A, Q );

        // if ( st ) {
        //     loc.ntiles = count_nonzero_blocks( A, p.b, p.b );
        //     *st = loc;
        // }
        LOG_INFO( "msg",
                  "HALO rectangular done",
                  "tau_row",
                  loc.tau_row,
                  "tau_col",
                  loc.tau_col );
        if ( out_row_perm )
            *out_row_perm = std::move( row_perm );
        if ( out_col_perm )
            *out_col_perm = std::move( Q );
    }

    // Symmetric flow: hub_row masks both rows and columns; LSH order provides
    // provisional column tiles for window-cached SPA packing before PAP^T apply.
    template <typename DataT, typename intT>
    void reorder_symmetric( CSR<DataT, intT>& A,
                            Params p = Params{},
                            std::vector<size_t>* out_perm = nullptr,
                            Stats* st = nullptr ) {
        check_params( p );
        const size_t N = static_cast<size_t>( A.rows );
        const size_t M = static_cast<size_t>( A.cols );
        expect( N == M );
        LOG_INFO( "msg", "HALO symmetric start", "b", p.b );

        Stats loc;
        std::vector<size_t> P( N, 0 );
        if ( N > 0 ) {
            std::vector<size_t> rdeg;
            detail::row_degrees( A, rdeg );
            double mn, sd;
            detail::degree_stats( rdeg, mn, sd );
            size_t tau = detail::adaptive_tau(
                mn, sd, detail::quantile_tau( rdeg, p.hub_frac ), 64 );
            Bitset is_hub( N );
#pragma omp parallel for schedule( static )
            for ( size_t r = 0; r < N; ++r )
                if ( rdeg[r] >= tau )
                    is_hub.set_atomic( r );

            std::vector<detail::RowSketchItem> sk;
            detail::opmh_sketches( A, is_hub, is_hub, p.K, p.seed, sk );
            detail::radix_sort64( sk );

            size_t n_core = 0;
            while ( n_core < N && sk[n_core].sort_key < UINT64_MAX - 1 )
                ++n_core;

            std::vector<uint32_t> lsh_col_inv( N );
#pragma omp parallel for schedule( static )
            for ( size_t i = 0; i < N; ++i )
                lsh_col_inv[sk[i].row_id] = static_cast<uint32_t>( i );

            detail::guarded_pack_direct( A,
                                         is_hub,
                                         lsh_col_inv,
                                         log2_pow2( p.b ),
                                         sk,
                                         n_core,
                                         p.b,
                                         p.lambda,
                                         p.W,
                                         P );
            loc.tau_row = tau;
            loc.tau_col = tau;
            loc.n_hub_rows = N - n_core;
            loc.n_hub_cols = N - n_core;
            loc.n_core_rows = n_core;
            loc.n_core_cols = n_core;
        }

        {
            std::vector<char> seen( N, 0 );
            for ( size_t x : P ) {
                expect( x < N && !seen[x] );
                seen[x] = 1;
            }
        }

        club::halo::apply_row_perm( A, P );
        club::halo::apply_col_perm( A, P );
        // if ( st ) {
        //     loc.ntiles = count_nonzero_blocks( A, p.b, p.b );
        //     *st = loc;
        // }
        LOG_INFO( "msg", "HALO symmetric done", "tau", loc.tau_row );
        if ( out_perm )
            *out_perm = std::move( P );
    }

    // Default entry: dispatches to symmetric only when square AND structurally symmetric.
    template <typename DataT, typename intT>
    void reorder( CSR<DataT, intT>& A,
                  Params p = Params{},
                  std::vector<size_t>* out_row_perm = nullptr,
                  std::vector<size_t>* out_col_perm = nullptr,
                  Stats* st = nullptr ) {
        if ( A.rows == A.cols && detail::is_structurally_symmetric_sample( A ) ) {
            std::vector<size_t> sym_perm;
            reorder_symmetric( A, p, &sym_perm, st );
            if ( out_col_perm )
                *out_col_perm = sym_perm;
            if ( out_row_perm )
                *out_row_perm = std::move( sym_perm );
        } else {
            reorder_rectangular( A, p, out_row_perm, out_col_perm, st );
        }
    }

} // namespace club::halo