#pragma once
#ifdef __AVX512F__
#include <immintrin.h>
#include "club/matrices.hpp"

namespace club::simd {

// ── Sorted set intersection via 8-way rotation ────────────────────────────
//
// For each pair of 8-element chunks [ia..ia+7] × [ib..ib+7], all 64 pairs
// are checked with 8 SIMD cmpeq + rotate steps instead of a scalar merge.
//
// Correctness: match(a[i], b[j]) fires exactly at rotation r = (j-i) mod 8.
// This is unique because both arrays have distinct elements — no double-count.
//
// Advancement: both pointers move based on each chunk's max element, mirroring
// the invariant of a scalar merge-join but operating on 8-element strides.

inline size_t intersect(const size_t* __restrict__ a, size_t na,
                        const size_t* __restrict__ b, size_t nb) noexcept {
    size_t count = 0, ia = 0, ib = 0;

    while (ia + 8 <= na && ib + 8 <= nb) {
        __m512i va   = _mm512_loadu_si512(a + ia);
        __m512i vb_r = _mm512_loadu_si512(b + ib);
        const size_t a_max = a[ia + 7]; // arrays are sorted → last = max
        const size_t b_max = b[ib + 7];

        // 8 rotations cover all 64 (a[i], b[j]) pairs exactly once each
        for (int r = 0; r < 8; ++r) {
            count += __builtin_popcount(
                static_cast<unsigned>(_mm512_cmpeq_epi64_mask(va, vb_r)));
            vb_r = _mm512_alignr_epi64(vb_r, vb_r, 1); // left-rotate by 1 lane
        }

        // Advance the side(s) whose max is no larger.
        // Both advance when equal: their shared max was already counted.
        if (a_max <= b_max) ia += 8;
        if (b_max <= a_max) ib += 8;
    }

    // Scalar tail for the remaining < 8 elements on either side
    while (ia < na && ib < nb) {
        if      (a[ia] == b[ib]) { ++count; ++ia; ++ib; }
        else if (a[ia] <  b[ib]) { ++ia; }
        else                      { ++ib; }
    }
    return count;
}

// Jaccard distance using the AVX512 intersection above
inline double jaccard(const CSR<size_t, size_t>& M,
                      size_t ri, size_t rj) noexcept {
    const size_t ni = M.nzcount[ri], nj = M.nzcount[rj];
    if (ni == 0 && nj == 0) return 0.0;
    if (ni == 0 || nj == 0) return 1.0;
    const size_t isect = intersect(
        M.col_ind.data() + M.row_ptr[ri], ni,
        M.col_ind.data() + M.row_ptr[rj], nj);
    return 1.0 - static_cast<double>(isect) / (ni + nj - isect);
}

// ── First non-zero column lookup via masked gather ─────────────────────────
//
// Replaces 8 serial col_ind[row_ptr[i]] dereferences with one gather.
// Masked-off lanes (nzcount == 0) receive `empty_sentinel` without a
// memory access, so no out-of-bounds concern for the last empty rows.

inline void fetch_first_cols(const CSR<size_t, size_t>& Ahat,
                              size_t  i_base,
                              size_t* out,
                              size_t  empty_sentinel) noexcept {
    __m512i vrow_ptr = _mm512_loadu_si512(Ahat.row_ptr.data() + i_base);
    __m512i vnzcount = _mm512_loadu_si512(Ahat.nzcount.data() + i_base);
    __mmask8 has_nz  = _mm512_cmpgt_epu64_mask(vnzcount, _mm512_setzero_si512());

    __m512i result = _mm512_mask_i64gather_epi64(
        _mm512_set1_epi64(static_cast<long long>(empty_sentinel)), // default for empty rows
        has_nz,
        vrow_ptr,
        Ahat.col_ind.data(),
        sizeof(size_t));  // scale=8: index → byte offset

    _mm512_storeu_si512(out, result);
}

} // namespace club::simd
#endif // __AVX512F__