#ifndef __MMIO_C_WRAPPER_H__
#define __MMIO_C_WRAPPER_H__

#include <stdbool.h>
#include <stdint.h> // For uint32_t, uint64_t
#include <stddef.h> // For size_t

// This guard allows the header to be included by C++ source files
#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================================
 * C-style structs mirroring the C++ CSR and COO templates.
 * One struct is defined for each supported (IndexType, ValueType) combination.
 * ============================================================================
 */

// --- Combination: uint32_t index, float value ---
typedef struct {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    float*    val;
} mmio_csr_u32_f32_t;

typedef struct {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t* row;
    uint32_t* col;
    float*    val;
} mmio_coo_u32_f32_t;

// --- Combination: uint32_t index, double value ---
typedef struct {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    double*   val;
} mmio_csr_u32_f64_t;

typedef struct {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t* row;
    uint32_t* col;
    double*   val;
} mmio_coo_u32_f64_t;

// --- Combination: uint64_t index, float value ---
typedef struct {
    uint64_t nrows;
    uint64_t ncols;
    uint64_t nnz;
    uint64_t* row_ptr;
    uint64_t* col_idx;
    float*    val;
} mmio_csr_u64_f32_t;

typedef struct {
    uint64_t nrows;
    uint64_t ncols;
    uint64_t nnz;
    uint64_t* row;
    uint64_t* col;
    float*    val;
} mmio_coo_u64_f32_t;

// --- Combination: uint64_t index, double value ---
typedef struct {
    uint64_t nrows;
    uint64_t ncols;
    uint64_t nnz;
    uint64_t* row_ptr;
    uint64_t* col_idx;
    double*   val;
} mmio_csr_u64_f64_t;

typedef struct {
    uint64_t nrows;
    uint64_t ncols;
    uint64_t nnz;
    uint64_t* row;
    uint64_t* col;
    double*   val;
} mmio_coo_u64_f64_t;


/*
 * ============================================================================
 * C-friendly function wrappers.
 * A function is provided for each data format and type combination.
 * The naming convention is: mmio_read/destroy_<format>_<index_type>_<value_type>
 * ============================================================================
 */

// --- uint32_t / float ---
mmio_csr_u32_f32_t* mmio_read_csr_u32_f32(const char* filename, bool alloc_val);
mmio_coo_u32_f32_t* mmio_read_coo_u32_f32(const char* filename, bool alloc_val);
void mmio_destroy_csr_u32_f32(mmio_csr_u32_f32_t* matrix);
void mmio_destroy_coo_u32_f32(mmio_coo_u32_f32_t* matrix);

// --- uint32_t / double ---
mmio_csr_u32_f64_t* mmio_read_csr_u32_f64(const char* filename, bool alloc_val);
mmio_coo_u32_f64_t* mmio_read_coo_u32_f64(const char* filename, bool alloc_val);
void mmio_destroy_csr_u32_f64(mmio_csr_u32_f64_t* matrix);
void mmio_destroy_coo_u32_f64(mmio_coo_u32_f64_t* matrix);

// --- uint64_t / float ---
mmio_csr_u64_f32_t* mmio_read_csr_u64_f32(const char* filename, bool alloc_val);
mmio_coo_u64_f32_t* mmio_read_coo_u64_f32(const char* filename, bool alloc_val);
void mmio_destroy_csr_u64_f32(mmio_csr_u64_f32_t* matrix);
void mmio_destroy_coo_u64_f32(mmio_coo_u64_f32_t* matrix);

// --- uint64_t / double ---
mmio_csr_u64_f64_t* mmio_read_csr_u64_f64(const char* filename, bool alloc_val);
mmio_coo_u64_f64_t* mmio_read_coo_u64_f64(const char* filename, bool alloc_val);
void mmio_destroy_csr_u64_f64(mmio_csr_u64_f64_t* matrix);
void mmio_destroy_coo_u64_f64(mmio_coo_u64_f64_t* matrix);


#ifdef __cplusplus
}
#endif

#endif // __MMIO_C_WRAPPER_H__