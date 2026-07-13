#include "../../include/mmio/mmio.h"      // Original C++ library header
#include "../../include/mmio/io.h"        // Original C++ library header
#include "../../include/c/wrapper.h" // New C API header

using Matrix_Metadata = mmio::Matrix_Metadata;
template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
template<typename IT, typename VT> using COO = mmio::COO<IT, VT>;
template<typename IT, typename VT> using CSR = mmio::CSR<IT, VT>;

// The entire file provides C-linkage, so we wrap it in extern "C".
extern "C" {

/*
 * ============================================================================
 * Implementations for uint32_t / float
 * ============================================================================
 */
mmio_csr_u32_f32_t* mmio_read_csr_u32_f32(const char* filename, bool alloc_val) {
    // Call the original C++ templated function
    CSR<uint32_t, float>* cpp_csr = mmio::CSR_read<uint32_t, float>(filename, alloc_val, NULL);
    // Cast the result to the C-style struct pointer. This is safe because layouts match.
    return reinterpret_cast<mmio_csr_u32_f32_t*>(cpp_csr);
}

mmio_coo_u32_f32_t* mmio_read_coo_u32_f32(const char* filename, bool alloc_val) {
    COO<uint32_t, float>* cpp_coo = mmio::COO_read<uint32_t, float>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_coo_u32_f32_t*>(cpp_coo);
}

void mmio_destroy_csr_u32_f32(mmio_csr_u32_f32_t* matrix) {
    // Cast the C-style pointer back to the C++ type
    CSR<uint32_t, float>* cpp_csr = reinterpret_cast<CSR<uint32_t, float>*>(matrix);
    // Call the C++ destroy function, which expects a pointer-to-pointer
    CSR_destroy(&cpp_csr);
}

void mmio_destroy_coo_u32_f32(mmio_coo_u32_f32_t* matrix) {
    COO<uint32_t, float>* cpp_coo = reinterpret_cast<COO<uint32_t, float>*>(matrix);
    COO_destroy(&cpp_coo);
}


/*
 * ============================================================================
 * Implementations for uint32_t / double
 * ============================================================================
 */
mmio_csr_u32_f64_t* mmio_read_csr_u32_f64(const char* filename, bool alloc_val) {
    CSR<uint32_t, double>* cpp_csr = mmio::CSR_read<uint32_t, double>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_csr_u32_f64_t*>(cpp_csr);
}

mmio_coo_u32_f64_t* mmio_read_coo_u32_f64(const char* filename, bool alloc_val) {
    COO<uint32_t, double>* cpp_coo = mmio::COO_read<uint32_t, double>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_coo_u32_f64_t*>(cpp_coo);
}

void mmio_destroy_csr_u32_f64(mmio_csr_u32_f64_t* matrix) {
    CSR<uint32_t, double>* cpp_csr = reinterpret_cast<CSR<uint32_t, double>*>(matrix);
    CSR_destroy(&cpp_csr);
}

void mmio_destroy_coo_u32_f64(mmio_coo_u32_f64_t* matrix) {
    COO<uint32_t, double>* cpp_coo = reinterpret_cast<COO<uint32_t, double>*>(matrix);
    COO_destroy(&cpp_coo);
}


/*
 * ============================================================================
 * Implementations for uint64_t / float
 * ============================================================================
 */
mmio_csr_u64_f32_t* mmio_read_csr_u64_f32(const char* filename, bool alloc_val) {
    CSR<uint64_t, float>* cpp_csr = mmio::CSR_read<uint64_t, float>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_csr_u64_f32_t*>(cpp_csr);
}

mmio_coo_u64_f32_t* mmio_read_coo_u64_f32(const char* filename, bool alloc_val) {
    COO<uint64_t, float>* cpp_coo = mmio::COO_read<uint64_t, float>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_coo_u64_f32_t*>(cpp_coo);
}

void mmio_destroy_csr_u64_f32(mmio_csr_u64_f32_t* matrix) {
    CSR<uint64_t, float>* cpp_csr = reinterpret_cast<CSR<uint64_t, float>*>(matrix);
    CSR_destroy(&cpp_csr);
}

void mmio_destroy_coo_u64_f32(mmio_coo_u64_f32_t* matrix) {
    COO<uint64_t, float>* cpp_coo = reinterpret_cast<COO<uint64_t, float>*>(matrix);
    COO_destroy(&cpp_coo);
}


/*
 * ============================================================================
 * Implementations for uint64_t / double
 * ============================================================================
 */
mmio_csr_u64_f64_t* mmio_read_csr_u64_f64(const char* filename, bool alloc_val) {
    CSR<uint64_t, double>* cpp_csr = mmio::CSR_read<uint64_t, double>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_csr_u64_f64_t*>(cpp_csr);
}

mmio_coo_u64_f64_t* mmio_read_coo_u64_f64(const char* filename, bool alloc_val) {
    COO<uint64_t, double>* cpp_coo = mmio::COO_read<uint64_t, double>(filename, alloc_val, NULL);
    return reinterpret_cast<mmio_coo_u64_f64_t*>(cpp_coo);
}

void mmio_destroy_csr_u64_f64(mmio_csr_u64_f64_t* matrix) {
    CSR<uint64_t, double>* cpp_csr = reinterpret_cast<CSR<uint64_t, double>*>(matrix);
    CSR_destroy(&cpp_csr);
}

void mmio_destroy_coo_u64_f64(mmio_coo_u64_f64_t* matrix) {
    COO<uint64_t, double>* cpp_coo = reinterpret_cast<COO<uint64_t, double>*>(matrix);
    COO_destroy(&cpp_coo);
}

} // extern "C"