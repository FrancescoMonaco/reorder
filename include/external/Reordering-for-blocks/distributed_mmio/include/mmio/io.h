#ifndef __MMIO_IO_H__
#define __MMIO_IO_H__

#include <cstdio>

#include "macros.h"
#include "mmio.h"

namespace mmio::io {

  // This is used for parsing the MTX files
  template<typename IT, typename VT>
  struct Entry {
    IT row;
    IT col;
    VT val;
  };

  template<typename IT, typename VT>
  void Entries_to_CSR(Entry<IT, VT> *entries, CSR<IT, VT> *csr);
  
  template<typename IT, typename VT>
  void Entries_to_COO(Entry<IT, VT> *entries, COO<IT, VT> *coo);
    
  /// @brief Serializes MM_typecode
  /// @param matcode 
  /// @return Serialized `matcode`
  char *mm_typecode_to_str(MM_typecode matcode);

  /// @brief Read the first line of the MTX file and populates `matcode`
  /// @param f The file to read from 
  /// @param matcode 
  /// @param is_bmtx If `true`, the function expects to find sizes (in bytes) for indices and values
  /// @param meta
  /// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
  int mm_read_banner(FILE *f, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta);

  /// @brief Reads and parses the MTX line containing matrix `nrows ncols nnz`
  /// @param f The file to read from
  /// @param M Number of rows will be stored here
  /// @param N Number of columns will be stored here
  /// @param nz Number of non-zeros will be stored here
  /// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
  int mm_read_mtx_crd_size(FILE *f, uint64_t *M, uint64_t *N, uint64_t *nz);

  /// @brief Reads and parses entries (i.e. MTX couples or triples)
  /// @tparam IT Index Data Type
  /// @tparam VT Value Data Type
  /// @param f The file to read from
  /// @param nentries Number of entries in the file (not necessarily the number of non-zeros)
  /// @param entries Entries will be written here
  /// @param matcode 
  /// @param is_bmtx 
  /// @param idx_bytes 
  /// @param value_bytes 
  /// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
  template<typename IT, typename VT>
  int mm_read_mtx_crd_data(FILE *f, int nentries, Entry<IT, VT> entries[], MM_typecode *matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t value_bytes);

  /// @brief Reads and parses the header including matrix shape (nrows, ncols, nnz)
  /// @tparam IT Index Data Type
  /// @tparam VT Value Data Type
  /// @param f The file to read from
  /// @param nrows 
  /// @param ncols 
  /// @param nentries number of entries in the MTX file 
  /// @param nnz_upperbound This is not equal in the case of symmetric matrices that have entries on the diagonal
  /// @param matcode 
  /// @param is_bmtx 
  /// @param meta 
  /// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
  template<typename IT, typename VT>
  int mm_parse_header(FILE *f, IT &nrows, IT &ncols, IT &nentries, IT &nnz_upperbound, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta);

  /// @brief Complete "non-distributed" MTX parsing. Reads and parses the header and (raw) entries 
  /// @tparam IT Index Data Type
  /// @tparam VT Value Data Type
  /// @param f The file to read from
  /// @param nrows 
  /// @param ncols 
  /// @param nnz 
  /// @param matcode 
  /// @param is_bmtx 
  /// @param meta 
  /// @return An "heap-allocated" array of entries. Not sorted (retain MTX file entries ordering). In case of symmetric matrices, entries will be explicitly duplicated.
  template<typename IT, typename VT>
  Entry<IT, VT>* mm_parse_file(FILE *f, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta);

  /// @brief If the matrix is symmetric (from `meta`) duplicate entries. This wont duplicate entries on the diagonal.
  /// @tparam IT Index Data Type
  /// @tparam VT Value Data Type
  /// @param entries 
  /// @param nentries 
  /// @param meta 
  /// @return The actual number of non-zeros
  template<typename IT, typename VT>
  IT mm_duplicate_entries_for_symmetric_matrices(Entry<IT, VT>* entries, IT nentries, IT nnz_upperbound, Matrix_Metadata* meta);

  template<typename IT, typename VT>
  IT mm_count_duplicates(Entry<IT, VT>* entries, IT nentries, IT nnz_upperbound, Matrix_Metadata* meta);

  /// @brief Sets `meta` values form `matcode`
  /// @param meta 
  /// @param matcode 
  void mm_set_metadata(Matrix_Metadata* meta, MM_typecode *matcode);

  /// @brief Writes COO to MTX file
  /// @tparam IT Index Data Type
  /// @tparam VT Value Data Type
  /// @param f The file to write to
  /// @param coo 
  /// @param meta Information to construct the header
  /// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
  template<typename IT, typename VT>
  int mm_write_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta);

  /// @brief Computes the minimum number of bytes required to store indices that range from `0` to `maxval` (using `unsigned integers`)  
  /// @param maxval 
  /// @return The number of bytes
  int required_bytes(uint64_t maxval);

  /// @brief Opens a file in read-only mode
  /// @param filename 
  /// @return The FILE*
  FILE *open_file_r(const char *filename);

  /// @brief Opens a file in write-only mode
  /// @param filename 
  /// @return The FILE*
  FILE *open_file_w(const char *filename);

  /// @brief A comparator for entries (mmio::io::Entry)
  /// @tparam IT Index type
  /// @tparam VT Value type
  /// @param a Entry 1
  /// @param b Entry 2
  /// @return a.row - b.row. If they are the same, a.col - b.col instead
  template<typename IT, typename VT>
  int compare_entries(const void *a, const void *b);

  /** BMTX **/

  /// @brief Detect if file is `bmtx` from file extension
  /// @param filename 
  /// @return `true` if `filename` ends with `.bmtx`, otherwise `false`
  bool mm_is_file_extension_bmtx(std::string filename);

  /// @brief Writes COO to a `bmtx` file (in binary format)
  /// @tparam IT Index Data Type
  /// @tparam VT Value Data Type
  /// @param f File to write to
  /// @param coo 
  /// @param meta Information to construct the header
  /// @return `0` if ok, otherwise one of `DMMIO_ERR_*`
  template<typename IT, typename VT>
  int mm_write_binary_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta);

} // namespace mmio::io

#endif // __MMIO_IO_H__
