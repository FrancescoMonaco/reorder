#ifndef __MMIO_H__
#define __MMIO_H__

#include <stdint.h>
#include <stdio.h>
#include <string>

#include "macros.h"

namespace mmio {

  /********************* Enums and Data Structures ***************************/

  enum ValueType {
    Real,
    Integer,
    Pattern
  };

  /// @brief IMPORTANT: initialize with "new"
  struct Matrix_Metadata {
    // Data type of matrix values
    ValueType value_type;
    // Size in bytes of each matrix value
    uint8_t value_bytes;
    // Size in bytes of each matrix index
    uint8_t index_bytes;
    // First line of the header
    std::string mm_header;
    // Following header lines
    std::string mm_header_body;
    bool is_symmetric;
    bool is_pattern;
  };

  template<typename IT, typename VT>
  struct COO {
    IT nrows;
    IT ncols;
    IT nnz;
    IT *row;
    IT *col;
    VT *val;
  };

  template<typename IT, typename VT>
  struct CSR {
    IT 	nrows;
    IT 	ncols;
    IT 	nnz;
    IT 	*row_ptr;
    IT 	*col_idx;
    VT 	*val;
  };

  template<typename IT, typename VT>
  struct CSC {
    IT 	nrows;
    IT 	ncols;
    IT 	nnz;
    IT 	*col_ptr;
    IT 	*row_idx;
    VT 	*val;
  };

  enum MajorDim {
    ROWS,  // i.e. CSR format
    COLS   // i.e. CSC format
  };

  template<typename IT, typename VT>
  struct CSX {
    MajorDim majordim;
    bool contig;
    IT 	nrows;
    IT 	ncols;
    IT 	nnz;
    size_t buf_size;
    char * buf;
    IT 	*ptr_vec;
    IT 	*idx_vec;
    VT 	*val;
  };

  template<typename IT, typename VT>
  struct DENSE {
    IT nrows;
    IT ncols;
    VT *val;

    // Equality operator
    bool operator==(const DENSE &other) const {
        if (nrows != other.nrows || ncols != other.ncols) {
            return false;
        }
        size_t size = static_cast<size_t>(nrows) * static_cast<size_t>(ncols);
        for (size_t i = 0; i < size; i++) {
            if (val[i] != other.val[i]) {
                return false;
            }
        }
        return true;
    }

    // Destructor
    ~DENSE() {
        delete[] val;
    }
  };

  /********************* Data Structures Functions ***************************/

  /** COO **/
  template<typename IT, typename VT>
  COO<IT, VT>* COO_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

  template<typename IT, typename VT>
  void COO_destroy(COO<IT, VT>** coo);

  template<typename IT, typename VT>
  COO<IT, VT>* COO_read(const char *filename, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

  template<typename IT, typename VT>
  COO<IT, VT>* COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

  template<typename IT, typename VT>
  int COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta);

  template<typename IT, typename VT>
  int COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta);

  /** CSR **/
  template<typename IT, typename VT>
  CSR<IT, VT>* CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

  template<typename IT, typename VT>
  void CSR_destroy(CSR<IT, VT>** csr);

  template<typename IT, typename VT>
  CSR<IT, VT>* CSR_read(const char *filename, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

  template<typename IT, typename VT>
  CSR<IT, VT>* CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx=false, Matrix_Metadata* meta=NULL);

  /** CSC **/
  template<typename IT, typename VT>
  CSC<IT, VT>* CSC_create(IT nrows, IT ncols, IT nnz, bool alloc_val);

  template<typename IT, typename VT>
  void CSC_destroy(CSC<IT, VT>** csc);

  // TODO

  /** CSX **/
  template<typename IT, typename VT>
  size_t CSX_buf_size(IT nrows, IT ncols, IT nnz, MajorDim majordim);

  template<typename IT, typename VT>
  size_t CSX_buf_size(CSX<IT, VT> * csx);

  template<typename IT, typename VT>
  void CSX_get_ptrs(IT nrows, IT ncols, IT nnz, char * buf,
                    IT ** ptr_vec, IT ** idx_vec, VT ** val_vec);

  template<typename IT, typename VT>
  CSX<IT, VT>* CSX_create(IT nrows, IT ncols, IT nnz, bool alloc_val, MajorDim majordim);

  template<typename IT, typename VT>
  CSX<IT, VT>* CSX_create_contig(IT nrows, IT ncols, IT nnz, bool alloc_val, MajorDim majordim);

  template<typename IT, typename VT>
  CSX<IT, VT>* CSX_create(IT nrows, IT ncols, IT nnz, MajorDim majordim, IT *ptr_vec, IT *idx_vec, VT *val_vec);

  template<typename IT, typename VT>
  CSX<IT, VT>* CSR2CSX(CSR<IT, VT> * csr);

  template<typename IT, typename VT>
  CSX<IT, VT>* CSC2CSX(CSC<IT, VT> * csc);

  template<typename IT, typename VT>
  COO<IT, VT>* CSX2COO(CSX<IT, VT> * csx);

  template<typename IT, typename VT>
  void CSX_destroy(CSX<IT, VT>** csx);

  /** DENSE **/
  template<typename IT, typename VT>
  DENSE<IT,VT>* DENSE_create(IT n, IT m);

  template<typename IT, typename VT>
  DENSE<IT,VT>* coo2dense(COO<IT, VT>* coo);

  template<typename IT, typename VT>
  DENSE<IT,VT>* csr2dense(const CSR<IT,VT>* csr);

  template<typename IT, typename VT>
  DENSE<IT,VT>* matmul(DENSE<IT,VT>* A, DENSE<IT,VT>* B);

} // namespace mmio

#endif // __MMIO_H__
