#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <string>
#include <unistd.h>

#include "../../include/mmio/mmio.h"
#include "../../include/mmio/io.h"

using Matrix_Metadata = mmio::Matrix_Metadata;
template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
template<typename IT, typename VT> using DENSE = mmio::DENSE<IT, VT>;
template<typename IT, typename VT> using COO   = mmio::COO<IT, VT>;
template<typename IT, typename VT> using CSR   = mmio::CSR<IT, VT>;
template<typename IT, typename VT> using CSC   = mmio::CSC<IT, VT>;
template<typename IT, typename VT> using CSX   = mmio::CSX<IT, VT>;

#define MMIO_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template size_t mmio::CSX_buf_size<IT, VT>(IT nrows, IT ncols, IT nnz, MajorDim majordim); \
  template size_t mmio::CSX_buf_size(CSX<IT, VT> * csx); \
  template void mmio::CSX_get_ptrs(IT nrows, IT ncols, IT nnz, char * buf, IT ** ptr_vec, IT ** idx_vec, VT ** val_vec); \
  template CSX<IT, VT>* mmio::CSX_create(IT nrows, IT ncols, IT nnz, bool alloc_val, MajorDim majordim); \
  template CSX<IT, VT>* mmio::CSX_create(IT nrows, IT ncols, IT nnz, MajorDim majordim, IT *ptr_vec, IT *idx_vec, VT *val_vec); \
  template CSX<IT, VT>* mmio::CSX_create_contig(IT nrows, IT ncols, IT nnz, bool alloc_val, MajorDim majordim); \
  template COO<IT, VT>* mmio::COO_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template CSR<IT, VT>* mmio::CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template CSC<IT, VT>* mmio::CSC_create(IT nrows, IT ncols, IT nnz, bool alloc_val); \
  template DENSE<IT,VT>*mmio::DENSE_create(IT n, IT m); \
  template void mmio::COO_destroy(COO<IT, VT> **coo); \
  template void mmio::CSR_destroy(CSR<IT, VT> **csr); \
  template void mmio::CSC_destroy(CSC<IT, VT> **csc); \
  template void mmio::CSX_destroy(CSX<IT, VT> **csx); \
  template COO<IT, VT>* mmio::COO_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template CSR<IT, VT>* mmio::CSR_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template COO<IT, VT>* mmio::COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template CSR<IT, VT>* mmio::CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta); \
  template int mmio::COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta); \
  template int mmio::COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta); \
  template CSX<IT, VT>*  mmio::CSR2CSX(CSR<IT, VT> * csr);                \
  template CSX<IT, VT>*  mmio::CSC2CSX(CSC<IT, VT> * csc);                \
  template COO<IT, VT>*  mmio::CSX2COO(CSX<IT, VT> * csx);                       \
  template DENSE<IT,VT>* mmio::coo2dense(COO<IT, VT>* coo);               \
  template DENSE<IT,VT>* mmio::csr2dense(const CSR<IT,VT>* csr);          \
  template DENSE<IT,VT>* mmio::matmul(DENSE<IT,VT>* A, DENSE<IT,VT>* B);

namespace mmio {
  
  /********************* COO ***************************/
  template<typename IT, typename VT>
  COO<IT, VT>* COO_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
    COO<IT, VT> *coo = (COO<IT, VT> *)malloc(sizeof(COO<IT, VT>));
    coo->nrows = nrows;
    coo->ncols = ncols;
    coo->nnz = nnz;
    coo->row = (IT *)malloc(nnz * sizeof(IT));
    coo->col = (IT *)malloc(nnz * sizeof(IT));
    coo->val = NULL;
    if (alloc_val) {
      coo->val = (VT *)malloc(nnz * sizeof(VT));
    }
    return coo;
  }

  template<typename IT, typename VT>
  void COO_destroy(COO<IT, VT> **coo) {
    if (coo != NULL && *coo != NULL) {
      if ((*coo)->row != NULL) {
        free((*coo)->row);
        (*coo)->row = NULL;
      }
      if ((*coo)->col != NULL) {
        free((*coo)->col);
        (*coo)->col = NULL;
      }
      if ((*coo)->val != NULL) {
        free((*coo)->val);
        (*coo)->val = NULL;
      }
      free(*coo);
      *coo = NULL;
    }
  }

  template<typename IT, typename VT>
  COO<IT, VT>* COO_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
    return mmio::COO_read_f<IT, VT>(mmio::io::open_file_r(filename), mmio::io::mm_is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta);
  }

  template<typename IT, typename VT>
  COO<IT, VT>* COO_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
    IT nrows, ncols, nnz;
    MM_typecode matcode;
    Entry<IT, VT> *entries = mmio::io::mm_parse_file<IT, VT>(f, nrows, ncols, nnz, &matcode, is_bmtx, meta);
    if (entries == NULL) return NULL;
    
    COO<IT, VT> *coo = COO_create<IT, VT>(nrows, ncols, nnz, expl_val_for_bin_mtx || !mm_is_pattern(matcode));
    mmio::io::Entries_to_COO<IT, VT>(entries, coo);
    
    free(entries);

    return coo;
  }

  template<typename IT, typename VT>
  int COO_write(COO<IT, VT>* coo, const char *filename, bool write_as_binary, Matrix_Metadata* meta) {
    return mmio::COO_write_f(coo, mmio::io::open_file_w(filename), write_as_binary, meta);
  }

  template<typename IT, typename VT>
  int COO_write_f(COO<IT, VT>* coo, FILE *f, bool write_as_binary, Matrix_Metadata* meta) {
    if (meta->mm_header.empty()) {
      meta->mm_header = "%%MatrixMarket matrix coordinate";

      switch (meta->value_type) {
        case mmio::ValueType::Real:    { meta->mm_header += std::string(MM_REAL_STR);    break; }
        case mmio::ValueType::Integer: { meta->mm_header += std::string(MM_INT_STR);     break; }
        case mmio::ValueType::Pattern: { meta->mm_header += std::string(MM_PATTERN_STR); break; }
        default:                  { fprintf(stderr, "BUG: ValueType not recognized\n"); return 100; }
      }

      meta->mm_header += (meta->is_symmetric && false) ? "symmetric" : "general";
    }

    std::string g("general");
    std::string s("symmetric");
    size_t pos = meta->mm_header.find("symmetric");
    if (pos != std::string::npos)
    {
        meta->mm_header.replace(pos, s.length(), g);
    }

    printf("%s\n", meta->mm_header.c_str());
    fflush(stdout);


    
    return write_as_binary ? mmio::io::mm_write_binary_matrix_market(f, coo, meta) : mmio::io::mm_write_matrix_market(f, coo, meta);
  }


  /********************* CSR ***************************/
  template<typename IT, typename VT>
  CSR<IT, VT>* CSR_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
    CSR<IT, VT> *csr = (CSR<IT, VT> *)malloc(sizeof(CSR<IT, VT>));
    csr->nrows = nrows;
    csr->ncols = ncols;
    csr->nnz = nnz;
    csr->row_ptr = (IT *)malloc((nrows + 1) * sizeof(IT));
    csr->col_idx = (IT *)malloc(nnz * sizeof(IT));
    csr->val = NULL;
    if (alloc_val) {
      csr->val = (VT *)malloc(nnz * sizeof(VT));
    }
    return csr;
  }

  template<typename IT, typename VT>
  void CSR_destroy(CSR<IT, VT> **csr) {
    if (*csr != NULL) {
      if ((*csr)->row_ptr != NULL) {
        free((*csr)->row_ptr);
        (*csr)->row_ptr = NULL;
      }
      if ((*csr)->col_idx != NULL) {
        free((*csr)->col_idx);
        (*csr)->col_idx = NULL;
      }
      if ((*csr)->val != NULL) {
        free((*csr)->val);
        (*csr)->val = NULL;
      }
      free(*csr);
      *csr = NULL;
    }
  }

  template<typename IT, typename VT>
  CSR<IT, VT>* CSR_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
    return mmio::CSR_read_f<IT, VT>(mmio::io::open_file_r(filename), mmio::io::mm_is_file_extension_bmtx(std::string(filename)), expl_val_for_bin_mtx, meta);
  }

  template<typename IT, typename VT>
  CSR<IT, VT>* CSR_read_f(FILE *f, bool is_bmtx, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
    IT nrows, ncols, nnz;
    MM_typecode matcode;
    Entry<IT, VT> *entries = mmio::io::mm_parse_file<IT, VT>(f, nrows, ncols, nnz, &matcode, is_bmtx, meta);
    if (entries == NULL) return NULL;

    CSR<IT, VT> *csr = CSR_create<IT, VT>(nrows, ncols, nnz, expl_val_for_bin_mtx || !mm_is_pattern(matcode));
    mmio::io::Entries_to_CSR<IT, VT>(entries, csr);

    free(entries);

    return csr;
  }

  /********************* CSC ***************************/
  template<typename IT, typename VT>
  CSC<IT, VT>* CSC_create(IT nrows, IT ncols, IT nnz, bool alloc_val) {
    CSC<IT, VT> *csc = (CSC<IT, VT> *)malloc(sizeof(CSC<IT, VT>));
    csc->nrows = nrows;
    csc->ncols = ncols;
    csc->nnz = nnz;
    csc->col_ptr = (IT *)malloc((ncols + 1) * sizeof(IT));
    csc->row_idx = (IT *)malloc(nnz * sizeof(IT));
    csc->val = NULL;
    if (alloc_val) {
      csc->val = (VT *)malloc(nnz * sizeof(VT));
    }
    return csc;
  }

  template<typename IT, typename VT>
  void CSC_destroy(CSC<IT, VT> **csc) {
    if (*csc != NULL) {
      if ((*csc)->col_ptr != NULL) {
        free((*csc)->col_ptr);
        (*csc)->col_ptr = NULL;
      }
      if ((*csc)->row_idx != NULL) {
        free((*csc)->row_idx);
        (*csc)->row_idx = NULL;
      }
      if ((*csc)->val != NULL) {
        free((*csc)->val);
        (*csc)->val = NULL;
      }
      free(*csc);
      *csc = NULL;
    }
  }

  // TODO
  // template<typename IT, typename VT>
  // CSC<IT, VT>* CSC_read(const char *filename, bool expl_val_for_bin_mtx, Matrix_Metadata* meta) {
  //   ....
  // }

  /********************* CSX ***************************/
  template<typename IT, typename VT>
  size_t CSX_buf_size(IT nrows, IT ncols, IT nnz, MajorDim majordim) 
  {
      assert(sizeof(IT) == sizeof(VT));
      size_t result = 0;
      result += sizeof(VT) * nnz;
      result += sizeof(IT) * nnz;
      if (majordim == MajorDim::ROWS)
      {
          result += sizeof(IT) * (nrows + 1);
      }
      else
      {
          result += sizeof(IT) * (ncols + 1);
      }
      return result;
  }


  template<typename IT, typename VT>
  size_t CSX_buf_size(CSX<IT, VT> * csx)
  {
      return CSX_buf_size<IT, VT>(csx->nrows, csx->ncols, csx->nnz, csx->majordim);
  }


  template<typename IT, typename VT>
  void CSX_get_ptrs(IT nrows, IT ncols, IT nnz, char * buf,
                    IT ** ptr_vec, IT ** idx_vec, VT ** val_vec)
  {
      assert(sizeof(IT) == sizeof(VT));
      size_t offset = 0;
      *val_vec = (VT*)buf;
      offset += sizeof(VT) * nnz;
      *idx_vec = (IT*)(buf + offset);
      offset += sizeof(IT) * nnz;
      *ptr_vec = (IT*)(buf + offset);
  }


  template<typename IT, typename VT>
  CSX<IT, VT>* CSX_create(IT nrows, IT ncols, IT nnz, bool alloc_val, MajorDim majordim) {
    CSX<IT, VT> *csx = (CSX<IT, VT> *)malloc(sizeof(CSX<IT, VT>));
    csx->majordim = majordim;
    csx->nrows    = nrows;
    csx->ncols    = ncols;
    csx->nnz      = nnz;
    csx->idx_vec  = (IT *)malloc(nnz * sizeof(IT));
    csx->contig = false;
    csx->buf = nullptr;
    csx->buf_size = 0;

    if (majordim == MajorDim::ROWS)
      csx->ptr_vec = (IT *)malloc((nrows + 1) * sizeof(IT));
    else
      csx->ptr_vec = (IT *)malloc((ncols + 1) * sizeof(IT));

    csx->val = NULL;
    if (alloc_val) {
      csx->val = (VT *)malloc(nnz * sizeof(VT));
    }

    return csx;
  }


  template<typename IT, typename VT>
  CSX<IT, VT>* CSX_create_contig(IT nrows, IT ncols, IT nnz, bool alloc_val, MajorDim majordim) {

    CSX<IT, VT> *csx = (CSX<IT, VT> *)malloc(sizeof(CSX<IT, VT>));
    csx->majordim = majordim;
    csx->nrows    = nrows;
    csx->ncols    = ncols;
    csx->nnz      = nnz;
    csx->contig   = true;

    csx->buf_size = CSX_buf_size<IT, VT>(nrows, ncols, nnz, majordim);
    csx->buf = (char *)malloc(csx->buf_size);

    CSX_get_ptrs(nrows, ncols, nnz, csx->buf, 
                 &(csx->ptr_vec), &(csx->idx_vec), &(csx->val));

    return csx;
  }


  template<typename IT, typename VT>
  CSX<IT, VT>* CSX_create(IT nrows, IT ncols, IT nnz, MajorDim majordim, IT *ptr_vec, IT *idx_vec, VT *val_vec) {
    mmio::CSX<IT, VT> *csx = (mmio::CSX<IT, VT>*)malloc(sizeof(mmio::CSX<IT, VT>));
    csx->majordim = majordim;
    csx->nrows    = nrows;
    csx->ncols    = ncols;
    csx->nnz      = nnz;
    csx->contig = false;
    csx->buf = nullptr;
    csx->buf_size = 0;

    csx->ptr_vec = ptr_vec;
    csx->idx_vec = idx_vec;
    csx->val     = val_vec;

    return(csx);
  }


  template<typename IT, typename VT>
  CSX<IT, VT>* CSR2CSX(CSR<IT, VT> * csr) {
    bool alloc_val = ((csr->val)!=NULL);
    IT nrows = csr->nrows, ncols = csr->ncols, nnz = csr->nnz;

    CSX<IT, VT>* csx = CSX_create<IT,VT>(nrows, ncols, nnz, alloc_val, MajorDim::ROWS);
    csx->ptr_vec     = csr->row_ptr;
    csx->idx_vec     = csr->col_idx;
    if (alloc_val)
      csx->val       = csr->val;

    return(csx);
  }

  template<typename IT, typename VT>
  CSX<IT, VT>* CSC2CSX(CSC<IT, VT> * csc) {
    bool alloc_val = ((csc->val)!=NULL);
    IT nrows = csc->nrows, ncols = csc->ncols, nnz = csc->nnz;

    CSX<IT, VT>* csx = CSX_create<IT,VT>(nrows, ncols, nnz, alloc_val, MajorDim::COLS);
    csx->ptr_vec     = csc->col_ptr;
    csx->idx_vec     = csc->row_idx;
    if (alloc_val)
      csx->val       = csc->val;

    return(csx);
  }

  template<typename IT, typename VT>
  COO<IT, VT>* CSX2COO(CSX<IT, VT> * csx) {
    IT n = csx->nrows, m = csx->ncols, nnz = csx->nnz;
    COO<IT, VT>* coo = COO_create<IT,VT>(n, m, nnz, true);

    MajorDim layout = csx->majordim;
    IT ptrvecsize = (layout == MajorDim::ROWS) ? n : m;
    for (IT i=0; i<ptrvecsize; i++) {
        for (int j=0; j<(csx->ptr_vec[i+1] - csx->ptr_vec[i]); j++) {
            IT dest_idx = csx->ptr_vec[i]+j;
            IT row = (layout == MajorDim::COLS) ? (csx->idx_vec[dest_idx]) : (i) ;
            IT col = (layout == MajorDim::ROWS) ? (csx->idx_vec[dest_idx]) : (i) ;
            VT val = csx->val[dest_idx];

            coo->row[dest_idx] = row;
            coo->col[dest_idx] = col;
            coo->val[dest_idx] = val;
        }
    }

    return(coo);
  }

  template<typename IT, typename VT>
  void CSX_destroy(CSX<IT, VT> **csx) {
    if (*csx != NULL) {

      if ((*csx)->contig) {
        free((*csx)->buf);
        (*csx)->buf = NULL;
      } else {
        if ((*csx)->ptr_vec != NULL) {
          free((*csx)->ptr_vec);
          (*csx)->ptr_vec = NULL;
        }
        if ((*csx)->idx_vec != NULL) {
          free((*csx)->idx_vec);
          (*csx)->idx_vec = NULL;
        }
        if ((*csx)->val != NULL) {
          free((*csx)->val);
          (*csx)->val = NULL;
        }
      }
      free(*csx);
      *csx = NULL;
    }
  }

  /******************** DENSE **************************/
  template<typename IT, typename VT>
  DENSE<IT,VT>* DENSE_create(IT n, IT m) {
    DENSE<IT,VT> *M = (DENSE<IT,VT>*)malloc(sizeof(DENSE<IT,VT>));

    M->nrows = n;
    M->ncols = m;
    M->val   = (VT*)malloc(sizeof(VT)*n*m);

    for(size_t i=0; i<(n*m); i++) M->val[i] = 0;
    return(M);
  }

  template<typename IT, typename VT>
  DENSE<IT,VT>* coo2dense(COO<IT, VT>* coo) {

    DENSE<IT,VT> *M = DENSE_create<IT,VT>(coo->nrows, coo->ncols);
    for(size_t i=0; i<(coo->nnz); i++) M->val[(coo->row[i])*M->ncols + (coo->col[i])] = coo->val[i];

    return(M);
  }

  template<typename IT, typename VT>
  DENSE<IT,VT>* csr2dense(const CSR<IT,VT>* csr) {
      DENSE<IT,VT> *dense = DENSE_create<IT,VT>(csr->nrows, csr->ncols);

      // Fill in nonzeros
      for (IT i = 0; i < csr->nrows; i++) {
          for (IT idx = csr->row_ptr[i]; idx < csr->row_ptr[i+1]; idx++) {
              IT j = csr->col_idx[idx];
              dense->val[static_cast<size_t>(i) * csr->ncols + j] = csr->val[idx];
          }
      }

      return(dense);
  }


  template<typename IT, typename VT>
  DENSE<IT,VT>* matmul(DENSE<IT,VT>* A_struct, DENSE<IT,VT>* B_struct) {

    if (A_struct->ncols != B_struct->nrows) {
      fprintf(stderr, "Error: matmult on uncompatible matrices [(%dx%d)*(%dx%d)]\n", A_struct->ncols, A_struct->nrows, B_struct->ncols, B_struct->nrows);
      return(nullptr);
    }
    IT n = A_struct->nrows, m = B_struct->ncols, k = A_struct->ncols;
    VT *A = A_struct->val, *B = B_struct->val;

    DENSE<IT,VT> *C = DENSE_create<IT,VT>(n, m);
    for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
              for (int h = 0; h < n; h++) {
                  C->val[i*m + j] += A[i*k + h] * B[h*m + j];
              }
          }
      }
      return(C);
  }

} // namespace mmio 


MMIO_EXPLICIT_TEMPLATE_INST(uint32_t, float)
MMIO_EXPLICIT_TEMPLATE_INST(uint32_t, double)
MMIO_EXPLICIT_TEMPLATE_INST(uint64_t, float)
MMIO_EXPLICIT_TEMPLATE_INST(uint64_t, double)
MMIO_EXPLICIT_TEMPLATE_INST(int, float)
MMIO_EXPLICIT_TEMPLATE_INST(int, double)
MMIO_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
MMIO_EXPLICIT_TEMPLATE_INST(int64_t, float)
MMIO_EXPLICIT_TEMPLATE_INST(int64_t, double)
