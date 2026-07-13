#include <stdio.h>
#include <string>
#include <vector>

#include "../../include/mmio/mmio.h"
#include "../../include/mmio/utils.h"

template<typename IT, typename VT> using COO = mmio::COO<IT, VT>;
template<typename IT, typename VT> using CSR = mmio::CSR<IT, VT>;
template<typename IT, typename VT> using CSX = mmio::CSX<IT, VT>;

#define MMIO_UTILS_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template void mmio::utils::CSR_print(CSR<IT, VT> *csr, std::string header, FILE* fp); \
  template void mmio::utils::CSR_print_as_dense<IT, VT>(CSR<IT, VT> *csr, std::string header, FILE* fp); \
  template void mmio::utils::COO_print(COO<IT, VT> *coo, std::string header, FILE* fp); \
  template void mmio::utils::COO_print_as_dense(COO<IT, VT> *coo, std::string header, FILE* fp); \
  template void mmio::utils::CSX_print_as_dense(CSX<IT, VT> *csx, std::string header, FILE* fp);


namespace mmio::utils {

  /********************* Common Utilities ***************************/

  // Format index type depending on IT
  template<typename IT>
  std::string index_fmt() {
    if constexpr (std::is_same<IT, uint64_t>::value)
      return "%3lu";
    else
      return "%3u";
  }

  // Print standard header
  template<typename IT>
  void print_header(FILE* fp, std::string header, IT nrows, IT ncols, IT nnz) {
    if (!header.empty()) {
      fprintf(fp, "%s -- ", header.c_str());
    }
    std::string I_FMT = index_fmt<IT>();
    char fmt[100];
    snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
    fprintf(fp, fmt, nrows, ncols, nnz);
  }

  // Print column indices header row
  template<typename IT>
  void print_col_header(FILE* fp, IT ncols) {
    fprintf(fp, "      "); // space for row label
    for (IT c = 0; c < ncols; ++c) {
      fprintf(fp, "[%4u] ", (unsigned)c);
    }
    fprintf(fp, "\n");
  }

  // Print value with unified format (float or int)
  template<typename VT>
  void print_val(FILE* fp, VT val) {
    if (val == static_cast<VT>(0)) {
      fprintf(fp, "    -  "); // placeholder for zero
    } else {
      if constexpr (std::is_floating_point<VT>::value) {
        fprintf(fp, "%6.1f ", val);
      } else {
        fprintf(fp, "%6d ", (int)val);
      }
    }
  }


  /********************* API Functions ***************************/

  template<typename IT, typename VT>
  void CSR_print(CSR<IT, VT> *csr, std::string header, FILE* fp) {
    if (header != "") {
      fprintf(fp, "%s -- ", header.c_str());
    }
    std::string I_FMT = "%3u";
    if constexpr (std::is_same<IT, uint64_t>::value)
      I_FMT = "%3lu";

    char fmt[100];
    snprintf(fmt, 100, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
    fprintf(fp, fmt, csr->nrows, csr->ncols, csr->nnz);

    snprintf(fmt, 100, "%s ", I_FMT.c_str());
    fprintf(fp, "idx   : ");
    for (IT i = 0; i < csr->nnz; ++i) fprintf(fp, fmt, i);
    fprintf(fp, "\nrowptr: ");
    for (IT i = 0; i <= csr->nrows; ++i) fprintf(fp, fmt, csr->row_ptr[i]);
    fprintf(fp, "\ncolidx: ");
    for (IT i = 0; i < csr->nnz; ++i) fprintf(fp, fmt, csr->col_idx[i]);
    if (csr->val != NULL) {
      fprintf(fp, "\nval:    ");
      for (IT i = 0; i < csr->nnz; ++i) {
        fprintf(fp, "%.1f ", csr->val[i]); // TODO handle different VT
      }
    }
    fprintf(fp, "\n");
  }

  template<typename IT, typename VT>
  void COO_print(COO<IT, VT> *coo, std::string header, FILE* fp) {
    if (header != "") {
      fprintf(fp, "%s -- ", header.c_str());
    }
    std::string I_FMT = "%4u";
    if constexpr (std::is_same<IT, uint64_t>::value)
      I_FMT = "%4lu";

    char fmt[200];
    snprintf(fmt, 200, "Matrix %s x %s (%s non-zeros)\n", I_FMT.c_str(), I_FMT.c_str(), I_FMT.c_str());
    fprintf(fp, fmt, coo->nrows, coo->ncols, coo->nnz);

    snprintf(fmt, 200, "%s ", I_FMT.c_str());
    fprintf(fp, "idx: ");
    for (IT i = 0; i < coo->nnz; ++i) fprintf(fp, fmt, i);
    fprintf(fp, "\nrow: ");
    for (IT i = 0; i < coo->nnz; ++i) fprintf(fp, fmt, coo->row[i]);
    fprintf(fp, "\ncol: ");
    for (IT i = 0; i < coo->nnz; ++i) fprintf(fp, fmt, coo->col[i]);
    if (coo->val != NULL) {
      fprintf(fp, "\nval: ");
      for (IT i = 0; i < coo->nnz; ++i) {
        fprintf(fp, "%4.1f ", coo->val[i]); // TODO handle different VT
      }
    }
    fprintf(fp, "\n");
  }

  template<typename IT, typename VT>
  void CSR_print_as_dense(CSR<IT, VT> *csr, std::string header, FILE* fp) {
    // Build dense matrix
    std::vector<std::vector<VT>> dense(csr->nrows, std::vector<VT>(csr->ncols, static_cast<VT>(0)));
    for (IT row = 0; row < csr->nrows; ++row) {
      for (IT idx = csr->row_ptr[row]; idx < csr->row_ptr[row + 1]; ++idx) {
        IT col = csr->col_idx[idx];
        dense[row][col] = csr->val ? csr->val[idx] : static_cast<VT>(1);
      }
    }

    print_header(fp, header, csr->nrows, csr->ncols, csr->nnz);
    print_col_header(fp, csr->ncols);

    for (IT row = 0; row < csr->nrows; ++row) {
      fprintf(fp, "[%3u] ", (unsigned)row);
      for (IT col = 0; col < csr->ncols; ++col) {
        print_val(fp, dense[row][col]);
      }
      fprintf(fp, "\n");
    }
  }

  // ===================== COO Dense Print (using same style) =====================
  template<typename IT, typename VT>
  void COO_print_as_dense(COO<IT, VT> *coo, std::string header, FILE* fp) {
    std::vector<std::vector<VT>> dense(coo->nrows, std::vector<VT>(coo->ncols, static_cast<VT>(0)));
    for (IT k = 0; k < coo->nnz; ++k) {
      IT r = coo->row[k];
      IT c = coo->col[k];
      dense[r][c] = coo->val ? coo->val[k] : static_cast<VT>(1);
    }

    print_header(fp, header, coo->nrows, coo->ncols, coo->nnz);
    print_col_header(fp, coo->ncols);

    for (IT row = 0; row < coo->nrows; ++row) {
      fprintf(fp, "[%3u] ", (unsigned)row);
      for (IT col = 0; col < coo->ncols; ++col) {
        print_val(fp, dense[row][col]);
      }
      fprintf(fp, "\n");
    }
  }

  template<typename IT, typename VT>
  void CSX_print_as_dense(CSX<IT, VT> *csx, std::string header, FILE* fp) {
    COO<IT,VT> *coo = CSX2COO(csx);
    COO_print_as_dense(coo, header, fp);
  }

} // namespace mmio::utils

MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint32_t, double)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, double)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(int, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(int, double)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(int64_t, float)
MMIO_UTILS_EXPLICIT_TEMPLATE_INST(int64_t, double)
