#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <unistd.h>
#include <cstdint>

#include <ccutils/colors.h>
#include <ccutils/macros.h>

#include "../../include/mmio/macros.h"
#include "../../include/mmio/mmio.h"
#include "../../include/mmio/io.h"

using Matrix_Metadata = mmio::Matrix_Metadata;
template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
template<typename IT, typename VT> using COO = mmio::COO<IT, VT>;
template<typename IT, typename VT> using CSR = mmio::CSR<IT, VT>;

#define MMIO_IO_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template void mmio::io::Entries_to_CSR(Entry<IT, VT> *entries, CSR<IT, VT> *csr); \
  template void mmio::io::Entries_to_COO(Entry<IT, VT> *entries, COO<IT, VT> *coo); \
  template int mmio::io::mm_read_mtx_crd_data(FILE *f, int nnz, Entry<IT, VT> *entries, MM_typecode *matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t value_bytes); \
  template int mmio::io::mm_write_binary_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta); \
  template Entry<IT, VT>* mmio::io::mm_parse_file(FILE *f, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta); \
  template int mmio::io::mm_write_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta); \
  template IT mmio::io::mm_count_duplicates(Entry<IT, VT>* entries, IT nentries, IT nnz_upperbound, Matrix_Metadata* meta);
  // template int mmio::io::compare_entries(const void *a, const void *b);
  // template int mmio::io::mm_parse_header(FILE *f, IT &nrows, IT &ncols, IT &nentries, IT &nnz_upperbound, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta); \
  
namespace mmio::io {
  
  FILE *open_file_r(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
      fprintf(stderr, "Could not open file [%s] (read).\n", filename);
      return NULL;
    }
    return f;
  }

  FILE *open_file_w(const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
      fprintf(stderr, "Could not open file [%s] (write).\n", filename);
      return NULL;
    }
    return f;
  }

  template<typename IT, typename VT>
  int compare_entries(const void *a, const void *b) {
    Entry<IT, VT> *ea = (Entry<IT, VT> *)a;
    Entry<IT, VT> *eb = (Entry<IT, VT> *)b;
    if (ea->row != eb->row)
      return ea->row - eb->row;
    return ea->col - eb->col;
  }

  template<typename IT, typename VT>
  void Entries_to_CSR(Entry<IT, VT> *entries, CSR<IT, VT> *csr) {
    qsort(entries, csr->nnz, sizeof(Entry<IT, VT>), mmio::io::compare_entries<IT, VT>);

    for (IT v = 0, i = 0; v < csr->nrows; v++) {
      csr->row_ptr[v] = i;
      while (i < csr->nnz && entries[i].row == v) {
        csr->col_idx[i] = entries[i].col;
        if (csr->val != NULL) {
          csr->val[i] = entries[i].val;
        }
        ++i;
      }
    }
    csr->row_ptr[csr->nrows] = csr->nnz;
  }

  template<typename IT, typename VT>
  void Entries_to_COO(Entry<IT, VT> *entries, COO<IT, VT> *coo) {
    for (IT i = 0; i < coo->nnz; ++i) {
      coo->row[i] = entries[i].row;
      coo->col[i] = entries[i].col;
      if (coo->val != NULL) coo->val[i] = entries[i].val;
    }
  }

  /**
   * Matrix Market parsing utilities.
   * See http://math.nist.gov/MatrixMarket for details.
   */ 
  int mm_read_banner(FILE *f, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta) {
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    uint8_t idx_bytes, value_bytes;
    char *p;

    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MMIO_ERR_PREMATURE_EOF;

    if (meta) {
      meta->mm_header = std::string(line);
      if (!meta->mm_header.empty() && meta->mm_header.back() == '\n') {
        meta->mm_header.pop_back();
      }
    }

    if (is_bmtx) {
      if (sscanf(line, "%s %s %s %s %s %hhu %hhu", banner, mtx, crd, data_type, storage_scheme, &idx_bytes, &value_bytes) != 7)
        return MMIO_ERR_PREMATURE_EOF;
      mm_set_idx_bytes(matcode, idx_bytes);
      mm_set_val_bytes(matcode, value_bytes);    
    } else {
      if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5)
        return MMIO_ERR_PREMATURE_EOF;
    }

    for (p = mtx; *p; ++p)            *p = (char)tolower((unsigned char)*p);
    for (p = crd; *p; ++p)            *p = (char)tolower((unsigned char)*p);
    for (p = data_type; *p; ++p)      *p = (char)tolower((unsigned char)*p);
    for (p = storage_scheme; *p; ++p) *p = (char)tolower((unsigned char)*p);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
      return MMIO_ERR_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
      return MMIO_ERR_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);

    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */

    if (strcmp(crd, MM_SPARSE_STR) == 0)
      mm_set_sparse(matcode);
    else if (strcmp(crd, MM_DENSE_STR) == 0)
      mm_set_dense(matcode);
    else
      return MMIO_ERR_UNSUPPORTED_TYPE;

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
      mm_set_real(matcode);
    else if (strcmp(data_type, MM_COMPLEX_STR) == 0)
      mm_set_complex(matcode);
    else if (strcmp(data_type, MM_PATTERN_STR) == 0)
      mm_set_pattern(matcode);
    else if (strcmp(data_type, MM_INT_STR) == 0)
      mm_set_integer(matcode);
    else
      return MMIO_ERR_UNSUPPORTED_TYPE;

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
      mm_set_general(matcode);
    else if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
      mm_set_symmetric(matcode);
    else if (strcmp(storage_scheme, MM_HERM_STR) == 0)
      mm_set_hermitian(matcode);
    else if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
      mm_set_skew(matcode);
    else
      return MMIO_ERR_UNSUPPORTED_TYPE;

    // Read and store all header lines (starting with '%')
    if (meta) {
      while (true) {
        long pos = ftell(f);
        if (!fgets(line, sizeof(line), f)) break;
        if (line[0] != '%') { fseek(f, pos, SEEK_SET); break; }
        meta->mm_header_body += line;
      }
    }

    return 0;
  }

  int mm_read_mtx_crd_size(FILE *f, uint64_t *nrows, uint64_t *ncols, uint64_t *nnz) {
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *nrows = *ncols = *nnz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do {
      if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MMIO_ERR_PREMATURE_EOF;
    } while (line[0] == '%');

    /* line[] is either blank or has nrows,ncols, nnz */
    if (sscanf(line, "%lu %lu %lu", nrows, ncols, nnz) == 3)
      return 0;

    else
      do {
        num_items_read = fscanf(f, "%lu %lu %lu", nrows, ncols, nnz);
        if (num_items_read == EOF)
          return MMIO_ERR_PREMATURE_EOF;
      } while (num_items_read != 3);

    return 0;
  }

  // FIXME this is a draft
  // template<typename IT, typename VT>
  // int parse_ascii_entries(FILE *f, int nentries, Entry<IT, VT> *entries, MM_typecode matcode) {
  //   // Heuristically assume a line is ~50–80 chars
  //   const size_t buf_size = nentries * 64;
  //   char *buffer = (char *)malloc(buf_size);
  //   if (!buffer) {
  //     fprintf(stderr, "Failed to allocate %zu bytes for ASCII read buffer.\n", buf_size);
  //     return MMIO_ERR_COULD_NOT_READ_FILE;
  //   }

  //   size_t read = fread(buffer, 1, buf_size, f);
  //   if (read == 0) {
  //     free(buffer);
  //     return MMIO_ERR_PREMATURE_EOF;
  //   }

  //   char *ptr = buffer;
  //   char *end = buffer + read;
  //   int count = 0;
  //   bool is_pattern = mm_is_pattern(matcode);
  //   bool is_real_or_int = mm_is_real(matcode) || mm_is_integer(matcode);

  //   while (ptr < end && count < nentries) {
  //     // Skip whitespace
  //     while (ptr < end && std::isspace(*ptr)) ++ptr;
  //     if (ptr >= end) break;

  //     // Parse row
  //     IT row = static_cast<IT>(strtoull(ptr, &ptr, 10)) - 1;

  //     // Parse col
  //     while (ptr < end && std::isspace(*ptr)) ++ptr;
  //     IT col = static_cast<IT>(strtoull(ptr, &ptr, 10)) - 1;

  //     VT val = static_cast<VT>(1.0); // default for pattern

  //     if (is_real_or_int) {
  //       while (ptr < end && std::isspace(*ptr)) ++ptr;
  //       val = static_cast<VT>(strtod(ptr, &ptr));
  //     }

  //     entries[count].row = row;
  //     entries[count].col = col;
  //     entries[count].val = val;
  //     ++count;
  //   }

  //   free(buffer);

  //   return (count == nentries) ? 0 : MMIO_ERR_PREMATURE_EOF;
  // }

  template<typename IT, typename VT>
  int mm_read_mtx_crd_data(FILE *f, int nentries, Entry<IT, VT> *entries, MM_typecode *matcode, bool is_bmtx, uint8_t idx_bytes, uint8_t value_bytes) {
    bool is_pattern = mm_is_pattern((*matcode));
    
    size_t entry_size = 2 * idx_bytes + (is_pattern ? 0 : value_bytes);
    size_t total_size = nentries * entry_size;

    if (!is_bmtx) {
      // FIXME uncomment and test return parse_ascii_entries(f, nentries, entries, matcode);

      // Original ASCII Matrix Market parsing
      const char *I_FMT = std::is_same<IT, uint64_t>::value ? "%lu" : "%u";
      const char *V_FMT = std::is_same<VT, double>::value   ? "%lg" : "%g";
      char fmt[32];
      int i;

      if (mm_is_real((*matcode)) || mm_is_integer((*matcode))) {
        snprintf(fmt, 32, "%s %s %s", I_FMT, I_FMT, V_FMT);
        for (i = 0; i < nentries; i++) {
          if (fscanf(f, fmt, &entries[i].row, &entries[i].col, &entries[i].val) != 3)
            return MMIO_ERR_PREMATURE_EOF;
          --entries[i].row;
          --entries[i].col;
        }
      } else if (is_pattern) {
        snprintf(fmt, sizeof(fmt), "%s %s", I_FMT, I_FMT);
        for (i = 0; i < nentries; i++) {
          if (fscanf(f, fmt, &entries[i].row, &entries[i].col) != 2)
            return MMIO_ERR_PREMATURE_EOF;
          --entries[i].row;
          --entries[i].col;
          entries[i].val = static_cast<VT>(1.0);
        }
      } else return MMIO_ERR_UNSUPPORTED_TYPE;

      return 0;
    }

    // Binary BMTX parsing
    
    // Allocate buffer to read the entire data block
    uint8_t *buffer = (uint8_t *)malloc(total_size);
    if (!buffer) {
      fprintf(stderr, "Failed to allocate %zu bytes for input buffer.\n", total_size);
      return MMIO_ERR_COULD_NOT_READ_FILE;
    }

    size_t bytes_read = fread(buffer, 1, total_size, f);
    if (bytes_read != total_size) {
      fprintf(stderr, "Failed to read expected %zu bytes from file. Only read %zu bytes. \n", total_size, bytes_read);
      free(buffer);
      return MMIO_ERR_PREMATURE_EOF;
    }

    uint8_t *ptr = buffer;

    for (int i = 0; i < nentries; ++i) {
      uint64_t row = 0, col = 0;

      // Read row
      memcpy(&row, ptr, idx_bytes);
      entries[i].row = static_cast<IT>(row);
      ptr += idx_bytes;

      // Read col
      memcpy(&col, ptr, idx_bytes);
      entries[i].col = static_cast<IT>(col);
      ptr += idx_bytes;

      // Read val if present
      if (!is_pattern) {
        if (value_bytes == 4) {
          float val_f;
          memcpy(&val_f, ptr, sizeof(float));
          entries[i].val = static_cast<VT>(val_f);
        } else if (value_bytes == 8) {
          double val_d;
          memcpy(&val_d, ptr, sizeof(double));
          entries[i].val = static_cast<VT>(val_d);
        } else {
          free(buffer);
          return MMIO_ERR_UNSUPPORTED_TYPE;
        }
        ptr += value_bytes;
      } else {
        entries[i].val = static_cast<VT>(1.0);  // Default for pattern
      }
    }

    free(buffer);
    return 0;
  }


  int required_bytes(uint64_t maxval) {
    if (maxval <= UINT8_MAX)  return 1;
    if (maxval <= UINT16_MAX) return 2;
    if (maxval <= UINT32_MAX) return 4;
    return 8;
  }


  /**
   * Read functions
   */ 



  bool mm_is_file_extension_bmtx(std::string filename) {
    return filename.size() >= 5 && filename.compare(filename.size() - 5, 5, ".bmtx") == 0;
  }

  int write_matrix_market_header(FILE *f, Matrix_Metadata *meta, int index_bytes, uint64_t nrows, uint64_t ncols, uint64_t nentries) {
    if (!f) return MMIO_ERR_COULD_NOT_WRITE_FILE;
    
    std::string header = meta->mm_header;
    // Strip trailing spaces
    while (!header.empty() && header.back() == ' ') header.pop_back();
    int space_count = std::count(header.begin(), header.end(), ' ');
    if (space_count >= 4) {
      // Find position of 5th space (start of 6th token)
      size_t pos = 0;
      int count = 0;
      while (count < 5 && pos != std::string::npos) {
        pos = header.find(' ', pos);
        if (pos != std::string::npos) {
          ++count;
          ++pos;
        }
      }
      if (pos != std::string::npos) {
        header = header.substr(0, pos); // up to and including 5th space
      }
    }
    if (index_bytes > 0) { // This determines if header is for bmtx
      header += " " + std::to_string(index_bytes) + " " + std::to_string(meta->value_bytes > 0 ? meta->value_bytes : 4);
    }
    fprintf(f, "%s\n", header.c_str());
    if (!meta->mm_header_body.empty()) {
      std::string header_body = meta->mm_header_body;
      // Remove all trailing newlines to ensure only one \n after print
      while (!header_body.empty() && header_body.back() == '\n') {
        header_body.pop_back();
      }
      fprintf(f, "%s\n", header_body.c_str());
    }
    // Write size line
    fprintf(f, "%ld %ld %ld\n", nrows, ncols, nentries);
    return 0;
  }

  template<typename IT, typename VT>
  int mm_write_binary_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta) {
    if (!f) return MMIO_ERR_COULD_NOT_WRITE_FILE;

    int index_bytes = required_bytes(std::max(coo->nrows, coo->ncols));

    IT nentries = coo->nnz;
    if (meta->is_symmetric && false) { // TODO optimize
      nentries = 0;
      for (IT i = 0; i < coo->nnz; ++i) {
        if (coo->row[i] <= coo->col[i]) {
          ++nentries;
        }
      }
    }

    int err = write_matrix_market_header(f, meta, index_bytes, coo->nrows, coo->ncols, nentries);
    printf("%s\n", meta->mm_header);
    if (err != 0) {
      fprintf(stderr, "Something went wrong writing the file header.\n");
      fclose(f);
      return err;
    }

    // Write binary data
    for (IT i = 0; i < coo->nnz; ++i) {
      //if (meta->is_symmetric && coo->row[i] > coo->col[i]) continue; // For patter matrices

      // Write row
      switch (index_bytes) {
        case 1: { uint8_t  v = (uint8_t)coo->row[i];  fwrite(&v, sizeof(v), 1, f); break; }
        case 2: { uint16_t v = (uint16_t)coo->row[i]; fwrite(&v, sizeof(v), 1, f); break; }
        case 4: { uint32_t v = (uint32_t)coo->row[i]; fwrite(&v, sizeof(v), 1, f); break; }
        case 8: { uint64_t v = (uint64_t)coo->row[i]; fwrite(&v, sizeof(v), 1, f); break; }
      }

      // Write column
      switch (index_bytes) {
        case 1: { uint8_t  v = (uint8_t)coo->col[i];  fwrite(&v, sizeof(v), 1, f); break; }
        case 2: { uint16_t v = (uint16_t)coo->col[i]; fwrite(&v, sizeof(v), 1, f); break; }
        case 4: { uint32_t v = (uint32_t)coo->col[i]; fwrite(&v, sizeof(v), 1, f); break; }
        case 8: { uint64_t v = (uint64_t)coo->col[i]; fwrite(&v, sizeof(v), 1, f); break; }
      }

      // Write value
      if (meta->value_type != mmio::ValueType::Pattern && coo->val != NULL) {
        // TODO generalize
        if (meta->value_bytes == 8) { double v = (double)coo->val[i]; fwrite(&v, sizeof(double), 1, f); }
        else                      { float  v =  (float)coo->val[i]; fwrite(&v, sizeof(float),  1, f); }
      }
    }

    fclose(f);
    return 0;
  }

  template<typename IT, typename VT>
  int mm_write_matrix_market(FILE *f, COO<IT, VT> *coo, Matrix_Metadata *meta) {
    if (!f) return MMIO_ERR_COULD_NOT_WRITE_FILE;

    IT nentries = coo->nnz;
    if (meta->is_symmetric) { // TODO optimize
      nentries = 0;
      for (IT i = 0; i < coo->nnz; ++i) {
        if (coo->row[i] >= coo->col[i]) {
          ++nentries;
        }
      }
    }

    int err = write_matrix_market_header(f, meta, -1, coo->nrows, coo->ncols, nentries);
    if (err != 0) {
      fprintf(stderr, "Something went wrong writing the file header.\n");
      fclose(f);
      return err;
    }

    for (IT i = 0; i < coo->nnz; ++i) {
      if (meta->is_symmetric && coo->row[i] < coo->col[i]) continue;
      if (meta->value_type == mmio::ValueType::Pattern) {
        fprintf(f, "%ld %ld\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1));
      } else if (meta->value_type == mmio::ValueType::Real) {
        if (meta->value_bytes == 8) {
          fprintf(f, "%ld %ld %.16g\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1), (double)coo->val[i]);
        } else {
          fprintf(f, "%ld %ld %.8g\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1), (float)coo->val[i]);
        }
      } else if (meta->value_type == mmio::ValueType::Integer) {
        fprintf(f, "%ld %ld %ld\n", (long)(coo->row[i] + 1), (long)(coo->col[i] + 1), (long)coo->val[i]);
      } else {
        return MMIO_ERR_UNSUPPORTED_TYPE;
      }
    }

    fclose(f);
    return 0;
  }

  void mm_set_metadata(Matrix_Metadata* meta, MM_typecode *matcode) {
    if (meta) {
      // Value type
      if (mm_is_real(*matcode)) {
        meta->value_type = mmio::ValueType::Real;
      } else if (mm_is_integer(*matcode)) {
        meta->value_type = mmio::ValueType::Integer;
      } else if (mm_is_pattern(*matcode)) {
        meta->value_type = mmio::ValueType::Pattern;
      } else {
        fprintf(stderr, "BUG: ValueType not recognized. Please report this.\n");
        exit(EXIT_FAILURE);
      }
      // Symmetry
      meta->is_symmetric = mm_is_symmetric(*matcode);
      meta->is_pattern = mm_is_pattern(*matcode);
    }
  }

  template<typename IT, typename VT>
  int mm_parse_header(FILE *f, IT &nrows, IT &ncols, IT &nentries, IT &nnz_upperbound, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta) {
    int err = mm_read_banner(f, matcode, is_bmtx, meta);
    if (err != 0) {
      fprintf(stderr, "Could not process Matrix Market banner. Error (%d)\n", err);
      return err;
    }
    if (mm_is_complex(*matcode)) {
      fprintf(stderr, "Cannot parse complex-valued matrices.\n");
      return MMIO_ERR_UNSUPPORTED_TYPE;
    }
    if (mm_is_array(*matcode)) {
      fprintf(stderr, "Cannot parse array matrices.\n");
      return MMIO_ERR_UNSUPPORTED_TYPE;
    }
    if (mm_is_skew(*matcode)) {
      fprintf(stderr, "Cannot parse skew-symmetric matrices.\n");
      return MMIO_ERR_UNSUPPORTED_TYPE;
    }
    if (mm_is_hermitian(*matcode)) {
      fprintf(stderr, "Cannot parse hermitian matrices.\n");
      return MMIO_ERR_UNSUPPORTED_TYPE;
    }

    uint64_t _nrows, _ncols, _nentries;
    err = mm_read_mtx_crd_size(f, &_nrows, &_ncols, &_nentries);
    if (err != 0) {
      fprintf(stderr, "Could not parse matrix shape (nrows, ncols, nnz).\n");
      return err;
    }


    uint8_t idx_bytes = 0;
    uint8_t value_bytes = 0;
    int IT_required_bytes = required_bytes(std::max(_nrows, _ncols));
    
    if (is_bmtx) {
      idx_bytes = mm_get_idx_bytes(*matcode);
      value_bytes = mm_get_val_bytes(*matcode);

      if(!(idx_bytes == 1 || idx_bytes == 2 || idx_bytes == 4 || idx_bytes == 8)
        && !(value_bytes == 1 || value_bytes == 2 || value_bytes == 4 || value_bytes == 8)) {
        fprintf(stderr, "BMTX BUG: this should not happen. idx: %hhu bytes, val: %hhu bytes. Please report this.\n", idx_bytes, value_bytes);
        return MMIO_ERR_SHOULD_NOT_HAPPEN;
      }
      if (idx_bytes < IT_required_bytes) {
        fprintf(stderr, "BMTX BUG: this should not happen. Need at least %d bytes, binary is written using %hhu bytes. Please report this.\n", IT_required_bytes, idx_bytes);
        return MMIO_ERR_SHOULD_NOT_HAPPEN;
      }
    }
    
    if (sizeof(IT) < (size_t)IT_required_bytes) {
      fprintf(stderr, "Error: Index Type (IT) is too small to represent matrix indices (need at least %d bytes, got %zu bytes).\n", IT_required_bytes, sizeof(IT));
      return MMIO_ERR_SHOULD_NOT_HAPPEN;
    }
    
    nnz_upperbound = mm_is_symmetric(*matcode) ? _nentries * 2 : _nentries; // For symmetric matrices THIS IS AN UPPER BOUND
    nrows = static_cast<IT>(_nrows);
    ncols = static_cast<IT>(_ncols);
    nentries = static_cast<IT>(_nentries);

    meta->index_bytes = idx_bytes;
    meta->value_bytes = value_bytes;

    mm_set_metadata(meta, matcode);

    return 0;
  }

  template<typename IT, typename VT>
  IT mm_count_duplicates(Entry<IT, VT>* entries, IT nentries, IT nnz_upperbound, Matrix_Metadata* meta) {
    IT nnz = nentries;
    if (meta->is_symmetric) {
      // Duplicate the entries for symmetric matrices
      for (uint64_t i = 0; i < nentries; ++i) {
        if (entries[i].row != entries[i].col) { // Do not duplicate diagonal
            nnz++;
        }
      }
    }
    return nnz;
  }

  template<typename IT, typename VT>
  IT mm_duplicate_entries_for_symmetric_matrices(Entry<IT, VT>* entries, IT nentries, IT nnz_upperbound, Matrix_Metadata* meta) {
    IT nnz = nentries;
    if (meta->is_symmetric) {
      // Duplicate the entries for symmetric matrices
      for (uint64_t i = 0, j = 0; i < nentries; ++i) {
        if (entries[i].row != entries[i].col) { // Do not duplicate diagonal
          entries[j + nentries].row = entries[i].col;
          entries[j + nentries].col = entries[i].row;
          entries[j + nentries].val = entries[i].val;
          ++nnz;
          ++j;
        }
      }
    }
    return nnz;
  }

  template<typename IT, typename VT>
  Entry<IT, VT>* mm_parse_file(FILE *f, IT &nrows, IT &ncols, IT &nnz, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta) {
    IT nentries = 0, nnz_upperbound = 0;
    int err = mm_parse_header<IT, VT>(f, nrows, ncols, nentries, nnz_upperbound, matcode, is_bmtx, meta);
    if (f == NULL || err != 0) {
      printf("Could not parse matrix header (error code: %d).\n", err);
      fclose(f);
      return NULL;
    }

    size_t entry_size = nentries * sizeof(Entry<IT, VT>);

    if (meta->is_symmetric) {
        entry_size *= 2;
    }

    Entry<IT, VT> *entries = (Entry<IT, VT> *)malloc(entry_size);
    err = mmio::io::mm_read_mtx_crd_data<IT, VT>(f, nentries, entries, matcode, is_bmtx, meta->index_bytes, meta->value_bytes);
    fclose(f);
    if (err != 0) {
      printf("Could not parse matrix data (error code: %d).\n", err);
      free(entries);
      return NULL;
    }

    nnz = mmio::io::mm_duplicate_entries_for_symmetric_matrices(entries, nentries, nnz_upperbound, meta);

    return entries;
  }

} // namespace mmio::io

MMIO_IO_EXPLICIT_TEMPLATE_INST(uint32_t, float)
MMIO_IO_EXPLICIT_TEMPLATE_INST(uint32_t, double)
MMIO_IO_EXPLICIT_TEMPLATE_INST(uint64_t, float)
MMIO_IO_EXPLICIT_TEMPLATE_INST(uint64_t, double)
MMIO_IO_EXPLICIT_TEMPLATE_INST(int, float)
MMIO_IO_EXPLICIT_TEMPLATE_INST(int, double)
MMIO_IO_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
MMIO_IO_EXPLICIT_TEMPLATE_INST(int64_t, float)
MMIO_IO_EXPLICIT_TEMPLATE_INST(int64_t, double)
