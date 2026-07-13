#ifndef __DMMIO_IO_H__
#define __DMMIO_IO_H__

#include <cstdint>
#include <cstdio>
#include <string>

#include "../mmio/io.h"
#include "dmmio.h"

namespace dmmio::io {

  inline int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
  }

  ProcessGrid* ProcessGrid_create(int row_size, int col_size, int node_size);

  template <typename IT, typename VT>
  mmio::io::Entry<IT, VT>* mm_parse_file_distributed(
    FILE* f,
    int rank,
    int mpi_comm_size,
    IT& nrows, IT& ncols,
    IT& local_nnz,
    MM_typecode* matcode,
    bool is_bmtx,
    mmio::Matrix_Metadata* meta
  );

  template <typename IT, typename VT>
  mmio::io::Entry<IT, VT>* sort_entries_by_owner(
    const mmio::io::Entry<IT, VT>* entries,
    const int* owner,
    size_t nentries
  );

  void compute_local_dims(
    const dmmio::Partitioning* part,
    uint64_t global_nrows, uint64_t global_ncols,
    uint64_t* loc_nrows, uint64_t* loc_ncols
  );

  void set_partitioning_type(dmmio::Partitioning* self, dmmio::PartitioningType partitioning_type, dmmio::Operation operation = dmmio::Operation::None);
  void set_partitioning_grid(dmmio::Partitioning* self, dmmio::ProcessGrid* grid);
  void set_partitioning_global_dim(dmmio::Partitioning* self, uint64_t n, uint64_t m);
  void set_partitioning_group_dim(dmmio::Partitioning* self);
  void set_partitioning_local_dim(dmmio::Partitioning* self);
  void set_partitioning_functions(dmmio::Partitioning* self);

  // uint64_t globalcol2groupcol (Partitioning *self, uint64_t glob_col_id);
  // uint64_t globalrow2grouprow (Partitioning *self, uint64_t glob_row_id);
  // uint64_t edge2group (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);
  // uint64_t edge2node (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);
  // uint64_t groupcol2localcol (Partitioning *self, uint64_t grp_col_id);
  // uint64_t grouprow2localrow (Partitioning *self, uint64_t grp_row_id);
  // uint64_t globalcol2localcol (Partitioning *self, uint64_t glob_col_id);
  // uint64_t globalrow2localrow (Partitioning *self, uint64_t glob_row_id);
  // uint64_t edge2globalprocess (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id);
  
} // namespace dmmio::io

#endif // DMMIO_IO_H
