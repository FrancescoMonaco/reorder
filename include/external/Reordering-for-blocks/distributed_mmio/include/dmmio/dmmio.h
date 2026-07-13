#ifndef __DMMIO_H__
#define __DMMIO_H__

#include <cstdint>
#include <cstdio>
#include <mpi.h>
#include <string>

#include "../mmio/mmio.h"

namespace dmmio {

  /** Partitioning types */
  enum class PartitioningType {
    Naive,
    Bcycle,
    RowSlicing,
    NaiveCycle,
    BcycleCycle,
    RowSlicingCycle
  };

  /** Partitioning options */
  enum class Operation {
    None,
    Transpose
  };

  /** Process grid */
  struct ProcessGrid {
    int global_rank;
    int global_size;

    int row_rank, col_rank, node_rank;
    int row_size, col_size, node_size;

    MPI_Comm world_comm, row_comm, col_comm, node_comm;
  };

  /********************* Partitioning Functions Interface ***************************/

  typedef uint64_t (*Edge2proc1dFunction)(uint64_t idx,
                                          uint64_t dimtosplit,
                                          int nprocs);

  typedef uint64_t (*Edge2proc2dFunction)(uint64_t i,         uint64_t j,
                                          uint64_t nrows,     uint64_t ncols,
                                          int procs_per_row,  int procs_per_col);

  typedef uint64_t (*GlobalIdx2LocalIdx)(uint64_t globalid, uint64_t globalsize, int nprocs);

  /** Partitioning descriptor */
  struct Partitioning {
    PartitioningType type;
    const char *type_str;
    Operation op;

    ProcessGrid* grid;

    uint64_t global_rows, global_cols;
    uint64_t group_rows, group_cols;
    uint64_t local_rows, local_cols;

    // Generic Function pointers
    Edge2proc2dFunction edge2group;
    Edge2proc1dFunction edge2node;
    GlobalIdx2LocalIdx  globalcol2groupcol;
    GlobalIdx2LocalIdx  globalrow2grouprow;
    GlobalIdx2LocalIdx  groupcol2localcol;
    GlobalIdx2LocalIdx  grouprow2localrow;
  };

  /** Distributed COO container */
  template <typename IT, typename VT>
  struct DCOO {
    Partitioning *partitioning;
    mmio::COO<IT, VT> *coo;
  };

  Partitioning* Partitioning_create(int matrix_rows, int matrix_cols, int grid_rows, int grid_cols, int grid_node_size, PartitioningType partitioning_type, Operation operation);
  void Partitioning_destroy(Partitioning **partitioning);

  // -------- Public API functions --------
  
  template <typename IT, typename VT>
  DCOO<IT, VT>* DCOO_read(
    const char* filename,
    int mpi_comm_size, int rank,
    int grid_rows, int grid_cols, int grid_node_size,
    PartitioningType part_type,
    Operation op = Operation::None,
    bool expl_val_for_bin_mtx = false,
    mmio::Matrix_Metadata* meta = nullptr,
    int padding=1
  );

  template <typename IT, typename VT>
  DCOO<IT, VT>* DCOO_read_f(
    FILE* f,
    int comm_size, int rank,
    int grid_rows, int grid_cols, int grid_node_size,
    PartitioningType part_type,
    Operation op = Operation::None,
    bool is_bmtx = false,
    bool expl_val_for_bin_mtx = false,
    mmio::Matrix_Metadata* meta = nullptr,
    int padding=1
  );

  template<typename IT, typename VT>
  void DCOO_destroy(DCOO<IT, VT>** dcoo);

  void Partitioning_destroy(Partitioning **partitioning);

} // namespace dmmio

#endif // __DMMIO_H__
