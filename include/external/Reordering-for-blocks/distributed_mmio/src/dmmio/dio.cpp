#include <algorithm>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <ccutils/colors.h>
#include <ccutils/macros.h>

#include "../../include/mmio/io.h"
#include "../../include/dmmio/dio.h"
#include "../../include/dmmio/dmmio.h"
#include "../../include/dmmio/partitioning.h"

using Matrix_Metadata = mmio::Matrix_Metadata;
using Partitioning = dmmio::Partitioning;
using PartitioningType = dmmio::PartitioningType;
using ProcessGrid = dmmio::ProcessGrid;
template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
template<typename IT, typename VT> using COO = mmio::COO<IT, VT>;
template<typename IT, typename VT> using CSR = mmio::CSR<IT, VT>;

#define DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(IT, VT) \
  template Entry<IT, VT>* dmmio::io::sort_entries_by_owner(const Entry<IT, VT>* entries, const int* owner, size_t nentries); \
  template Entry<IT, VT>* dmmio::io::mm_parse_file_distributed(FILE *f, int rank, int mpi_comm_size, IT &nrows, IT &ncols, IT &local_nnz, MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta);


namespace dmmio::io {

  ProcessGrid *ProcessGrid_create(int row_size, int col_size, int node_size) {
    ASSERT(row_size > 0, "row_size must be > 0");
    ASSERT(col_size > 0, "col_size must be > 0");
    ASSERT(node_size > 0, "node_size must be > 0");
    ASSERT(row_size == col_size, "Currently only square grids are supported");

    ProcessGrid *grid = (ProcessGrid*)malloc(sizeof(ProcessGrid));

    grid->row_size = row_size;
    grid->col_size = col_size;
    grid->node_size = node_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &(grid->global_rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->global_size));
    grid->world_comm = MPI_COMM_WORLD;

    ASSERT((grid->global_size % (row_size*col_size) == 0), "Total number of processes must be a multiple of row_size * col_size");

    // Compute 3D coordinates from linear rank
    int i = grid->global_rank / (col_size * node_size);      // row index
    int remainder = grid->global_rank % (col_size * node_size);
    int j = remainder / node_size;                   // col index
    int k = remainder % node_size;                   // node index

    // --- Node communicator (same i,j, different k)
    int node_color = i * col_size + j; // unique per (i,j)
    int node_key   = k;
    MPI_Comm_split(MPI_COMM_WORLD, node_color, node_key, &(grid->node_comm));

    // --- Row communicator: (i, *, k) → fix i and k, vary j
    int row_color = i * node_size + k; // unique per (i,k)
    int row_key   = j;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, row_key, &(grid->row_comm));

    // --- Column communicator: (*, j, k) → fix j and k, vary i
    int col_color = j * node_size + k; // unique per (j,k)
    int col_key   = i;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, col_key, &(grid->col_comm));

    MPI_Barrier(grid->world_comm);

    MPI_Comm_rank(grid->row_comm, &(grid->row_rank));
    MPI_Comm_size(grid->row_comm, &(grid->row_size));

    MPI_Comm_rank(grid->col_comm, &(grid->col_rank));
    MPI_Comm_size(grid->col_comm, &(grid->col_size));

    MPI_Comm_rank(grid->node_comm, &(grid->node_rank));
    MPI_Comm_size(grid->node_comm, &(grid->node_size));

    MPI_Comm_set_name(grid->row_comm,  "COMMROW");
    MPI_Comm_set_name(grid->col_comm,  "COMMCOL");
    MPI_Comm_set_name(grid->node_comm, "COMMNODE");

    // if (grid->row_size != row_size)  MPI_Abort(MPI_COMM_WORLD, __LINE__);
    // if (grid->col_size != col_size)  MPI_Abort(MPI_COMM_WORLD, __LINE__);
    // if (grid->node_size != node_size) MPI_Abort(MPI_COMM_WORLD, __LINE__);

    return grid;
  }

  // ==================================================== Partitioning functions ==================================================
  /*  Functions and macros to compute the partitioning
  *
  *  The partitioning could be hyerarcycal ...to explain...
  *
  */

  void init_partitioning(Partitioning *self) {
    self->type_str = "";

    self->grid = NULL;

    self->op = dmmio::Operation::None;

    self->global_rows = 0;
    self->global_cols = 0;
    self->group_rows = 0;
    self->group_cols = 0;
    self->local_rows = 0;
    self->local_cols = 0;

    self->edge2group = NULL;
    self->edge2node = NULL;
    self->globalcol2groupcol = NULL;
    self->globalrow2grouprow = NULL;
    self->groupcol2localcol = NULL;
    self->grouprow2localrow = NULL;
  }

  void set_partitioning_str (Partitioning *self) {
    switch (self->type) {
      case PartitioningType::RowSlicing:
        self->type_str = "partRslicing";
        break;
      case PartitioningType::Bcycle:
        self->type_str = "partBcycle";
        break;
      case PartitioningType::NaiveCycle:
        self->type_str = "partNaiveCycle";
        break;
      case PartitioningType::BcycleCycle:
        self->type_str = "partBcycleCycle";
        break;
      case PartitioningType::RowSlicingCycle:
        self->type_str = "partRslicingCycle";
        break;
      default:
        self->type_str = "partNaive";
        break;
    }
  }

  void set_partitioning_transpose(Partitioning *self, Operation op) {
    // ASSERT((op=='l')||(op=='r'), "Error in function %s, op must be 'l' (left) or 'r' (right).", __func__);
    self->op = op;
  }

  void set_partitioning_type (Partitioning *self, PartitioningType type, Operation op) {
    init_partitioning(self);
    self->type = type;
    set_partitioning_str(self);
    set_partitioning_transpose(self, op);
  }

  void set_partitioning_grid (Partitioning *self, ProcessGrid* grid) {
    self->grid = grid;
  }

  void set_partitioning_global_dim (Partitioning *self, uint64_t n, uint64_t m) {
    self->global_rows = n;
    self->global_cols = m;
  }

  void set_partitioning_group_dim (Partitioning *self) {
    ASSERT((self->grid!=NULL), "%s call before self->grid set\n", __func__);

    ASSERT(((self->global_rows!=0)&&(self->global_cols!=0)), \
      "%s call before matrix dim are set:\n\tglobal_rows: %lu\n\tglobal_cols: %lu\n", \
      __func__, self->global_rows, self->global_cols);

    self->group_rows = PART_CHUNK_SIZE(self->global_rows, (self->grid)->row_size);
    self->group_cols = PART_CHUNK_SIZE(self->global_cols, (self->grid)->col_size);
  }

  void set_partitioning_local_dim (Partitioning *self) {
    ASSERT((self->grid!=NULL), "%s call before self->grid set\n", __func__);

    ASSERT(((self->global_rows!=0)&&(self->global_cols!=0)), \
      "%s call before matrix dim are set:\n\tglobal_rows: %lu\n\tglobal_cols: %lu\n", \
      __func__, self->global_rows, self->global_cols);

    ASSERT(((self->group_rows!=0)&&(self->group_cols!=0)), \
      "%s call before matrix dim are set:\n\tgroup_rows: %lu\n\tgroup_cols: %lu\n", \
      __func__, self->group_rows, self->group_cols);

    // BUG, tmp fix
    //     if (self->op == 'l') {
    //         self->local_matrix_rows = PART_CHUNK_SIZE(self->group_rows, (self->grid)->pz);
    //         self->local_matrix_cols = self->group_cols;
    //     } else {
    //         self->local_matrix_rows = self->group_rows;
    //         self->local_matrix_cols = PART_CHUNK_SIZE(self->group_cols, (self->grid)->pz);
    //     }
    self->local_rows = PART_CHUNK_SIZE(self->group_rows, (self->grid)->node_size);
    self->local_cols = self->group_cols;
  }

  void set_partitioning_functions (Partitioning *self) {
    ASSERT((self->grid!=NULL), "%s call before self->grid set\n", __func__);

    ASSERT(((self->global_rows!=0)&&(self->global_cols!=0)), \
      "%s call before matrix dim are set:\n\tglobal_rows: %lu\n\tglobal_cols: %lu\n", \
      __func__, self->global_rows, self->global_cols);

    ASSERT(((self->group_rows!=0)&&(self->group_cols!=0)), \
      "%s call before group matrix dim are set:\n\tgroup_rows: %lu\n\tgroup_cols: %lu\n", \
      __func__, self->group_rows, self->group_cols);

    switch (self->type) {
      case PartitioningType::Bcycle: {
        self->edge2group          = dmmio::partitioning::edgeowner::base::d2::blockcycle; // Previous: edge2proc_2d_blockcycle
        self->edge2node           = dmmio::partitioning::edgeowner::base::d1::naive; // Previous: edge2proc_1d_naive
        self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::cycle; // Previous: globalindex2localindex_cycle
        self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::cycle; // Previous: globalindex2localindex_cycle
        self->groupcol2localcol   = dmmio::partitioning::indextransform::base::naive; // Previous: globalindex2localindex_naive
        self->grouprow2localrow   = dmmio::partitioning::indextransform::base::naive; // Previous: globalindex2localindex_naive
        break;
      }

      case PartitioningType::BcycleCycle: {
        self->edge2group          = dmmio::partitioning::edgeowner::base::d2::blockcycle;
        self->edge2node           = dmmio::partitioning::edgeowner::base::d1::cycle; // Previous: edge2proc_1d_cycle
        self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::cycle;
        self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::cycle;
        self->groupcol2localcol   = dmmio::partitioning::indextransform::base::naive;
        self->grouprow2localrow   = dmmio::partitioning::indextransform::base::cycle;
        break;
      }

      case PartitioningType::RowSlicing: {
        self->edge2node = dmmio::partitioning::edgeowner::base::d1::naive;

        if (self->op == Operation::None) {
          self->edge2group          = dmmio::partitioning::edgeowner::base::d2::rowslicing; // Previous: edge2proc_2d_rowslicing
          self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::naive;
          self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::cycle;
        } else {
          self->edge2group          = dmmio::partitioning::edgeowner::base::d2::rowslicing_transpose; // Previous: edge2proc_2d_rowslicing_transpose
          self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::cycle;
          self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::naive;
        }

        self->groupcol2localcol = dmmio::partitioning::indextransform::base::naive;
        self->grouprow2localrow = dmmio::partitioning::indextransform::base::naive;
        break;
      }

      case PartitioningType::RowSlicingCycle: {
        self->edge2node = dmmio::partitioning::edgeowner::base::d1::cycle;

        if (self->op == Operation::None) {
          self->edge2group          = dmmio::partitioning::edgeowner::base::d2::rowslicing;
          self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::naive;
          self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::cycle;
        } else {
          self->edge2group          = dmmio::partitioning::edgeowner::base::d2::rowslicing_transpose;
          self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::cycle;
          self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::naive;
        }

        self->groupcol2localcol = self->op == Operation::None 
          ? dmmio::partitioning::indextransform::base::naive
          : dmmio::partitioning::indextransform::base::cycle;
        self->grouprow2localrow = self->op == Operation::None
          ? dmmio::partitioning::indextransform::base::cycle
          : dmmio::partitioning::indextransform::base::naive;
        break;
      }

      case PartitioningType::Naive: {
          self->edge2group          = dmmio::partitioning::edgeowner::base::d2::naive; // Previous: edge2proc_2d_naive
          self->edge2node           = dmmio::partitioning::edgeowner::base::d1::naive;
          self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::naive;
          self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::naive;
          self->groupcol2localcol   = dmmio::partitioning::indextransform::base::naive;
          self->grouprow2localrow   = dmmio::partitioning::indextransform::base::naive;
          break;
      }

      case PartitioningType::NaiveCycle: {
        // PartitioningType::RowSlicing and Partitioning_NAIVE use the same column partitioning
        self->edge2group          = dmmio::partitioning::edgeowner::base::d2::naive;
        self->edge2node           = dmmio::partitioning::edgeowner::base::d1::cycle;
        self->globalcol2groupcol  = dmmio::partitioning::indextransform::base::naive;
        self->globalrow2grouprow  = dmmio::partitioning::indextransform::base::naive;
        self->groupcol2localcol   = dmmio::partitioning::indextransform::base::naive;
        self->grouprow2localrow   = dmmio::partitioning::indextransform::base::cycle;
        break;
      }
    }
  }

  // Sort entries by owner, producing a new sorted vector
  template<typename IT, typename VT>
  Entry<IT, VT>* sort_entries_by_owner(const Entry<IT, VT>* entries, const int* owner, size_t nentries) {
    // Combine entries and owner into a vector of pairs
    std::vector<std::pair<int, Entry<IT, VT>>> combined(nentries);
    for (size_t i = 0; i < nentries; ++i) {
      combined[i] = { owner[i], entries[i] };
    }

    // Sort by owner
    std::sort(
      combined.begin(), combined.end(),
      [](const std::pair<int, Entry<IT, VT>>& a, const std::pair<int, Entry<IT, VT>>& b) {
        return a.first < b.first;
      }
    );

    // Allocate new array for sorted entries
    Entry<IT, VT>* sorted_entries = (Entry<IT, VT>*)malloc(nentries * sizeof(Entry<IT, VT>));
    for (size_t i = 0; i < nentries; ++i) {
      sorted_entries[i] = combined[i].second;
    }

    return sorted_entries;
  }

  template<typename IT, typename VT>
  Entry<IT, VT>* mm_parse_file_distributed(
    FILE *f,
    int rank, int mpi_comm_size,
    IT &nrows, IT &ncols, IT &local_nnz,
    MM_typecode *matcode, bool is_bmtx, Matrix_Metadata* meta
  ) {
    ASSERT(is_bmtx, "Distributed read of non-binary MTX files is not supported yet") // TODO implement

    IT nentries = 0, nnz_upperbound = 0;
    if (f == NULL) {
      printf("File pointer is NULL\n");
      return NULL;
    }
    int err = mmio::io::mm_parse_header<IT, VT>(f, nrows, ncols, nentries, nnz_upperbound, matcode, is_bmtx, meta);
    if (err != 0) {
      printf("Could not parse matrix header (error code: %d).\n", err);
      fclose(f);
      return NULL;
    }
    
    long int pos = ftell(f);
    uint16_t line_size = meta->is_pattern ? (2 * meta->index_bytes) : (2 * meta->index_bytes + meta->value_bytes);

    nentries = (rank < nnz_upperbound% mpi_comm_size) ? (nnz_upperbound / mpi_comm_size + 1) : (nnz_upperbound / mpi_comm_size);


    uint32_t to_skip = (rank < nnz_upperbound % mpi_comm_size) ? (nentries * rank) : (nentries * rank + nnz_upperbound % mpi_comm_size);
    Entry<IT, VT> *entries = (Entry<IT, VT> *)malloc(nentries * sizeof(Entry<IT, VT>));

    if (fseek(f, to_skip*line_size, SEEK_CUR) != 0) {
      perror("fseek failed");
      fclose(f);
      return(NULL);
    }

    long int new_pos = ftell(f);
    // fprintf(stdout, "[DEBUG at line %d] process %d: %u nentries, %u to skip, %ld is the starting position and %ld is the current position (%u B)\n", __LINE__, rank, nentries, to_skip, pos, new_pos, to_skip*line_size);

    err = mm_read_mtx_crd_data<IT, VT>(f, nentries, entries, matcode, is_bmtx, meta->index_bytes, meta->value_bytes);
    fclose(f);
    if (err != 0) {
      printf("Could not parse matrix data (error code: %d).\n", err);
      free(entries);
      return NULL;
    }

    local_nnz = mm_duplicate_entries_for_symmetric_matrices(entries, nentries, local_nnz, meta);

    return entries;
  }

} // namespace dmmio::io

DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint32_t, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint32_t, double)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint64_t, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint64_t, double)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(int, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(int, double)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(int64_t, float)
DMMIO_DISTRIBUTED_EXPLICIT_TEMPLATE_INST(int64_t, double)
