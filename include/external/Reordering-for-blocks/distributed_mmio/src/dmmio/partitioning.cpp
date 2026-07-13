#include <ccutils/macros.h>

#include "../../include/dmmio/partitioning.h"

namespace dmmio::partitioning {

  namespace edgeowner {
    namespace base {
      /* Edge to process functions
      *
      * These functions retrive the owner process starting by them 'global' indices.
      * Function are 1d or 2d functions; these two classes differs by the nprocess input parameteers. For
      * each of those, a typedef is specified.
      *
      * Currently, the only 1D scheme supported is the naive one.
      * Currently, the 2D supported scheme are naive, row-slicing and block-cycle.
      *
      */

      namespace d1 {
        // Defined in the header
        // typedef uint64_t (*Edge2proc1dFunction)(uint64_t i, uint64_t j,
        // 										uint64_t nrows, uint64_t ncols,
        // 										int nprocs);

        uint64_t naive( // Previous name: edge2proc_1d_naive
          uint64_t idx,
          uint64_t dimtosplit,
          int nprocs
        ) {
          int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs);
          return (idx / chunk_size );
        }

        uint64_t cycle( // Previous name: edge2proc_1d_cycle
          uint64_t idx,
          uint64_t dimtosplit,
          int nprocs
        ) {
          int chunk_size = PART_CHUNK_SIZE(dimtosplit,nprocs*nprocs);
          return ( (idx/chunk_size) % nprocs );
        }
      }

      namespace d2 {
        // Defined in the header
        // typedef uint64_t (*Edge2proc2dFunction)(uint64_t i, uint64_t j,
        // 										uint64_t nrows, uint64_t ncols,
        // 										int procs_per_row, int procs_per_col);

        uint64_t naive( // Previous name: edge2proc_2d_naive
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        ) {
          int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
          int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);
          return ( (i/rows_chunk_size)*procs_per_row + j/cols_chunk_size );
        }

        uint64_t rowslicing( // Previous name: edge2proc_2d_rowslicing
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        ) {
          int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
          int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row);

          int process_col_coord = j/cols_chunk_size;
          int process_row_coord = (i/rows_chunk_size) % procs_per_col;

          return process_row_coord*procs_per_row + process_col_coord;
        }

        uint64_t rowslicing_transpose( // Previous name: edge2proc_2d_rowslicing_transpose
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        ) {
          int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col);
          int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

          int process_col_coord = (j/cols_chunk_size) % procs_per_row;
          int process_row_coord = i/rows_chunk_size;
          return ( process_row_coord*procs_per_row + process_col_coord);
        }

        uint64_t blockcycle( // Previous name: edge2proc_2d_blockcycle
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        ) {
          int rows_chunk_size = PART_CHUNK_SIZE(nrows,procs_per_col*procs_per_col);
          int cols_chunk_size = PART_CHUNK_SIZE(ncols,procs_per_row*procs_per_row);

          int process_row_coord = (i/rows_chunk_size) % procs_per_col;
          int process_col_coord = (j/cols_chunk_size) % procs_per_row;
          return ( process_row_coord*procs_per_row + process_col_coord);
        }
      }

      // NOTE: I don't know where it is used
      uint64_t edge2proc_2d_1d(
        uint64_t i, uint64_t j,
        uint64_t m, uint64_t n,
        ProcessGrid * grid, int transpose
      ) {
        ASSERT((n % grid->col_size == 0), "pc %d does not divide n %lu\n", grid->col_size, n); // Process grid must evenly divide matrix dims
        ASSERT((m % (grid->row_size * grid->node_size) == 0), "pr*pz %d does not divide m %lu\n", grid->node_size * grid->node_size, m);

        int rpg = m / grid->row_size;
        int cpg = n / grid->col_size;


        int gid = (i / rpg) * grid->col_size + j / cpg;

        int rpp = rpg / grid->node_size;
        int intragroup_id = (i % rpg) / rpp;

        int pid = gid * grid->node_size + intragroup_id;

      #if DEBUG_PARTITION
        printf("(%lu, %lu) mapped to %d. rpg:%d, cpg:%d, gid:%d, rpp:%d, ig_id: %d\n",
                        i, j, pid, rpg, cpg, gid, rpp, intragroup_id);
        FLUSH_WAIT(500000);
      #endif

        ASSERT((pid < grid->global_size), "pid is %d, must be < %d\n", pid, grid->global_size);

        return pid;
      }
    }

    // Functions from edge to process
    uint64_t groupowner (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) { // Previous name: edge2group
      #ifndef SKIP_SETPARTFUNC_ASSERT
        ASSERT((self->grouprow2localrow!=NULL), "%s call before set_partitioning_functions\n", __func__);
      #endif
        return( self->edge2group(glob_row_id, glob_col_id, self->global_rows, self->global_cols, (self->grid)->col_size, (self->grid)->row_size) );
      }

      uint64_t internodeidowner (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) { // Previous name: edge2node
      #ifndef SKIP_SETPARTFUNC_ASSERT
        ASSERT((self->grouprow2localrow!=NULL), "%s call before set_partitioning_functions\n", __func__);
      #endif
        // uint64_t idxtosplit = (self->op=='l') ? (globalrow2grouprow(self, glob_row_id)) : (globalcol2groupcol(self, glob_col_id)) ;
        // uint64_t dimtosplit = (self->op=='l') ? self->group_rows : self->group_cols ;
        // BUG, tmp fix
        uint64_t idxtosplit = (indextransform::global2group::row(self, glob_row_id)); // Previous: globalrow2grouprow
        uint64_t dimtosplit = self->group_rows;
        return( self->edge2node(idxtosplit, dimtosplit, (self->grid)->node_size) );
      }

      uint64_t edge2owner (Partitioning *self, uint64_t glob_row_id, uint64_t glob_col_id) { // Previous name: edge2globalprocess

        // BUG for the row slicing we need to solve the problem of the transpose
        int gid = groupowner(self, glob_row_id, glob_col_id); // Previous: edge2group

        int intragroup_id = internodeidowner(self, glob_row_id, glob_col_id); // Previous: edge2node
        int pid = gid * ((self->grid)->node_size) + intragroup_id;

      #if DEBUG_PARTITION
        printf("%s: (%lu, %lu) mapped to %d. gid: %d, ig_id: %d\n",
                        self->my_part_str, glob_row_id, glob_col_id, pid, gid, intragroup_id);
        FLUSH_WAIT(500000);
      #endif
        return(pid);
      }
  }

  namespace indextransform {
    /* Global indices to local indices
    *
    * These functions retrive the lacal indices starting by the global indices. Note that global and local
    * are independent by the 3D process grid. All these function are base partitioning that can be composed
    * togheter to generate hybrid partitionings (this is also true for the edge to process functions).
    *
    * Function are 1d or 2d functions; these two classes differs by the nprocess input parameteers. For
    * each of those, a typedef is specified.
    *
    * Currently, the only 1D scheme supported is the naive one.
    * Currently, the 2D supported scheme are naive, row-slicing and block-cycle.
    *
    */

    namespace base {
      // General function for translating global indices to local indices (into header)
      // typedef uint64_t (*GlobalIdx2LocalIdx)(uint64_t globalid, uint64_t globalsize, int nprocs);

      uint64_t naive(uint64_t globalid, uint64_t globalsize, int nprocs) { // Previous name: globalindex2localindex_naive
        int chunk_size = PART_CHUNK_SIZE(globalsize,nprocs);
        return (globalid % chunk_size );
      }

      uint64_t cycle(uint64_t globalid, uint64_t globalsize, int nprocs) { // Previous name: globalindex2localindex_cycle
        int chunk_size = PART_CHUNK_SIZE(globalsize,nprocs*nprocs);
        return ( ((globalid/chunk_size)/nprocs)*chunk_size + (globalid%chunk_size) );
      }

      /* Each partitioning scheme use one of the previous defined general function to
      *  translate global indices to local indices.
      *
      * Naive partitioning:
      *     use naive function both for rows and columns.
      *
      * Row slicing partitioning:
      *     use cycle function for rows and naive function for columns.
      *
      * Block cycle partitioning:
      *     use cycle function both for rows and columns.
      *
      */
    }

    namespace global2group {
        // Function from upper index to inner index
        uint64_t col (Partitioning *self, uint64_t glob_col_id) { // Previous name: globalcol2groupcol
        #ifndef SKIP_SETPARTFUNC_ASSERT
          ASSERT((self->grouprow2localrow!=NULL), "%s call before set_partitioning_functions\n", __func__);
        #endif
          return( self->globalcol2groupcol(glob_col_id, self->global_cols, (self->grid)->col_size) );
        }

        uint64_t row (Partitioning *self, uint64_t glob_row_id) { // Previous name: globalrow2grouprow
        #ifndef SKIP_SETPARTFUNC_ASSERT
          ASSERT((self->grouprow2localrow!=NULL), "%s call before set_partitioning_functions\n", __func__);
        #endif
          return( self->globalrow2grouprow(glob_row_id, self->global_rows, (self->grid)->row_size) );
        }
    }

    namespace group2local {
      uint64_t col (Partitioning *self, uint64_t grp_col_id) { // Previous name: groupcol2localcol
      #ifndef SKIP_SETPARTFUNC_ASSERT
        ASSERT((self->grouprow2localrow!=NULL), "%s call before set_partitioning_functions\n", __func__);
      #endif
      //     int ncolsplit = (self->op=='l') ? 1 : ((self->grid)->pz) ;
        int ncolsplit = 1; // BUG, tmp fix
        return( self->groupcol2localcol(grp_col_id, self->group_cols, ncolsplit) );
      }

      uint64_t row (Partitioning *self, uint64_t grp_row_id) { // Previous name: grouprow2localrow
      #ifndef SKIP_SETPARTFUNC_ASSERT
        ASSERT((self->grouprow2localrow!=NULL), "%s call before set_partitioning_functions\n", __func__);
      #endif
        // int nrowsplit = (self->op=='l') ? ((self->grid)->pz) : 1 ;
        int nrowsplit = ((self->grid)->node_size); // BUG, tmp fix
        return( self->grouprow2localrow(grp_row_id, self->group_rows, nrowsplit) );
      }
    }

    namespace global2local {
      // Composed function: from global index to local index and from edge to global rank
      uint64_t col (Partitioning *self, uint64_t glob_col_id) { // Previous name: globalcol2localcol
        uint64_t grp_col_id = global2group::col (self, glob_col_id); // Previous: globalcol2groupcol
        return(  group2local::col(self, grp_col_id) ); // Previous: groupcol2localcol
      }

      uint64_t row (Partitioning *self, uint64_t glob_row_id) { // Previous name: globalrow2localrow
        uint64_t grp_row_id = global2group::row (self, glob_row_id); // Previous: globalrow2grouprow
        return( group2local::row(self, grp_row_id) ); // Previous: grouprow2localrow
      }
    }

    namespace transformCoo {
      template<typename IT, typename VT>
      void base(dmmio::DCOO<IT, VT>* dcoo, IndexTransformFn rowFn, IndexTransformFn colFn) {

          IT nnz = dcoo->coo->nnz;
          IT *row = dcoo->coo->row, *col = dcoo->coo->col;
          for (size_t i=0; i<nnz; i++) {
              row[i] = static_cast<IT>(rowFn(dcoo->partitioning, static_cast<uint64_t>(row[i])));
              col[i] = static_cast<IT>(colFn(dcoo->partitioning, static_cast<uint64_t>(col[i])));
          }
      }

      template<typename IT, typename VT>
      void global2group(dmmio::DCOO<IT, VT>* dcoo) {
          base(dcoo, global2group::row, global2group::row);

          Partitioning *part = dcoo->partitioning;
          dcoo->coo->nrows = part->group_rows;
          dcoo->coo->ncols = part->group_cols;
      }

      template<typename IT, typename VT>
      void group2local(dmmio::DCOO<IT, VT>* dcoo) {
          base(dcoo, group2local::row, group2local::row);

          Partitioning *part = dcoo->partitioning;
          dcoo->coo->nrows = part->local_rows;
          dcoo->coo->ncols = part->local_cols;
      }

      template<typename IT, typename VT>
      void global2local(dmmio::DCOO<IT, VT>* dcoo) {
          base(dcoo, global2local::row, global2local::row);

          Partitioning *part = dcoo->partitioning;
          dcoo->coo->nrows = part->local_rows;
          dcoo->coo->ncols = part->local_cols;
      }

      #define TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(IT, VT) \
        template void base(dmmio::DCOO<IT, VT>* dcoo, IndexTransformFn rowFn, IndexTransformFn colFn); \
        template void global2group(dmmio::DCOO<IT, VT>* dcoo); \
        template void group2local(dmmio::DCOO<IT, VT>* dcoo); \
        template void global2local(dmmio::DCOO<IT, VT>* dcoo);

      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(uint32_t, float)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(uint32_t, double)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(uint64_t, float)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(uint64_t, double)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(int, float)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(int, double)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(uint64_t, uint64_t)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(int64_t, float)
      TRANSFORMCOO_EXPLICIT_TEMPLATE_INST(int64_t, double)
    }

  }

} // namespace dmmio::partitioning
