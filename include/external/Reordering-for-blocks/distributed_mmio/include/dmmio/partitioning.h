#ifndef DMMIO_PARTITIONING_H
#define DMMIO_PARTITIONING_H

#include "dmmio.h"

// Common macros: these macros are commonly used by all the partition functions
/*
* This macro compute the partitioning chunk size. GS stands for global size while NC stands for number of chunks
* To menage global sizes that are not multiples of the number of chunks, we define the chunk size as ((GS)/(NC))+1;
*   this mean that the last chunk can have less elements than the others (however, the unbalance is less then NC-1).
*/
#define PART_CHUNK_SIZE(GS, NC) (((GS)%(NC))==0) ? ((GS)/(NC)) : (((GS)/(NC))+1)

/*
* This macro retrive the process id from them grid coordinates. The M parameteer is neaded to retrive the grid and
* represents the number of columns inside the grid. RI and CI stands for row id and column id and represent the
* process grid coordinates.
*
* NOTE: currently unused
*/
#define GRIDCOO2PROCID(RI, CI, M) ((RI)*(M) + (CI))


namespace dmmio::partitioning {

  // Internal edge-to-proc mapping
  uint64_t edge2proc_2d_1d(
    uint64_t i, uint64_t j,
    uint64_t m, uint64_t n,
    ProcessGrid* grid,
    int transpose
  );

  // Accessors
  namespace edgeowner {
    namespace base {
      namespace d1 {
        uint64_t naive( // Previous name: edge2proc_1d_naive
          uint64_t idx,
          uint64_t dimtosplit,
          int nprocs
        );

        uint64_t cycle( // Previous name: edge2proc_1d_cycle
          uint64_t idx,
          uint64_t dimtosplit,
          int nprocs
        );
      }

      namespace d2 {
        uint64_t naive( // Previous name: edge2proc_2d_naive
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        );

        uint64_t rowslicing( // Previous name: edge2proc_2d_rowslicing
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        );

        uint64_t rowslicing_transpose( // Previous name: edge2proc_2d_rowslicing_transpose
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        );

        uint64_t blockcycle( // Previous name: edge2proc_2d_blockcycle
          uint64_t i, uint64_t j,
          uint64_t nrows, uint64_t ncols,
          int procs_per_row, int procs_per_col
        );
      }

      uint64_t edge2proc_2d_1d(
        uint64_t i, uint64_t j,
        uint64_t m, uint64_t n,
        ProcessGrid * grid, int transpose
      );
    }

    uint64_t groupowner(Partitioning* self, uint64_t glob_row_id, uint64_t glob_col_id); // Previous name: edge2group
    uint64_t internodeidowner(Partitioning* self, uint64_t glob_row_id, uint64_t glob_col_id); // Previous name: edge2node

    uint64_t edge2owner(Partitioning* self, uint64_t glob_row_id, uint64_t glob_col_id); // Previous name: edge2globalprocess
  }

  namespace indextransform {
    typedef uint64_t (*IndexTransformFn)(Partitioning* self, uint64_t glob_col_id);

    namespace base {
      uint64_t naive(uint64_t globalid, uint64_t globalsize, int nprocs); // Previous name: globalindex2localindex_naive
      uint64_t cycle(uint64_t globalid, uint64_t globalsize, int nprocs); // Previous name: globalindex2localindex_cycle
    }

    namespace global2group {
      uint64_t col(Partitioning* self, uint64_t glob_col_id); // Previous name: globalcol2groupcol
      uint64_t row(Partitioning* self, uint64_t glob_row_id); // Previous name: globalrow2grouprow
    }

    namespace group2local {
      uint64_t col(Partitioning* self, uint64_t grp_col_id); // Previous name: groupcol2localcol
      uint64_t row(Partitioning* self, uint64_t grp_row_id); // Previous name: grouprow2localrow
    }

    namespace global2local {
      uint64_t col(Partitioning* self, uint64_t glob_col_id); // Previous name: globalcol2localcol
      uint64_t row(Partitioning* self, uint64_t glob_row_id); // Previous name: globalrow2localrow
    }

    namespace transformCoo {
      template<typename IT, typename VT>
      void base(dmmio::DCOO<IT, VT>* dcoo, IndexTransformFn rowFn, IndexTransformFn colFn);

      template<typename IT, typename VT>
      void global2group(dmmio::DCOO<IT, VT>* dcoo);

      template<typename IT, typename VT>
      void group2local(dmmio::DCOO<IT, VT>* dcoo);

      template<typename IT, typename VT>
      void global2local(dmmio::DCOO<IT, VT>* dcoo);
    }
  }

} // namespace dmmio::partitioning

#endif // DMMIO_PARTITIONING_H
