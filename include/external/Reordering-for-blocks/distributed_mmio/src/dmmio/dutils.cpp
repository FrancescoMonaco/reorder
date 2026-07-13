#include <unistd.h>
#include <mpi.h>

#include "../../include/dmmio/dmmio.h"

namespace dmmio::utils {

  void ProcessGrid_print(const dmmio::ProcessGrid *grid, FILE* fp) {
	  if (grid->global_rank==0) {
      fprintf(fp, "========================\n");
      fprintf(fp, " ProcessGrid Details \n");
      fprintf(fp, "========================\n");
      fprintf(fp, "Total processes:\t %d\n", grid->global_size);
      fprintf(fp, "row size:\t %d\n", grid->row_size);
      fprintf(fp, "col size:\t %d\n", grid->col_size);
      fprintf(fp, "node size:\t %d\n", grid->node_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
	  sleep(1);

    for (int i=0; i<grid->global_size; i++) {
      if (grid->global_rank == i) {
        fprintf(fp, "----- Process %d -----\n", grid->global_rank);
        fprintf(fp, "Rank:\t %d\n", grid->global_rank);
        fprintf(fp, "row rank:\t %d\n", grid->row_rank);
        fprintf(fp, "col rank:\t %d\n", grid->col_rank);
        fprintf(fp, "node rank:\t %d\n", grid->node_rank);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    fflush(stdout);
    sleep(1);
    if (grid->global_rank == 0) fprintf(fp, "========================\n");
    MPI_Barrier(MPI_COMM_WORLD);
  }

  int ProcessGrid_graph(const dmmio::ProcessGrid *grid, FILE* fp) {
      int row_size    = grid->row_size;
      int col_size    = grid->col_size;
      int node_size   = grid->node_size;
      int global_size = grid->global_size;

      // Allocate 3D grid dynamically: [row][col][node]
      int ***tmp_grid = (int ***)malloc(col_size * sizeof(int **));
      for (int row = 0; row < col_size; ++row) {
          tmp_grid[row] = (int **)malloc(row_size * sizeof(int *));
          for (int col = 0; col < row_size; ++col) {
              tmp_grid[row][col] = (int *)malloc(node_size * sizeof(int));
              // Initialize with -1 (or any invalid rank)
              for (int node = 0; node < node_size; ++node) {
                  tmp_grid[row][col][node] = -1;
              }
          }
      }

      // Populate the grid
      int myvalues[3];
      myvalues[0] = grid->col_rank;
      myvalues[1] = grid->row_rank;
      myvalues[2] = grid->node_rank;
      int *allvalues = (int*)malloc(sizeof(int)*global_size*3);
      MPI_Allgather(myvalues, 3, MPI_INT, allvalues, 3, MPI_INT, MPI_COMM_WORLD);
      for (int i = 0; i < global_size; i++) {
          tmp_grid[allvalues[3*i]][allvalues[3*i+1]][allvalues[3*i+2]] = i;
      }

      // Header row: column labels
      if(grid->global_rank==0) fprintf(fp, "         ");
      for (int col = 0; col < row_size; ++col) {
          if(grid->global_rank==0) fprintf(fp, "col %-2d         ", col);
      }
      if(grid->global_rank==0) fprintf(fp, "\n");

      // Top border
      if(grid->global_rank==0) fprintf(fp, "       ");
      for (int col = 0; col < row_size; ++col) {
          if(grid->global_rank==0) fprintf(fp, "------------------- ");
      }
      if(grid->global_rank==0) fprintf(fp, "\n");

      // For each row
      for (int row = 0; row < col_size; ++row) {
          for (int node = 0; node < node_size; ++node) {
              if (node == 0) {
                  if(grid->global_rank==0) fprintf(fp, "row %-2d |", row);
              } else {
                  if(grid->global_rank==0) fprintf(fp, "       |");
              }

              for (int col = 0; col < row_size; ++col) {
                  int gid = tmp_grid[row][col][node];
                  if(grid->global_rank==0) fprintf(fp, " Node%d [%-3d]     |", node, gid);
              }
              if(grid->global_rank==0) fprintf(fp, "\n");
          }

          // Separator line
          if(grid->global_rank==0) fprintf(fp, "       ");
          for (int col = 0; col < row_size; ++col) {
              if(grid->global_rank==0) fprintf(fp, "------------------- ");
          }
          if(grid->global_rank==0) fprintf(fp, "\n");
      }

      // Cleanup memory
      for (int row = 0; row < col_size; ++row) {
          for (int col = 0; col < row_size; ++col) {
              free(tmp_grid[row][col]);
          }
          free(tmp_grid[row]);
      }
      free(tmp_grid);

      return 0;
  }

} // namespace dmmio::utils
