#include <mpi.h>
#include <iostream>

#include <mmio/io.h>
#include <mmio/mmio.h>
#include <mmio/utils.h>
#include <dmmio/dio.h>
#include <dmmio/dmmio.h>
#include <dmmio/dutils.h>
#include <dmmio/partitioning.h>

#include <ccutils/mpi/mpi_macros.h>

#include <cstdint>
#include <string.h>
#include <memory>

#define DEBUG_RESULT
#define ENCODE_SHIFT 10000
#define ENCODE_COO2VAL( R , C ) ((R) * ENCODE_SHIFT + (C))

template<typename IT, typename VT> using Entry = mmio::io::Entry<IT, VT>;
	
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size;
  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int name_len;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name, &name_len);

  std::cout << "Hello world from processor " << processor_name
            << ", rank " << world_rank << " out of " << world_size << " processors\n";
  MPI_Barrier(MPI_COMM_WORLD);

  // Default file path
  std::string mtx_path = "";
  int nprocrows = 2, nproccols = 2, part_num = 0, transpose_flag = 0;

  // Simple argument parsing
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-f" && i + 1 < argc) {
        mtx_path = argv[++i];
    }
    if (arg == "-r" && i + 1 < argc) {
        nprocrows = atoi(argv[++i]);
    }
    if (arg == "-c" && i + 1 < argc) {
        nproccols = atoi(argv[++i]);
    }
    if (arg == "-p" && i + 1 < argc) {
        part_num = atoi(argv[++i]);
    }
    if (arg == "-t" && i + 1 < argc) {
        transpose_flag = atoi(argv[++i]);
    }
  }

  if (mtx_path.empty()) {
    if (world_rank == 0) {
      std::cerr << "Error: Missing matrix file path. Use -f <path_to_mtx_file>\n";
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  if (world_rank == 0) {
    std::cout << "Matrix file path: " << mtx_path << std::endl;
    std::cout << "Number of processes per row: " << nprocrows << std::endl;
    std::cout << "Number of processes per col: " << nproccols << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ==== End of Setup ====


  mmio::Matrix_Metadata *meta = new mmio::Matrix_Metadata();
  dmmio::DCOO<uint32_t, float> *dcoo = dmmio::DCOO_read<uint32_t, float>(
    mtx_path.c_str(), 
    world_size, world_rank,
    nprocrows, nproccols, world_size/(nprocrows*nproccols),
    (dmmio::PartitioningType)part_num, transpose_flag ? dmmio::Operation::Transpose : dmmio::Operation::None,
    false, meta
  );

  MPI_Barrier(MPI_COMM_WORLD);
  dmmio::utils::ProcessGrid_print(dcoo->partitioning->grid);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_ALL_PRINT(mmio::utils::COO_print_as_dense(dcoo->coo, std::string("Rank ") + std::to_string(world_rank), fp))

  uint32_t local_nnz = dcoo->coo->nnz;
  // Define the dbg_entry (which embed an entry and the entry owner)
  typedef struct dbg_entry {
    Entry<uint32_t, float> entry;
    int owner;
  } DbgEntry;
  DbgEntry* dbg_entries = (DbgEntry*)malloc(local_nnz * sizeof(DbgEntry));

  // MPI_ALL_PRINT(
  //   fprintf(fp, "Rank %d, local nnz: %u, sizeof(DbgEntry): %u, dbg_entries: %p\n", world_rank, local_nnz, sizeof(DbgEntry), dbg_entries);
  //   fprintf(fp, "COO:\n");
  //   for (int i=0; i<local_nnz; i++) {
  //     fprintf(fp, "\t%lu, %lu, %.1f --> %d\n", dcoo->coo->row[i], dcoo->coo->col[i], 1.0, 0);
  //   }
  //   fprintf(fp, "\n")
  // )

  //dmmio::partitioning::indextransform:: 
  for (int i=0; i<local_nnz; i++) {
    // FIXME convert back to global indices
    dbg_entries[i].entry.row = dcoo->coo->row[i];
    dbg_entries[i].entry.col = dcoo->coo->col[i];
    dbg_entries[i].entry.val = 1.0;
    dbg_entries[i].owner = dmmio::partitioning::edgeowner::edge2owner(dcoo->partitioning, dcoo->coo->row[i], dcoo->coo->col[i]);
  }

  // #define DEBUG
  #ifdef DEBUG
  MPI_ALL_PRINT(
    fprintf(fp, "Receved entries:\n");
    for (int i=0; i<local_nnz; i++) {
      fprintf(fp, "\t%lu, %lu, %lu --> %d\n", dbg_entries[i].entry.row, dbg_entries[i].entry.col, dbg_entries[i].entry.val, dbg_entries[i].owner);
    }
    fprintf(fp, "\n")
  )
  #endif

  // Allghaterv to collect all the coordinates on a single process and perform a debug print
  int* gather_counts = (int*)malloc(sizeof(int) * world_size);
  int* gather_displs = (int*)malloc(sizeof(int) * world_size);

  MPI_Allgather(&local_nnz, 1, MPI_INT, gather_counts, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i = 0; i < world_size; i++) gather_counts[i] *= sizeof(DbgEntry);

  gather_displs[0] = 0;
  for (int i = 1; i < world_size; i++) gather_displs[i] = gather_displs[i - 1] + gather_counts[i - 1];
  int total_entries = (gather_displs[world_size - 1] + gather_counts[world_size - 1]) / sizeof(DbgEntry);

  DbgEntry* all_entries = (DbgEntry*)malloc(total_entries * sizeof(DbgEntry));

  MPI_Allgatherv(
    dbg_entries,                   // send buffer
    local_nnz * sizeof(DbgEntry),  // send count
    MPI_BYTE,                      // type
    all_entries,                   // receive buffer
    gather_counts,                 // receive counts
    gather_displs,                 // displacements
    MPI_BYTE,                      // type
    MPI_COMM_WORLD
  );

  #ifdef DEBUG_ALLGHATERV
  MPI_PROCESS_PRINT( MPI_COMM_WORLD, 0,
      fprintf(stdout, "\n==================== ALLGATHERV CHECK ====================\n");
      fprintf(stdout, "Rank %d: gathered %d entries (total %d):\n", world_rank, total_entries, total_entries);
      for (int i = 0; i < total_entries; i++) {
          fprintf(stdout, "\trow: %lu, col: %lu, val: %lu, owner: %d\n",
                  all_entries[i].entry.row, all_entries[i].entry.col, all_entries[i].entry.val, all_entries[i].owner);
      }
      fprintf(stdout, "\n")
  )
  #endif

  // Prints coords on a file to be printed by using 'plotPartitioningCheck.py' script
  if (world_rank == 0) {
    // char csvname[200];
    // sprintf(csvname, "checktest_%d_%c.csv", partitioning_type, op);
    // FILE *statfile = fopen(csvname, "w");
    // FILE *statfile = stdout;
    printf("[OUT CSV]\n");
    printf("rank,rowid,colid,val\n");

    for (int k=0; k<total_entries; k++) {
      printf("%4d,%4lu,%4lu,%lu\n",
              all_entries[k].owner,
              all_entries[k].entry.row,
              all_entries[k].entry.col,
              all_entries[k].entry.val);
    }
    printf("[END OUT CSV]\n");
    // fclose(statfile);
  }

  // Test the COO Index Transform
  if (world_rank == 0) fprintf(stdout, "Global to group index transform...\n");
  MPI_Barrier(MPI_COMM_WORLD);
  dmmio::partitioning::indextransform::transformCoo::global2group(dcoo);
  MPI_ALL_PRINT(mmio::utils::COO_print_as_dense(dcoo->coo, std::string("Rank ") + std::to_string(world_rank), fp))

  delete meta;
  dmmio::DCOO_destroy(&dcoo);

  MPI_Finalize();
  return 0;
}

