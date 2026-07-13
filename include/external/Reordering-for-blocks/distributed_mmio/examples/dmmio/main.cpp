#include <mpi.h>
#include <iostream>

#include <mmio/mmio.h>
#include <mmio/utils.h>
#include <dmmio/dmmio.h>
#include <dmmio/dutils.h>

#include <cstdint>
#include <string.h>
#include <memory>

#define DEBUG_RESULT
#define ENCODE_SHIFT 10000
#define ENCODE_COO2VAL( R , C ) ((R) * ENCODE_SHIFT + (C))
	
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

  mmio::Matrix_Metadata *meta = new mmio::Matrix_Metadata();
  dmmio::DCOO<uint64_t, double> *dcoo = dmmio::DCOO_read<uint64_t, double>(
    mtx_path.c_str(), 
    world_size, world_rank,
    nprocrows, nproccols, world_size/(nprocrows*nproccols),
    (dmmio::PartitioningType)part_num, transpose_flag ? dmmio::Operation::Transpose : dmmio::Operation::None,
    false, meta
  );

  MPI_Barrier(MPI_COMM_WORLD);
  dmmio::utils::ProcessGrid_print(dcoo->partitioning->grid);
  MPI_Barrier(MPI_COMM_WORLD);

  // TODO print

  // delete meta;
  dmmio::DCOO_destroy(&dcoo);

  MPI_Finalize();
  return 0;
}

