#include <iostream>

#include <mmio/mmio.h>
#include <mmio/utils.h>

#include <cstdint>
#include <string.h>
	
int main(int argc, char** argv) {
  std::string mtx_path = "";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-f" && i + 1 < argc) {
      mtx_path = argv[++i];
    }
  }

  // To avoid prefixing functions and structs you can use the following
  // using namespace mmio;

  mmio::Matrix_Metadata *meta = new mmio::Matrix_Metadata();

  mmio::COO<uint64_t, double> *coo_matrix = mmio::COO_read<uint64_t, double>(mtx_path.c_str(), false, meta);
  mmio::utils::COO_print_as_dense(coo_matrix, "Example COO");
  mmio::COO_destroy(&coo_matrix);
  
  mmio::CSR<uint64_t, double> *csr_matrix = mmio::CSR_read<uint64_t, double>(mtx_path.c_str(), false, meta);
  mmio::utils::CSR_print_as_dense(csr_matrix, "Example CSR");
  mmio::CSR_destroy(&csr_matrix);

  return 0;
}

