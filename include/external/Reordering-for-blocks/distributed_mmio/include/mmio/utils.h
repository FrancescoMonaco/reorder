#ifndef __MMIO_UTILS_H__
#define __MMIO_UTILS_H__

#include "mmio.h"

namespace mmio::utils {

  template<typename IT, typename VT>
  void CSR_print(mmio::CSR<IT, VT> *csr, std::string header="", FILE* fp=stdout);

  template<typename IT, typename VT>
  void CSR_print_as_dense(mmio::CSR<IT, VT> *csr, std::string header="", FILE* fp=stdout);

  template<typename IT, typename VT>
  void COO_print(mmio::COO<IT, VT> *coo, std::string header="", FILE* fp=stdout);

  template<typename IT, typename VT>
  void COO_print_as_dense(mmio::COO<IT, VT> *coo, std::string header="", FILE* fp=stdout);

  template<typename IT, typename VT>
  void CSX_print_as_dense(mmio::CSX<IT, VT> *csx, std::string header="", FILE* fp=stdout);

} // namespace mmio::utils

#endif // __MMIO_UTILS_H__
