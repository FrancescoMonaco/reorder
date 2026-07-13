cmake_minimum_required(VERSION 3.10)
project(Distributed_MMIO_Example LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

set(DMMIO_ENABLE_MPI ON CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE MPI::MPI_CXX dmmio)

