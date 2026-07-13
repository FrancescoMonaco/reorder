cmake_minimum_required(VERSION 3.10)
project(Distributed_MMIO_Tests LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)

set(DMMIO_ENABLE_MPI ON CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)

if(NOT TARGET ccutils)
    set(CCUTILS_ENABLE_MPI ON CACHE BOOL "Enable MPI support in ccutils")
    add_subdirectory(ccutils)
endif()

add_executable(test_dmmio test_dmmio.cpp)
target_link_libraries(test_dmmio PRIVATE MPI::MPI_CXX dmmio ccutils::ccutils ccutils::ccutils_mpi)

