cmake_minimum_required(VERSION 3.10)
project(DMMIO_Example LANGUAGES CXX)

set(DMMIO_ENABLE_MPI OFF CACHE BOOL "Enable MPI support in dmmio")
add_subdirectory(distributed_mmio)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE dmmio)

